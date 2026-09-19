"""
Multi-object tracking and line counting.

WHY THIS MODULE EXISTS - the single most important idea in the project:

A detector answers "what is in THIS frame?". It has no memory. If one car waits
at a red light for 20 seconds at 30fps, the detector reports a car 600 times.
Summing those gives 600 "vehicles", which is nonsense.

A tracker answers "which objects are these, over time?". It assigns a stable id
to each vehicle, so we can ask the questions that actually drive a traffic
system:

  - FLOW RATE    how many DISTINCT vehicles crossed the stop line per minute
                 -> this is the LSTM's input feature
  - QUEUE LENGTH how many distinct vehicles are currently stationary
                 -> this is what the signal controller reacts to
  - DWELL TIME   how long has each vehicle been waiting
                 -> this is the metric we are trying to minimise
  - SPEED        pixel displacement over time, scaled to real units

None of those are computable from per-frame detections alone.

ALGORITHM: centroid tracking with optimal assignment.
Each frame we build a cost matrix of distances between existing tracks and new
detections, then solve it with the Hungarian algorithm (scipy's
linear_sum_assignment). That is better than greedy nearest-neighbour, which
mis-assigns when two vehicles pass close to each other - greedy commits to the
first match it finds, the Hungarian solution minimises total error across all
pairs at once.

LIMITS, stated honestly: centroid tracking is cheap and dependency-light, but it
loses identity through long occlusions (a bus hiding a car for 2s). If that
turns out to matter on real footage, the upgrade path is a Kalman filter plus
appearance embeddings (i.e. ByteTrack / DeepSORT). The interface below would not
change - only the internals of update().
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrackedObject:
    """One object followed across frames."""

    object_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2 - most recent
    first_seen: float                         # unix timestamp
    last_seen: float
    # (timestamp, cx, cy) history, newest last. Bounded so memory stays flat.
    history: Deque[Tuple[float, int, int]] = field(default_factory=lambda: deque(maxlen=30))
    disappeared: int = 0                      # consecutive frames with no match
    counted: bool = False                     # has it already crossed the counting line
    confidence: float = 0.0

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def age_seconds(self) -> float:
        """How long we have been tracking this object. This is the dwell time."""
        return self.last_seen - self.first_seen

    def speed_px_per_sec(self) -> float:
        """
        Average speed in pixels/second over the tracked history.

        Measured end-to-end across the whole history rather than frame-to-frame,
        because single-frame deltas are dominated by bounding-box jitter - a box
        that wobbles 3px between frames at 30fps reads as 90 px/s of phantom
        motion. Averaging over ~1s of history cancels that out.
        """
        if len(self.history) < 2:
            return 0.0
        t0, x0, y0 = self.history[0]
        t1, x1, y1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return float(np.hypot(x1 - x0, y1 - y0) / dt)

    def speed_kmh(self, pixels_per_meter: Optional[float]) -> Optional[float]:
        """
        Speed in km/h, or None if the camera has not been calibrated.

        Returning None rather than a number is deliberate. Pixel speed cannot be
        converted to real speed without knowing the scale, and a fabricated
        figure in a portfolio project is worse than an honest gap. Calibrate by
        measuring a known real-world distance in the frame (lane width is ~3.5m
        in India) and setting pixels_per_meter for that camera in config.yaml.
        """
        if not pixels_per_meter:
            return None
        return self.speed_px_per_sec() / pixels_per_meter * 3.6

    def is_stationary(self, threshold_px_per_sec: float = 5.0) -> bool:
        """Roughly stationary - used for queue length."""
        return self.speed_px_per_sec() < threshold_px_per_sec


class CentroidTracker:
    """
    Assigns stable ids to detections across frames.

    Usage:
        tracker = CentroidTracker()
        tracks = tracker.update(detections)   # call once per frame
    """

    def __init__(self,
                 max_disappeared: int = 30,
                 max_distance: float = 80.0,
                 pixels_per_meter: Optional[float] = None,
                 speed_zone: Optional[Tuple[int, int, int, int]] = None):
        """
        Args:
            max_disappeared: drop a track after this many frames without a match.
                At 30fps, 30 frames = 1 second of tolerance for a missed detection
                or a brief occlusion.
            max_distance: maximum centroid movement, in pixels, that still counts
                as the same object between frames. Too high and two vehicles swap
                ids; too low and one fast vehicle becomes two tracks. Scale it
                with resolution and expected speed.
            pixels_per_meter: calibration for real-world speed. None = speeds
                reported in pixels only.
            speed_zone: (x1, y1, x2, y2) band in which pixels_per_meter is valid
                - normally a strip around the counting line where the scale was
                measured. Speed is averaged over tracks inside it only. Without
                it, average_speed_kmh() returns None. See that method for why.
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.pixels_per_meter = pixels_per_meter
        self.speed_zone = speed_zone

        self._next_id = 0
        self._tracks: Dict[int, TrackedObject] = {}

    # ---- public API ---------------------------------------------------

    @property
    def tracks(self) -> Dict[int, TrackedObject]:
        """Currently active tracks, keyed by object id."""
        return self._tracks

    def update(self,
               detections: List[Detection],
               timestamp: Optional[float] = None) -> Dict[int, TrackedObject]:
        """
        Advance the tracker by one frame.

        Args:
            detections: this frame's detections, from ObjectDetector.detect()
            timestamp: MEDIA time for this frame, in seconds. Pass the video's
                own timestamp when reading a file; omit it for a live camera,
                where wall-clock time is the media clock.

                WHY THIS MATTERS: speed is displacement over time, and the time
                must be the time that elapsed IN THE FOOTAGE. Detection on CPU
                runs at about 10fps on 720p, so a 25fps video is processed at
                0.4x real time. Using wall-clock time then divides real
                displacement by 2.5x too much elapsed time, and every speed and
                dwell figure is wrong by that ratio - silently, with no error.

        Returns:
            The active tracks after this frame.
        """
        now = time.time() if timestamp is None else timestamp

        # No detections: age every track, retire the ones that have been gone
        # too long. We do NOT clear everything - a one-frame detector miss should
        # not restart every id.
        if not detections:
            for object_id in list(self._tracks.keys()):
                self._tracks[object_id].disappeared += 1
                if self._tracks[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)
            return self._tracks

        # First frame, or all tracks expired: everything is new.
        if not self._tracks:
            for det in detections:
                self._register(det, now)
            return self._tracks

        # Match existing tracks to new detections.
        track_ids = list(self._tracks.keys())
        track_centroids = np.array([self._tracks[t].centroid for t in track_ids], dtype=float)
        det_centroids = np.array(
            [((d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2) for d in detections], dtype=float
        )

        # Pairwise euclidean distance: rows = tracks, cols = detections.
        cost = np.linalg.norm(track_centroids[:, None, :] - det_centroids[None, :, :], axis=2)

        # Penalise cross-class matches heavily so a 'person' track never inherits
        # a 'car' detection just because they are close together on screen.
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                if self._tracks[tid].class_name != det.class_name:
                    cost[i, j] += 1e6

        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_idx, col_idx):
            # Reject assignments the solver made only because it had to.
            if cost[r, c] > self.max_distance:
                continue
            object_id = track_ids[r]
            self._update_track(object_id, detections[c], now)
            matched_tracks.add(object_id)
            matched_dets.add(c)

        # Unmatched tracks: age them out.
        for i, object_id in enumerate(track_ids):
            if object_id not in matched_tracks:
                self._tracks[object_id].disappeared += 1
                if self._tracks[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)

        # Unmatched detections: new objects entering the scene.
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._register(det, now)

        return self._tracks

    # ---- derived traffic metrics --------------------------------------

    def queue_length(self, class_filter: Optional[set] = None) -> int:
        """Count of distinct stationary objects - i.e. vehicles waiting."""
        return sum(
            1 for t in self._tracks.values()
            if t.disappeared == 0
            and t.is_stationary()
            and (class_filter is None or t.class_name in class_filter)
        )

    def average_dwell_seconds(self, class_filter: Optional[set] = None) -> float:
        """Mean time tracked objects have been in view. Proxy for waiting time."""
        ages = [
            t.age_seconds for t in self._tracks.values()
            if t.disappeared == 0 and (class_filter is None or t.class_name in class_filter)
        ]
        return float(np.mean(ages)) if ages else 0.0

    def average_speed_kmh(self, class_filter: Optional[set] = None) -> Optional[float]:
        """
        Mean speed of moving objects, or None if the camera is uncalibrated.

        ONLY TRACKS INSIDE speed_zone ARE INCLUDED, and here is why that is not
        an optimisation but a correctness requirement:

        pixels_per_meter is measured at ONE place in the frame. In a perspective
        view down a road, a vehicle near the horizon covers a handful of pixels
        per second while doing the same 100 km/h as one in the foreground
        covering hundreds. Averaging across the whole frame with a single scale
        factor mixes both and produces a number that is not the speed of
        anything - measured on real motorway footage it reported 4.7 km/h.

        Restricting the average to a band around the place where the scale was
        actually measured makes the figure meaningful. Without a speed_zone we
        return None rather than a confidently wrong number.
        """
        if not self.pixels_per_meter:
            return None
        if self.speed_zone is None:
            return None

        x1, y1, x2, y2 = self.speed_zone
        speeds = []
        for t in self._tracks.values():
            if t.disappeared or (class_filter is not None and t.class_name not in class_filter):
                continue
            cx, cy = t.centroid
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
            s = t.speed_kmh(self.pixels_per_meter)
            if s is not None and s > 1.0:
                speeds.append(s)
        return float(np.mean(speeds)) if speeds else 0.0

    # ---- internals ----------------------------------------------------

    def _register(self, det: Detection, now: float) -> None:
        obj = TrackedObject(
            object_id=self._next_id,
            class_name=det.class_name,
            bbox=(det.x1, det.y1, det.x2, det.y2),
            first_seen=now,
            last_seen=now,
            confidence=det.confidence,
        )
        obj.history.append((now, *obj.centroid))
        self._tracks[self._next_id] = obj
        self._next_id += 1

    def _update_track(self, object_id: int, det: Detection, now: float) -> None:
        track = self._tracks[object_id]
        track.bbox = (det.x1, det.y1, det.x2, det.y2)
        track.confidence = det.confidence
        track.last_seen = now
        track.disappeared = 0
        track.history.append((now, *track.centroid))

    def _deregister(self, object_id: int) -> None:
        track = self._tracks.pop(object_id, None)
        if track:
            log.debug("Track {} ({}) retired after {:.1f}s",
                      object_id, track.class_name, track.age_seconds)


class LineCounter:
    """
    Counts objects crossing a virtual line - this is what produces FLOW RATE.

    Place the line across the road at the stop line. Each tracked object is
    counted exactly once, the first time its centroid crosses from one side to
    the other, and the crossing direction tells you which way it was going.

    The line is defined by two points. For each track we compute the signed
    PERPENDICULAR DISTANCE from that line, in pixels: the magnitude says how far
    away the object is, the sign says which side it is on. A sign change between
    two consecutive observations means the object crossed.

    TWO GUARDS, both learned the hard way:

    1. DEAD ZONE. An object sitting exactly on the line has distance 0, which is
       neither side. The first version of this class tested `previous < 0 < side`
       and silently counted nothing at all, because a vehicle stepping
       -20 -> 0 -> +20 satisfies neither comparison at either step. Now anything
       within `dead_zone_px` of the line is skipped WITHOUT updating the stored
       side, so the comparison happens between the last committed side before
       the line and the first committed side after it.

    2. COOLDOWN. Bounding-box centroids jitter by a few pixels frame to frame.
       A vehicle stopped just past the line could otherwise flip sign repeatedly
       and be counted a dozen times. The same object cannot be counted twice
       within `recount_cooldown` seconds.
    """

    def __init__(self,
                 p1: Tuple[int, int],
                 p2: Tuple[int, int],
                 name: str = "counter",
                 dead_zone_px: float = 3.0,
                 recount_cooldown: float = 1.0):
        self.p1 = p1
        self.p2 = p2
        self.name = name
        self.dead_zone_px = dead_zone_px
        self.recount_cooldown = recount_cooldown
        self.count_forward = 0     # crossings from the negative side to positive
        self.count_backward = 0
        self._last_side: Dict[int, float] = {}
        self._last_counted: Dict[int, float] = {}
        self._crossing_times: Deque[float] = deque(maxlen=1000)
        # Precomputed so _signed_distance stays cheap - it runs per track per frame.
        self._length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) or 1.0

    def _signed_distance(self, point: Tuple[int, int]) -> float:
        """Perpendicular distance from the line in pixels; sign indicates the side."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross / self._length

    def update(self,
               tracks: Dict[int, TrackedObject],
               timestamp: Optional[float] = None) -> int:
        """
        Check all tracks for crossings. Call once per frame, after tracker.update().

        Args:
            tracks: the dict returned by CentroidTracker.update()
            timestamp: media time, as for CentroidTracker.update(). Flow rate is
                per minute of FOOTAGE, so it must use the same clock.

        Returns:
            Number of new crossings detected in this frame.
        """
        new_crossings = 0
        now = time.time() if timestamp is None else timestamp

        for object_id, track in tracks.items():
            if track.disappeared > 0:
                continue

            distance = self._signed_distance(track.centroid)

            # Guard 1: too close to call. Leave the stored side untouched.
            if abs(distance) < self.dead_zone_px:
                continue

            side = 1.0 if distance > 0 else -1.0
            previous = self._last_side.get(object_id)
            self._last_side[object_id] = side

            if previous is None or previous == side:
                continue

            # Guard 2: the same object flipping sides again immediately is jitter.
            if now - self._last_counted.get(object_id, 0.0) < self.recount_cooldown:
                continue

            if side > 0:
                self.count_forward += 1
            else:
                self.count_backward += 1

            self._last_counted[object_id] = now
            self._crossing_times.append(now)
            track.counted = True
            new_crossings += 1

        # Forget ids that no longer exist so these dicts do not grow forever.
        stale = set(self._last_side) - set(tracks)
        for object_id in stale:
            self._last_side.pop(object_id, None)
            self._last_counted.pop(object_id, None)

        return new_crossings

    def flow_rate_per_minute(self,
                             window_seconds: float = 60.0,
                             now: Optional[float] = None) -> float:
        """
        Vehicles per minute over a trailing window.

        THIS IS THE HEADLINE NUMBER: it is what the dashboard shows, what the
        signal controller reacts to, and what the LSTM is trained to predict.

        Args:
            window_seconds: length of the trailing window
            now: current media time. Must come from the same clock as update(),
                or the window covers the wrong span and the rate is wrong.
        """
        current = time.time() if now is None else now
        cutoff = current - window_seconds
        recent = sum(1 for t in self._crossing_times if t >= cutoff)
        return recent / window_seconds * 60.0

    @property
    def total(self) -> int:
        return self.count_forward + self.count_backward

    def reset(self) -> None:
        self.count_forward = 0
        self.count_backward = 0
        self._last_side.clear()
        self._last_counted.clear()
        self._crossing_times.clear()
