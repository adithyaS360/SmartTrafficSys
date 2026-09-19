"""
Camera capture and per-camera processing pipeline.

TWO CLASSES, TWO JOBS:

CameraStream - reads frames from a source in a background thread and always
holds only the LATEST frame.

  WHY THREADED: cv2.VideoCapture.read() is blocking, and OpenCV buffers frames
  internally. If detection takes 100ms but the camera produces a frame every
  33ms, a naive read-then-detect loop falls further behind every iteration -
  after a minute you are analysing footage from 40 seconds ago and confidently
  timing a traffic light on it. A reader thread that overwrites its single
  frame slot means we always detect on what the camera sees NOW, and simply
  skip the frames we had no time for. Dropping frames is correct here; lagging
  is not.

TrafficMonitor - owns one camera's full pipeline: stream -> detector ->
tracker -> line counter, and emits a TrafficSnapshot.

  WHY ONE PER CAMERA: each camera needs its own tracker state (ids must not be
  shared between intersections) and its own counting line. Making the unit of
  composition "one camera" is what makes multi-camera work fall out for free -
  the app just holds a dict of these.

The original script saved every frame to disk as a .jpg. At 30fps that is about
100GB a day per camera and nothing ever reads them back. We keep frames in
memory, extract numbers, and persist the numbers.
"""

import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.detector import DetectionResult, ObjectDetector
from src.tracker import CentroidTracker, LineCounter
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------

@dataclass
class TrafficSnapshot:
    """
    One moment of traffic state for one camera.

    This is the hand-off point to Phase 2: these fields map one-to-one onto the
    database table, and onto the feature vector the LSTM will be trained on.
    Designing it now, before the schema exists, keeps the schema honest - we
    store what we actually measure rather than what sounded good in a README.
    """

    camera_id: str
    timestamp: datetime
    vehicle_count: int          # distinct vehicles currently in frame
    pedestrian_count: int
    queue_length: int           # distinct vehicles currently stationary
    flow_rate: float            # vehicles per minute crossing the line
    avg_dwell_seconds: float    # mean time in view - proxy for waiting time
    avg_speed_kmh: Optional[float]   # None when the camera is uncalibrated
    total_crossings: int        # cumulative since this monitor started
    crossings_delta: int = 0    # crossings since the PREVIOUS snapshot
    class_breakdown: Dict[str, int] = field(default_factory=dict)
    processing_ms: float = 0.0  # how long this frame took - watch for drift

    # WHY BOTH total_crossings AND crossings_delta:
    # The cumulative total is what a live display wants. The delta is what the
    # database stores, because counts must be summable across a time bucket -
    # summing cumulative values would multiply-count, and a restart would reset
    # the counter and yield a negative difference. The delta is always the true
    # number of crossings in that interval, restart or not.

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


# ---------------------------------------------------------------------------
# Threaded capture
# ---------------------------------------------------------------------------

class CameraStream:
    """Background frame reader that always exposes the most recent frame."""

    def __init__(self,
                 source: Union[int, str],
                 camera_id: str = "camera",
                 reconnect_delay: float = 5.0,
                 loop_video_files: bool = True):
        """
        Args:
            source: one of
                - int (0, 1, ...)        USB webcam index
                - "rtsp://..."           IP camera stream
                - "path/to/video.mp4"    video file, for development
            camera_id: identifier used in logs and snapshots
            reconnect_delay: seconds to wait before retrying a dropped stream
            loop_video_files: restart video files when they end, so a short clip
                behaves like a continuous feed during development
        """
        self.source = self._normalise_source(source)
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self.loop_video_files = loop_video_files

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_time: float = 0.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._frames_read = 0
        self._frames_dropped = 0
        self._last_media_time = 0.0
        self._media_offset = 0.0

    @staticmethod
    def _normalise_source(source: Union[int, str]) -> Union[int, str]:
        """
        Accept "0" from YAML as webcam index 0, while leaving paths and URLs
        alone. YAML has no way to distinguish the two, so we do it here.
        """
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and not self.source.startswith(
            ("rtsp://", "http://", "https://")
        )

    def start(self) -> "CameraStream":
        """Open the source and begin reading in the background."""
        if self._running.is_set():
            return self

        self._open()
        self._running.set()
        self._thread = threading.Thread(
            target=self._reader_loop, name=f"camera-{self.camera_id}", daemon=True
        )
        self._thread.start()
        log.info("Camera '{}' started (source={})", self.camera_id, self.source)
        return self

    def _open(self) -> None:
        self._capture = cv2.VideoCapture(self.source)
        # Ask OpenCV to keep the smallest possible internal buffer. Not every
        # backend honours it, which is exactly why we also drop frames ourselves.
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera '{self.camera_id}' at source '{self.source}'. "
                f"For a webcam try index 0; for RTSP check the URL and credentials; "
                f"for a file check the path is relative to the project root."
            )

    def _reader_loop(self) -> None:
        while self._running.is_set():
            if self._capture is None or not self._capture.isOpened():
                self._reconnect()
                continue

            ok, frame = self._capture.read()

            if not ok:
                if self.is_file and self.loop_video_files:
                    # End of clip - rewind so development feeds never run dry.
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                log.warning("Camera '{}' read failed; reconnecting", self.camera_id)
                self._reconnect()
                continue

            # MEDIA TIME, not wall-clock. For a file this is the frame's own
            # position in the video; for a live stream the two are the same
            # thing. Downstream, speed and flow rate divide by elapsed time, and
            # that must be time elapsed IN THE FOOTAGE - detection on CPU is
            # slower than real time, so wall-clock would understate every speed.
            if self.is_file:
                media_time = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                # POS_MSEC restarts at 0 when we loop the file; keep the clock
                # monotonic so tracks are not given negative time deltas.
                if media_time < self._last_media_time:
                    self._media_offset += self._last_media_time
                self._last_media_time = media_time
                media_time += self._media_offset
            else:
                media_time = time.time()

            with self._lock:
                # If the previous frame was never consumed, we are dropping it.
                # Tracking that count tells us whether detection is keeping up.
                if self._frame is not None:
                    self._frames_dropped += 1
                self._frame = frame
                self._frame_time = media_time
                self._frames_read += 1

    def _reconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        time.sleep(self.reconnect_delay)
        if not self._running.is_set():
            return
        try:
            self._open()
            log.info("Camera '{}' reconnected", self.camera_id)
        except RuntimeError as exc:
            log.error("Camera '{}' reconnect failed: {}", self.camera_id, exc)

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Take the latest frame and its MEDIA timestamp, or (None, 0.0) if nothing
        new has arrived.

        Consuming clears the slot, so a caller that polls faster than the camera
        produces gets None rather than processing the same frame twice.
        """
        with self._lock:
            frame, ts = self._frame, self._frame_time
            self._frame = None
        return (frame.copy() if frame is not None else None), ts

    @property
    def stats(self) -> Dict[str, int]:
        return {"frames_read": self._frames_read, "frames_dropped": self._frames_dropped}

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        log.info("Camera '{}' stopped ({} read, {} dropped)",
                 self.camera_id, self._frames_read, self._frames_dropped)

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------

class TrafficMonitor:
    """Full pipeline for a single camera: capture -> detect -> track -> count."""

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self,
                 camera_config: Dict[str, Any],
                 detector: ObjectDetector,
                 counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 pixels_per_meter: Optional[float] = None):
        """
        Args:
            camera_config: one entry from the 'cameras' section of config.yaml
            detector: a detector instance. Passed in rather than constructed
                here so several cameras can share one loaded model - loading
                YOLO once and reusing it saves both memory and startup time.
            counting_line: ((x1,y1), (x2,y2)) across the road. Defaults to a
                horizontal line at mid-frame, which is a placeholder, not a
                sensible value - set it per camera once you see real footage.
            pixels_per_meter: calibration for speed. None = speeds omitted.
        """
        self.camera_id = camera_config["id"]
        self.name = camera_config.get("name", self.camera_id)
        self.direction = camera_config.get("direction", "unknown")
        self.roi = camera_config.get("roi")

        self.detector = detector
        self.stream = CameraStream(camera_config["source"], camera_id=self.camera_id)

        line = counting_line or camera_config.get("counting_line")
        if line is None:
            y = (self.roi or {}).get("y2", 1080) // 2
            x2 = (self.roi or {}).get("x2", 1920)
            line = ((0, y), (x2, y))
            log.warning(
                "Camera '{}' has no counting_line configured; using a placeholder "
                "across mid-frame. Flow rate will be meaningless until you set a "
                "real one in config.yaml.", self.camera_id
            )
        self.counter = LineCounter(tuple(line[0]), tuple(line[1]), name=self.camera_id)

        # Speed is only valid where pixels_per_meter was measured, and you
        # measure it at the counting line. So the speed zone is a band around
        # that line - by default half its length tall, which comfortably covers
        # the stretch of road where the scale still roughly holds. Vehicles
        # further up the frame are tracked and counted as normal; they just do
        # not contribute to the speed average. See
        # CentroidTracker.average_speed_kmh for why this is required, not tuning.
        band = camera_config.get("speed_zone_height")
        if band is None:
            band = int(abs(line[1][0] - line[0][0]) * 0.25) or 150
        line_y = (line[0][1] + line[1][1]) // 2
        line_x1, line_x2 = sorted((line[0][0], line[1][0]))
        self.speed_zone = (line_x1, line_y - band // 2, line_x2, line_y + band // 2)

        self.tracker = CentroidTracker(
            pixels_per_meter=pixels_per_meter,
            speed_zone=self.speed_zone,
        )

        self._latest: Optional[TrafficSnapshot] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._last_crossing_total = 0

    def start(self) -> "TrafficMonitor":
        self.stream.start()
        return self

    def _crop_roi(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Restrict detection to the region of interest.

        WHY CROP: a junction camera usually sees the road plus a footpath, a
        parking bay and the road behind. Detecting there inflates every count
        with vehicles that are not in this queue. It is also a straight speed
        win - YOLO on a quarter of the pixels is roughly four times faster.

        RETURNS THE OFFSET TOO, and that is not incidental. A cropped frame
        makes the detector report coordinates relative to the CROP, while the
        counting line and speed zone are defined on the FULL frame - that is
        what tools/calibrate.py shows you and what config.yaml stores. Without
        adding the offset back, every detection sits roi.y1 pixels too high,
        so the counting line is in the wrong place and flow rate is wrong.
        Nothing crashes; the numbers are just quietly incorrect, which is worse.
        """
        if not self.roi:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x1 = max(0, int(self.roi.get("x1", 0)))
        y1 = max(0, int(self.roi.get("y1", 0)))
        x2 = min(w, int(self.roi.get("x2", w)))
        y2 = min(h, int(self.roi.get("y2", h)))
        if x2 <= x1 or y2 <= y1:
            return frame, (0, 0)
        return frame[y1:y2, x1:x2], (x1, y1)

    def process_once(self, annotate: bool = False) -> Optional[TrafficSnapshot]:
        """
        Process the newest available frame.

        Returns:
            A TrafficSnapshot, or None if no new frame was ready. Returning None
            is normal - the caller simply polls again.
        """
        frame, media_time = self.stream.read()
        if frame is None:
            return None

        started = time.perf_counter()

        cropped, (off_x, off_y) = self._crop_roi(frame)
        result: DetectionResult = self.detector.detect(cropped)

        # Translate detections from crop space back to full-frame space, so they
        # share a coordinate system with the counting line and speed zone.
        if off_x or off_y:
            for det in result.detections:
                det.x1 += off_x; det.x2 += off_x
                det.y1 += off_y; det.y2 += off_y

        # Media time, not wall-clock: see CentroidTracker.update().
        tracks = self.tracker.update(result.detections, timestamp=media_time)
        self.counter.update(tracks, timestamp=media_time)

        active = [t for t in tracks.values() if t.disappeared == 0]
        breakdown: Dict[str, int] = {}
        for track in active:
            breakdown[track.class_name] = breakdown.get(track.class_name, 0) + 1

        total_crossings = self.counter.total
        # max(..., 0) guards the case where the counter was reset underneath us.
        crossings_delta = max(total_crossings - self._last_crossing_total, 0)
        self._last_crossing_total = total_crossings

        snapshot = TrafficSnapshot(
            camera_id=self.camera_id,
            timestamp=datetime.now(timezone.utc),
            vehicle_count=sum(1 for t in active if t.class_name in self.VEHICLE_CLASSES),
            pedestrian_count=sum(1 for t in active if t.class_name == "person"),
            queue_length=self.tracker.queue_length(self.VEHICLE_CLASSES),
            flow_rate=self.counter.flow_rate_per_minute(now=media_time),
            avg_dwell_seconds=self.tracker.average_dwell_seconds(self.VEHICLE_CLASSES),
            avg_speed_kmh=self.tracker.average_speed_kmh(self.VEHICLE_CLASSES),
            total_crossings=total_crossings,
            crossings_delta=crossings_delta,
            class_breakdown=breakdown,
            processing_ms=(time.perf_counter() - started) * 1000.0,
        )

        self._latest = snapshot
        if annotate:
            self._latest_annotated = self._annotate(frame, result)

        return snapshot

    def _annotate(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw tracks, ids and the counting line. Used by the dashboard later."""
        annotated = self.detector._annotate_frame(frame, result)
        cv2.line(annotated, self.counter.p1, self.counter.p2, (0, 165, 255), 2)
        for track in self.tracker.tracks.values():
            if track.disappeared:
                continue
            cv2.putText(annotated, f"#{track.object_id}", track.centroid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return annotated

    @property
    def latest(self) -> Optional[TrafficSnapshot]:
        return self._latest

    @property
    def latest_annotated(self) -> Optional[np.ndarray]:
        return self._latest_annotated

    def stop(self) -> None:
        self.stream.stop()


# ---------------------------------------------------------------------------
# Multi-camera coordinator
# ---------------------------------------------------------------------------

class MonitorFleet:
    """
    Runs every configured camera together.

    One shared detector, one monitor per camera. This is where the multi-camera
    requirement lands: adding a junction is a config.yaml edit, not a code change.
    """

    def __init__(self, config, detector: Optional[ObjectDetector] = None):
        self.config = config
        self.detector = detector or ObjectDetector(
            model_weights=config.get("model.weights_path", "yolov8n.pt"),
            confidence_threshold=config.get("model.confidence_threshold", 0.5),
            iou_threshold=config.get("model.iou_threshold", 0.45),
            device=config.get("model.device", "cpu"),
            camera_id="shared",
        )
        self.monitors: Dict[str, TrafficMonitor] = {}

        for name, cam in config.cameras.items():
            monitor = TrafficMonitor(
                camera_config=cam,
                detector=self.detector,
                pixels_per_meter=cam.get("pixels_per_meter"),
            )
            self.monitors[monitor.camera_id] = monitor
            log.info("Registered camera '{}' ({})", monitor.camera_id, name)

    def start(self) -> "MonitorFleet":
        for monitor in self.monitors.values():
            monitor.start()
        return self

    def poll(self) -> List[TrafficSnapshot]:
        """Process one frame from each camera. Returns whatever was ready."""
        snapshots = []
        for monitor in self.monitors.values():
            snap = monitor.process_once()
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def snapshot_all(self) -> Dict[str, Optional[TrafficSnapshot]]:
        """Most recent snapshot per camera, without processing a new frame."""
        return {cam_id: m.latest for cam_id, m in self.monitors.items()}

    def stop(self) -> None:
        for monitor in self.monitors.values():
            monitor.stop()
        self.detector.cleanup()


if __name__ == "__main__":
    # Smoke test against a webcam. Run from the project root:
    #     python -m src.data_collector
    from src.utils.logger import setup_logging

    setup_logging(level="INFO")

    detector = ObjectDetector(model_weights="yolov8n.pt", camera_id="test")
    monitor = TrafficMonitor(
        camera_config={"id": "test_cam", "source": 0, "name": "Webcam"},
        detector=detector,
    ).start()

    try:
        while True:
            snap = monitor.process_once()
            if snap:
                speed = f"{snap.avg_speed_kmh:.1f} km/h" if snap.avg_speed_kmh else "uncalibrated"
                print(
                    f"[{snap.camera_id}] vehicles={snap.vehicle_count} "
                    f"queue={snap.queue_length} flow={snap.flow_rate:.1f}/min "
                    f"dwell={snap.avg_dwell_seconds:.1f}s speed={speed} "
                    f"({snap.processing_ms:.0f}ms)"
                )
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        detector.cleanup()
