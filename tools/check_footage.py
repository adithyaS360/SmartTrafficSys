"""
Assess whether a traffic clip is usable for this project, before you build on it.

    python tools/check_footage.py data/my_junction.mp4

Answers the four questions that decide whether a clip is worth calibrating, and
says which ones it fails and why:

  1. IS THE CAMERA STATIC?  Measured, not eyeballed, with phase correlation
     between frames. A drifting camera makes every vehicle appear to move, so
     queue length reads zero forever and speeds are meaningless. This is the
     single most common reason a promising-looking clip is unusable, and the
     drift is often too slow to notice by watching.

  2. DOES ANYTHING QUEUE?  Detects vehicles and checks whether any of them stay
     roughly still for several seconds. A motorway clip passes every other test
     and is still useless for signal control, because nothing ever stops: there
     is no queue to clear and no delay to reduce. The sample footage this
     project ships with fails exactly here.

  3. DOES DETECTION WORK ON IT?  Runs YOLO over a sample and reports how many
     vehicles it finds, at what confidence, and in which classes. On Indian
     footage expect auto-rickshaws to be missed or misclassified - COCO has no
     class for them - and this is where you will see that.

  4. WHERE SHOULD THE COUNTING LINE GO?  Suggests a position from where
     detections actually cluster, as a starting point for tools/calibrate.py.

Use it on every candidate clip. Two minutes here saves an afternoon of
calibrating footage that was never going to work.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STATIC_DRIFT_LIMIT = 4.0        # px of cumulative drift, measured on a 480px proxy
STATIONARY_PX_PER_SEC = 6.0     # below this a tracked object counts as stopped
QUEUE_DWELL_SECONDS = 3.0       # stopped this long = genuinely queuing


def probe(path: str, max_frames: int, sample_every: int, weights: str) -> dict:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open '{path}'. Check the path and the codec.")

    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0

    from src.detector import ObjectDetector
    from src.tracker import CentroidTracker
    detector = ObjectDetector(model_weights=weights, confidence_threshold=0.35,
                              camera_id="check")
    tracker = CentroidTracker(max_disappeared=25, max_distance=90)

    shifts, proxy_prev, proxy_first = [], None, None
    displacements = []
    classes, confidences = Counter(), []
    stationary_spans = defaultdict(float)
    per_frame_counts, centroids = [], []
    index = 0

    while index < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        # --- camera motion, on a cheap downscaled proxy ---
        #
        # Displacement is measured against the FIRST frame, not by summing
        # frame-to-frame shifts. Phase correlation carries a small sub-pixel
        # bias, and summing four hundred of them accumulates that bias into
        # tens of pixels of phantom drift - the first version of this check
        # reported a verified-static clip as moving for exactly that reason.
        # Correlating against a fixed reference measures the real displacement
        # directly and does not compound.
        proxy = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if proxy_first is None:
            proxy_first = proxy
        else:
            (dx, dy), _ = cv2.phaseCorrelate(proxy_first, proxy)
            displacements.append(float(np.hypot(dx, dy)))
        if proxy_prev is not None:
            (jx, jy), _ = cv2.phaseCorrelate(proxy_prev, proxy)
            shifts.append(float(np.hypot(jx, jy)))     # per-frame jitter only
        proxy_prev = proxy

        # --- detection and tracking, on every Nth frame ---
        if index % sample_every == 0:
            result = detector.detect(frame)
            media_time = index / info["fps"]
            tracks = tracker.update(result.detections, timestamp=media_time)

            per_frame_counts.append(result.vehicle_count)
            for det in result.detections:
                classes[det.class_name] += 1
                confidences.append(det.confidence)

            step = sample_every / info["fps"]
            for track in tracks.values():
                if track.disappeared:
                    continue
                centroids.append(track.centroid)
                if track.speed_px_per_sec() < STATIONARY_PX_PER_SEC and len(track.history) > 2:
                    stationary_spans[track.object_id] += step

        index += 1

    capture.release()
    detector.cleanup()

    return {
        **info,
        "sampled": index,
        "drift": float(np.max(displacements)) if displacements else 0.0,
        "drift_final": float(displacements[-1]) if displacements else 0.0,
        "per_frame_shift": float(np.mean(shifts)) if shifts else 0.0,
        "classes": classes,
        "confidences": confidences,
        "counts": per_frame_counts,
        "queued": [v for v in stationary_spans.values() if v >= QUEUE_DWELL_SECONDS],
        "tracks_seen": tracker._next_id,
        "centroids": centroids,
    }


def report(path: str, r: dict) -> int:
    print(f"\n{'=' * 64}\n  {Path(path).name}\n{'=' * 64}")
    print(f"  {r['width']}x{r['height']}  {r['fps']:.1f} fps  "
          f"{r['duration']:.1f}s  ({r['frames']} frames, {r['sampled']} examined)")

    verdicts = []

    # --- 1. static camera ---
    static = r["drift"] <= STATIC_DRIFT_LIMIT
    print(f"\n  1. CAMERA MOTION")
    print(f"     max displacement from first frame {r['drift']:.2f} px "
          f"(ending at {r['drift_final']:.2f} px), per-frame jitter "
          f"{r['per_frame_shift']:.3f} px")
    print(f"     measured on a 480px-wide proxy, so multiply by "
          f"{r['width'] / 480:.1f} for full-resolution pixels")
    if static:
        print(f"     [PASS] static - safe for tracking")
    else:
        print(f"     [FAIL] THE CAMERA MOVES. Tracking cannot work on this clip: every")
        print(f"            vehicle registers as moving, so queue length stays at zero and")
        print(f"            speeds are meaningless. Drone and handheld footage fails here.")
    verdicts.append(static)

    # --- 2. queuing ---
    queued = r["queued"]
    has_queue = len(queued) >= 2
    print(f"\n  2. QUEUING")
    print(f"     {len(queued)} vehicle(s) stationary for >= {QUEUE_DWELL_SECONDS:g}s"
          + (f", longest {max(queued):.1f}s" if queued else ""))
    if has_queue:
        print(f"     [PASS] vehicles stop and wait - there is a queue to manage")
    else:
        print(f"     [FAIL] NOTHING QUEUES. Detection and counting will work, but this")
        print(f"            clip cannot demonstrate signal control: with no stopped")
        print(f"            traffic there is no delay to reduce and nothing for the")
        print(f"            controller to act on. Free-flowing motorway footage looks")
        print(f"            like this. You need a junction where traffic waits.")
    verdicts.append(has_queue)

    # --- 3. detection quality ---
    counts, confs = r["counts"], r["confidences"]
    mean_count = float(np.mean(counts)) if counts else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    detects = mean_count >= 1.0
    print(f"\n  3. DETECTION")
    print(f"     mean {mean_count:.1f} vehicles/frame, {r['tracks_seen']} distinct tracks, "
          f"mean confidence {mean_conf:.2f}")
    if r["classes"]:
        top = ", ".join(f"{k} {v}" for k, v in r["classes"].most_common(6))
        print(f"     classes: {top}")
    if detects:
        print(f"     [PASS] YOLO finds vehicles reliably")
    else:
        print(f"     [WARN] almost nothing detected. Either the clip is genuinely empty,")
        print(f"            or the vehicles are too small in frame - try footage shot")
        print(f"            closer to the junction, or a larger model (yolov8s/m).")
    verdicts.append(detects)

    # Indian-traffic note, triggered by what is actually in the frame
    two_wheelers = r["classes"].get("motorcycle", 0) + r["classes"].get("bicycle", 0)
    if two_wheelers > sum(r["classes"].values()) * 0.25:
        print(f"     NOTE: two-wheelers are {two_wheelers / max(sum(r['classes'].values()),1):.0%} "
              f"of detections. COCO has no auto-rickshaw class, so autos here are")
        print(f"           being counted as car/truck or missed entirely. Worth measuring")
        print(f"           against a hand count - that gap is a real finding.")

    # --- 4. suggested counting line ---
    print(f"\n  4. SUGGESTED COUNTING LINE")
    if r["centroids"]:
        ys = np.array([c[1] for c in r["centroids"]])
        xs = np.array([c[0] for c in r["centroids"]])
        y = int(np.median(ys))
        x1, x2 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
        print(f"     detections cluster around y={y}; try")
        print(f"       counting_line: [[{x1}, {y}], [{x2}, {y}]]")
        print(f"     Then refine it with: python tools/calibrate.py {path}")
    else:
        print(f"     not enough detections to suggest one")

    # --- verdict ---
    print(f"\n{'-' * 64}")
    if all(verdicts):
        print("  VERDICT: USABLE. Calibrate it and run --mode live.")
        code = 0
    elif static and detects and not has_queue:
        print("  VERDICT: PARTIALLY USABLE. Good for validating detection, tracking and")
        print("           flow counting. Cannot demonstrate signal control - use")
        print("           'python app.py --mode simulated' for that until you have")
        print("           footage of a junction where traffic actually queues.")
        code = 1
    else:
        print("  VERDICT: NOT USABLE. See the failures above.")
        code = 2
    print(f"{'-' * 64}\n")
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="path to the clip to assess")
    parser.add_argument("--frames", type=int, default=400, help="frames to examine")
    parser.add_argument("--every", type=int, default=4, help="run detection every Nth frame")
    parser.add_argument("--weights", default="yolov8n.pt")
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"File not found: {args.video}")

    from src.utils.logger import setup_logging
    setup_logging(level="WARNING")
    sys.exit(report(args.video, probe(args.video, args.frames, args.every, args.weights)))


if __name__ == "__main__":
    main()
