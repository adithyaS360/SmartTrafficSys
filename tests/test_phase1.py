"""
Verification of Phase 1 logic: tracker identity, queue length, line counting,
flow rate, and config loading/validation.

cv2 and ultralytics are stubbed so this runs without the heavy vision stack -
none of the logic under test touches them.
"""
import sys, types, time, os
from pathlib import Path

BUILD = Path(__file__).resolve().parents[1]     # project root
sys.path.insert(0, str(BUILD))

# --- stub heavy deps -------------------------------------------------------
cv2 = types.ModuleType("cv2")
cv2.dnn = types.SimpleNamespace()
cv2.getTickCount = lambda: 0
cv2.getTickFrequency = lambda: 1
for name in ("rectangle", "putText", "line", "imshow", "waitKey", "destroyAllWindows"):
    setattr(cv2, name, lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0
cv2.VideoCapture = object
cv2.CAP_PROP_BUFFERSIZE = 38
cv2.CAP_PROP_POS_FRAMES = 1
sys.modules["cv2"] = cv2

ultra = types.ModuleType("ultralytics")
ultra.YOLO = object
sys.modules["ultralytics"] = ultra

from src.detector import Detection
from src.tracker import CentroidTracker, LineCounter

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))

def det(cx, cy, cls="car", w=40, h=40, conf=0.9):
    return Detection(class_id=2, class_name=cls, confidence=conf,
                     x1=cx - w // 2, y1=cy - h // 2, x2=cx + w // 2, y2=cy + h // 2)

print("\n=== 1. Identity is stable across frames ===")
t = CentroidTracker()
t.update([det(100, 100), det(300, 100)])
ids_first = set(t.tracks)
for step in range(1, 6):
    t.update([det(100 + step * 10, 100), det(300 + step * 10, 100)])
check("two objects tracked, no id churn over 6 frames",
      set(t.tracks) == ids_first and len(t.tracks) == 2,
      f"ids={sorted(t.tracks)}")

print("\n=== 2. A stationary car is ONE vehicle, not one per frame ===")
t2 = CentroidTracker()
for _ in range(60):                      # 60 frames of a parked car
    t2.update([det(200, 200)])
check("60 frames of one stationary car -> 1 track", len(t2.tracks) == 1,
      f"tracks={len(t2.tracks)} (naive per-frame counting would say 60)")

print("\n=== 3. Queue length counts stationary vehicles only ===")
t3 = CentroidTracker()
for step in range(12):
    t3.update([
        det(200, 200),                    # parked
        det(400, 200),                    # parked
        det(100 + step * 60, 400),        # moving fast
    ])
    time.sleep(0.01)
q = t3.queue_length({"car"})
check("2 of 3 vehicles counted as queued", q == 2, f"queue_length={q}")

print("\n=== 4. New object registers, vanished object expires ===")
t4 = CentroidTracker(max_disappeared=3)
t4.update([det(100, 100)])
t4.update([det(100, 100), det(500, 500)])
check("new detection creates a second track", len(t4.tracks) == 2, f"tracks={len(t4.tracks)}")
for _ in range(5):
    t4.update([det(100, 100)])
check("track expires after max_disappeared frames", len(t4.tracks) == 1, f"tracks={len(t4.tracks)}")

print("\n=== 5. Classes do not swap identity ===")
t5 = CentroidTracker()
t5.update([det(100, 100, "car"), det(120, 100, "person")])
before = {i: tr.class_name for i, tr in t5.tracks.items()}
for _ in range(4):
    t5.update([det(100, 100, "car"), det(120, 100, "person")])
after = {i: tr.class_name for i, tr in t5.tracks.items()}
check("adjacent car and person keep their own ids", before == after, f"{before} -> {after}")

print("\n=== 6. Line counting: each vehicle counted exactly once ===")
t6 = CentroidTracker()
lc = LineCounter((0, 300), (800, 300))
for y in range(100, 500, 20):            # one car drives down through y=300
    lc.update(t6.update([det(400, y)]))
check("single crossing counted once", lc.total == 1,
      f"total={lc.total} fwd={lc.count_forward} bwd={lc.count_backward}")

print("\n=== 7. Line counting: direction is distinguished ===")
# One tracker, two vehicles passing at the same time in opposite directions.
t7 = CentroidTracker()
lc2 = LineCounter((0, 300), (800, 300))
for step, y in enumerate(range(100, 500, 20)):
    lc2.update(t7.update([det(200, y),            # southbound
                          det(600, 480 - step * 20)]))  # northbound
check("exactly one crossing each way",
      lc2.count_forward == 1 and lc2.count_backward == 1,
      f"fwd={lc2.count_forward} bwd={lc2.count_backward}")

print("\n=== 8. A car idling ON the line is not counted repeatedly ===")
t8 = CentroidTracker()
lc3 = LineCounter((0, 300), (800, 300))
for _ in range(40):
    lc3.update(t8.update([det(400, 299)]))
check("40 frames parked beside the line -> 0 crossings", lc3.total == 0, f"total={lc3.total}")

print("\n=== 9. Flow rate reports vehicles per minute ===")
# Three cars in three lanes crossing together, tracked by a single tracker.
t9 = CentroidTracker()
lc4 = LineCounter((0, 300), (800, 300))
for y in range(100, 500, 20):
    lc4.update(t9.update([det(100, y), det(400, y), det(700, y)]))
rate = lc4.flow_rate_per_minute(window_seconds=60)
check("3 crossings -> flow rate 3.0/min", lc4.total == 3 and abs(rate - 3.0) < 0.01,
      f"total={lc4.total} rate={rate:.2f}/min")

print("\n=== 9b. Jitter at the line does not inflate the count ===")
t9b = CentroidTracker()
lc5 = LineCounter((0, 300), (800, 300))
for y in range(100, 300, 20):                      # approach
    lc5.update(t9b.update([det(400, y)]))
for _ in range(30):                                # stop just past it, wobbling
    for offset in (306, 310, 305, 309):
        lc5.update(t9b.update([det(400, offset)]))
check("one vehicle crossing then idling -> exactly 1 count", lc5.total == 1,
      f"total={lc5.total}")

print("\n=== 10. Speed is None until the camera is calibrated ===")
t10 = CentroidTracker(pixels_per_meter=None)
for step in range(10):
    t10.update([det(100 + step * 20, 100)])
    time.sleep(0.01)
check("uncalibrated camera reports None, not a fabricated number",
      t10.average_speed_kmh({"car"}) is None)

# A calibrated camera with no speed_zone also reports None: pixels_per_meter is
# only valid where it was measured, so averaging across the whole frame mixes
# depths and is meaningless. See CentroidTracker.average_speed_kmh.
t10b = CentroidTracker(pixels_per_meter=10.0, speed_zone=None)
for step in range(10):
    t10b.update([det(100 + step * 20, 100)], timestamp=step * 0.04)
check("calibrated but no speed zone -> None, not a cross-depth average",
      t10b.average_speed_kmh({"car"}) is None)

# With an explicit media clock, speed is exact and checkable: 20px per 0.04s
# = 500 px/s; at 10 px/m that is 50 m/s = 180 km/h.
t11 = CentroidTracker(pixels_per_meter=10.0, speed_zone=(0, 0, 2000, 2000))
for step in range(10):
    t11.update([det(100 + step * 20, 100)], timestamp=step * 0.04)
s = t11.average_speed_kmh({"car"})
check("media-time clock gives an exact, predictable speed",
      s is not None and abs(s - 180.0) < 1.0, f"{s:.1f} km/h (expected 180.0)")

# The same motion timed by wall-clock would be wrong - this is the bug that
# reported motorway traffic at 4.7 km/h.
t11b = CentroidTracker(pixels_per_meter=10.0, speed_zone=(0, 0, 2000, 2000))
for step in range(10):
    t11b.update([det(100 + step * 20, 100)])   # no timestamp -> wall-clock
    time.sleep(0.01)
s_wall = t11b.average_speed_kmh({"car"})
check("wall-clock on a faster-than-realtime source overstates speed",
      s_wall is not None and s_wall > 200, f"{s_wall:.1f} km/h vs true 180.0")

print("\n=== 11. Config: env substitution and validation ===")
os.environ["DB_PASSWORD"] = "s3cret"
from src.utils.config_loader import load_config, ConfigError
cfg = load_config(str(BUILD / "config" / "config.yaml"), env_file=None)
check("${DB_PASSWORD} resolved from environment",
      cfg.get("database.password") == "s3cret")
check("${DB_HOST:-localhost} falls back to its default",
      cfg.get("database.host") == "localhost")
check("camera block loaded from config", len(cfg.cameras) >= 1, f"{list(cfg.cameras)}")
check("calibrated values are real, not placeholders",
      cfg.get("cameras.highway_bridge.pixels_per_meter") == 7.5
      and cfg.get("cameras.highway_bridge.counting_line") == [[200, 450], [1180, 450]])
# config.yaml ships with database.type: sqlite for local development.
# The postgres URL form is covered in tests/test_phase2.py, case 14.
check("database_url built correctly",
      cfg.database_url() == "sqlite:///data/traffic.db", cfg.database_url())
check("dotted access works", cfg.get("model.confidence_threshold") == 0.5)

print("\n=== 12. Config: bad input fails loudly at startup ===")
from src.utils.config_loader import Config
try:
    Config({"database": {}, "model": {}, "cameras": {}, "traffic_signals": {}})
    check("empty cameras rejected", False)
except ConfigError as e:
    check("empty cameras rejected", True, str(e)[:60])
try:
    Config({"database": {}, "model": {},
            "cameras": {"a": {"id": "cam_1", "source": 0}},
            "traffic_signals": {"s": {"camera_id": "cam_99"}}})
    check("signal pointing at unknown camera rejected", False)
except ConfigError as e:
    check("signal pointing at unknown camera rejected", True, str(e)[:60])
try:
    Config({"database": {}, "model": {},
            "cameras": {"a": {"id": "c", "source": 0}, "b": {"id": "c", "source": 1}},
            "traffic_signals": {}})
    check("duplicate camera id rejected", False)
except ConfigError as e:
    check("duplicate camera id rejected", True, str(e)[:60])

print("\n=== 13. Numeric and shape validation catches real mistakes ===")

def bad(overrides, cam_extra=None, signals=None):
    """Build a config with one thing wrong and return the error, or None."""
    data = {"database": {"type": "sqlite"}, "model": {},
            "cameras": {"a": {"id": "c1", "source": 0, **(cam_extra or {})}},
            "traffic_signals": signals or {}}
    for path, value in overrides.items():
        section, key = path.split(".")
        data.setdefault(section, {})[key] = value
    try:
        Config(data)
        return None
    except ConfigError as e:
        return str(e)

# The off-by-100 that silently produces a detector finding nothing.
e = bad({"model.confidence_threshold": 50})
check("confidence_threshold of 50 rejected", e is not None and "0.0 to 1.0" in e,
      (e or "")[:58])
check("confidence_threshold of 0.5 accepted", bad({"model.confidence_threshold": 0.5}) is None)
check("a string where a number belongs is rejected",
      bad({"model.iou_threshold": "high"}) is not None)
check("bucket_seconds of 0 rejected", bad({"data_storage.bucket_seconds": 0}) is not None)
check("port 99999 rejected", bad({"webapp.port": 99999}) is not None)

# An inverted ROI produces an empty crop and zero detections, with no error.
e = bad({}, cam_extra={"roi": {"x1": 900, "y1": 0, "x2": 100, "y2": 500}})
check("inverted ROI rejected", e is not None and "no width" in e, (e or "")[:58])
check("valid ROI accepted",
      bad({}, cam_extra={"roi": {"x1": 0, "y1": 0, "x2": 1280, "y2": 720}}) is None)

# A malformed counting line would crash or mis-count deep in the tracker.
check("three-point counting line rejected",
      bad({}, cam_extra={"counting_line": [[0, 0], [1, 1], [2, 2]]}) is not None)
check("non-integer counting line rejected",
      bad({}, cam_extra={"counting_line": [[0, "top"], [100, 200]]}) is not None)
e = bad({}, cam_extra={"counting_line": [[400, 300], [400, 300]]})
check("zero-length counting line rejected", e is not None and "zero length" in e,
      (e or "")[:58])
check("valid counting line accepted",
      bad({}, cam_extra={"counting_line": [[0, 450], [1280, 450]]}) is None)

# Negative scale would invert every speed.
check("negative pixels_per_meter rejected",
      bad({}, cam_extra={"pixels_per_meter": -5}) is not None)
check("null pixels_per_meter accepted (means uncalibrated)",
      bad({}, cam_extra={"pixels_per_meter": None}) is None)

# Unsafe signal timings must not survive config, not just Phase construction.
sig = lambda t: {"s": {"camera_id": "c1", "timings": t}}
check("green_max below green_min rejected",
      bad({}, signals=sig({"green_min": 30, "green_max": 10})) is not None)
e = bad({}, signals=sig({"green_min": 2}))
check("green_min of 2s rejected as unsafe", e is not None and "unsafe" in e, (e or "")[:58])
e = bad({}, signals=sig({"yellow": 1}))
check("yellow of 1s rejected as unsafe", e is not None and "unsafe" in e, (e or "")[:58])
check("safe timings accepted",
      bad({}, signals=sig({"green_min": 10, "green_max": 50, "yellow": 3})) is None)

print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("="*60)
sys.exit(1 if FAIL else 0)
