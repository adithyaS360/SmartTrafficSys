"""
Verification of Phase 2: schema, UTC handling, bucket aggregation, idempotent
writes, foreign-key enforcement, and query roll-ups.

Runs against an on-disk SQLite database in a temp directory, so it exercises the
real engine, the real pragmas and the real constraints - not mocks.
"""
import sys, types, tempfile, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- stub the vision stack; nothing under test touches it -------------------
cv2 = types.ModuleType("cv2")
cv2.__getattr__ = lambda name: (lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0; cv2.CAP_PROP_BUFFERSIZE = 38; cv2.CAP_PROP_POS_FRAMES = 1
cv2.VideoCapture = object
sys.modules["cv2"] = cv2
ultra = types.ModuleType("ultralytics"); ultra.YOLO = object
sys.modules["ultralytics"] = ultra

from sqlalchemy.exc import IntegrityError
from src.data_collector import TrafficSnapshot
from src.database.db_handler import Database, SnapshotWriter, TrafficRepository
from src.database.models import Camera, TrafficSnapshotRow

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))

TMP = Path(tempfile.mkdtemp(prefix="stms_test_"))

def fresh_db():
    path = TMP / f"db_{len(list(TMP.iterdir()))}.db"
    db = Database(f"sqlite:///{path}")
    db.create_all()
    with db.session() as s:
        s.add(Camera(id="cam_1", name="North", direction="north"))
        s.add(Camera(id="cam_2", name="South", direction="south"))
    return db

def snap(ts, camera="cam_1", vehicles=10, queue=3, crossings=0, speed=None, dwell=5.0):
    return TrafficSnapshot(
        camera_id=camera, timestamp=ts, vehicle_count=vehicles, pedestrian_count=1,
        queue_length=queue, flow_rate=12.0, avg_dwell_seconds=dwell,
        avg_speed_kmh=speed, total_crossings=0, crossings_delta=crossings,
        class_breakdown={"car": vehicles}, processing_ms=40.0,
    )

BASE = datetime(2026, 9, 18, 10, 0, 0, tzinfo=timezone.utc)

print("\n=== 1. SQLite pragmas are actually applied ===")
db = fresh_db()
with db.session() as s:
    jm = s.execute(__import__("sqlalchemy").text("PRAGMA journal_mode")).scalar()
    fk = s.execute(__import__("sqlalchemy").text("PRAGMA foreign_keys")).scalar()
check("journal_mode=WAL (dashboard can read while collector writes)", jm.lower() == "wal", f"={jm}")
check("foreign_keys=ON (constraints are enforced, not decorative)", fk == 1, f"={fk}")

print("\n=== 2. Foreign keys reject an unknown camera ===")
rejected = False
try:
    with db.session() as s:
        s.add(TrafficSnapshotRow(camera_id="ghost_cam", bucket_start=BASE, bucket_seconds=5))
except IntegrityError:
    rejected = True
check("snapshot for a non-existent camera is rejected", rejected)

print("\n=== 3. Naive datetimes are refused at the boundary ===")
refused = False
try:
    with db.session() as s:
        s.add(TrafficSnapshotRow(camera_id="cam_1",
                                 bucket_start=datetime(2026, 9, 18, 10, 0, 0),  # no tzinfo
                                 bucket_seconds=5))
except (ValueError, Exception) as e:
    refused = "naive" in str(e).lower()
check("naive datetime rejected with a useful message", refused)

print("\n=== 4. Timestamps survive the round-trip as UTC-aware ===")
db4 = fresh_db()
with db4.session() as s:
    s.add(TrafficSnapshotRow(camera_id="cam_1", bucket_start=BASE, bucket_seconds=5))
row = TrafficRepository(db4).latest("cam_1")
check("returned timestamp is timezone-aware", row.bucket_start.tzinfo is not None)
check("returned timestamp equals what went in", row.bucket_start == BASE,
      f"{row.bucket_start} vs {BASE}")

print("\n=== 5. 150 frames collapse into ONE bucket row ===")
db5 = fresh_db()
w5 = SnapshotWriter(db5, bucket_seconds=5)
for i in range(150):                       # 5 seconds at 30fps
    w5.add(snap(BASE + timedelta(seconds=i * 5 / 150)))
w5.close()
rows = TrafficRepository(db5).series("cam_1", minutes=60 * 24 * 365)
check("150 snapshots -> 1 row", len(rows) == 1, f"rows={len(rows)}")
check("sample_count records all 150 frames", rows and rows[0].sample_count == 150,
      f"sample_count={rows[0].sample_count if rows else '-'}")

print("\n=== 6. Each field uses the statistic that preserves its meaning ===")
db6 = fresh_db()
w6 = SnapshotWriter(db6, bucket_seconds=5)
# queue spikes to 20 mid-bucket; crossings happen on two frames
for i, (q, x) in enumerate([(2, 0), (5, 1), (20, 0), (4, 1), (3, 0)]):
    w6.add(snap(BASE + timedelta(seconds=i * 0.5), queue=q, crossings=x, vehicles=10))
w6.close()
r = TrafficRepository(db6).series("cam_1", minutes=60 * 24 * 365)[0]
check("queue_length uses MAX - the peak is what must be cleared", r.queue_length == 20,
      f"queue_length={r.queue_length} (mean would be 6.8)")
check("crossings uses SUM - counts add", r.crossings == 2, f"crossings={r.crossings}")
check("vehicle_count uses MEAN - typical occupancy", abs(r.vehicle_count - 10.0) < 0.01,
      f"vehicle_count={r.vehicle_count}")

print("\n=== 7. Uncalibrated speed stays NULL, never a fake zero ===")
db7 = fresh_db()
w7 = SnapshotWriter(db7, bucket_seconds=5)
for i in range(10):
    w7.add(snap(BASE + timedelta(seconds=i * 0.4), speed=None))
w7.close()
r7 = TrafficRepository(db7).series("cam_1", minutes=60 * 24 * 365)[0]
check("all-uncalibrated bucket -> NULL speed", r7.avg_speed_kmh is None,
      f"avg_speed_kmh={r7.avg_speed_kmh}")

db7b = fresh_db()
w7b = SnapshotWriter(db7b, bucket_seconds=5)
for i in range(10):
    w7b.add(snap(BASE + timedelta(seconds=i * 0.4), speed=30.0 + i))
w7b.close()
r7b = TrafficRepository(db7b).series("cam_1", minutes=60 * 24 * 365)[0]
check("calibrated bucket -> mean speed", abs(r7b.avg_speed_kmh - 34.5) < 0.01,
      f"avg_speed_kmh={r7b.avg_speed_kmh}")

print("\n=== 8. Separate cameras get separate buckets ===")
db8 = fresh_db()
w8 = SnapshotWriter(db8, bucket_seconds=5)
for i in range(20):
    w8.add(snap(BASE + timedelta(seconds=i * 0.2), camera="cam_1", vehicles=5))
    w8.add(snap(BASE + timedelta(seconds=i * 0.2), camera="cam_2", vehicles=17))
w8.close()
repo8 = TrafficRepository(db8)
a = repo8.series("cam_1", minutes=60 * 24 * 365)
b = repo8.series("cam_2", minutes=60 * 24 * 365)
check("one row per camera, not merged", len(a) == 1 and len(b) == 1, f"cam_1={len(a)} cam_2={len(b)}")
check("counts are not cross-contaminated",
      abs(a[0].vehicle_count - 5) < 0.01 and abs(b[0].vehicle_count - 17) < 0.01,
      f"cam_1={a[0].vehicle_count} cam_2={b[0].vehicle_count}")

print("\n=== 9. Frames spread over 30s produce 6 buckets ===")
db9 = fresh_db()
w9 = SnapshotWriter(db9, bucket_seconds=5)
for i in range(180):                       # 30s at 6fps
    w9.add(snap(BASE + timedelta(seconds=i / 6.0)))
w9.close()
rows9 = TrafficRepository(db9).series("cam_1", minutes=60 * 24 * 365)
check("30s of frames -> 6 five-second buckets", len(rows9) == 6, f"buckets={len(rows9)}")
check("buckets are aligned to the 5s grid",
      all(r.bucket_start.second % 5 == 0 for r in rows9),
      f"starts={[r.bucket_start.second for r in rows9]}")

print("\n=== 10. Duplicate bucket is skipped, rest of batch survives ===")
db10 = fresh_db()
w10 = SnapshotWriter(db10, bucket_seconds=5)
for i in range(5):
    w10.add(snap(BASE + timedelta(seconds=i * 0.5)))
w10.close()
# Simulate a restart re-emitting the same bucket plus a new one.
w10b = SnapshotWriter(db10, bucket_seconds=5)
for i in range(5):
    w10b.add(snap(BASE + timedelta(seconds=i * 0.5)))          # duplicate
    w10b.add(snap(BASE + timedelta(seconds=10 + i * 0.5)))     # new
w10b.close()
rows10 = TrafficRepository(db10).series("cam_1", minutes=60 * 24 * 365)
check("duplicate rejected, new bucket still written", len(rows10) == 2, f"rows={len(rows10)}")
check("rejection was counted, not silently swallowed", w10b.rows_rejected == 1,
      f"rejected={w10b.rows_rejected} written={w10b.rows_written}")

print("\n=== 11. per_minute() rolls 5s buckets up for the LSTM ===")
db11 = fresh_db()
w11 = SnapshotWriter(db11, bucket_seconds=5)
now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
for minute in range(3):
    for i in range(60):                    # 60 frames per minute
        w11.add(snap(now - timedelta(minutes=2 - minute) + timedelta(seconds=i),
                     queue=minute * 5, crossings=1 if i % 20 == 0 else 0))
w11.close()
series = TrafficRepository(db11).per_minute("cam_1", hours=1)
check("three minutes of data -> three rows", len(series) == 3, f"rows={len(series)}")
check("crossings summed per minute", all(r["crossings"] == 3 for r in series),
      f"crossings={[r['crossings'] for r in series]}")
check("queue_length is the peak within the minute",
      [r["queue_length"] for r in series] == [0, 5, 10],
      f"{[r['queue_length'] for r in series]}")

print("\n=== 12. Signal events are recorded and queryable ===")
db12 = fresh_db()
repo12 = TrafficRepository(db12)
for i, (phase, reason) in enumerate([("green", "queue_threshold"), ("yellow", "phase_end"),
                                     ("red", "phase_end"), ("green", "min_green")]):
    repo12.record_signal_event(signal_id="sig_1", camera_id="cam_1", phase=phase,
                               duration_seconds=20 + i, reason=reason,
                               queue_at_decision=i * 4, flow_at_decision=10.0 + i)
events = repo12.signal_events("sig_1", hours=1)
check("all four decisions stored", len(events) == 4, f"events={len(events)}")
check("decision context preserved for auditing",
      events[0].queue_at_decision == 0 and events[-1].reason == "min_green")

print("\n=== 13. Retention purge deletes only old rows ===")
db13 = fresh_db()
with db13.session() as s:
    s.add(TrafficSnapshotRow(camera_id="cam_1",
                             bucket_start=datetime.now(timezone.utc) - timedelta(days=120),
                             bucket_seconds=5))
    s.add(TrafficSnapshotRow(camera_id="cam_1",
                             bucket_start=datetime.now(timezone.utc) - timedelta(days=1),
                             bucket_seconds=5))
deleted = TrafficRepository(db13).purge_older_than(90)
remaining = TrafficRepository(db13).series("cam_1", minutes=60 * 24 * 365)
check("one old row purged, recent row kept", deleted == 1 and len(remaining) == 1,
      f"deleted={deleted} remaining={len(remaining)}")

print("\n=== 14. Config switches backend without touching any other code ===")
from src.utils.config_loader import Config, ConfigError
sqlite_cfg = Config({"database": {"type": "sqlite", "path": "data/traffic.db"},
                     "model": {}, "cameras": {"a": {"id": "c1", "source": 0}},
                     "traffic_signals": {}})
check("sqlite URL built", sqlite_cfg.database_url() == "sqlite:///data/traffic.db",
      sqlite_cfg.database_url())
pg_cfg = Config({"database": {"type": "postgresql", "host": "db", "port": 5432,
                              "name": "traffic_db", "user": "u", "password": "p"},
                 "model": {}, "cameras": {"a": {"id": "c1", "source": 0}},
                 "traffic_signals": {}})
check("postgres URL built", pg_cfg.database_url() ==
      "postgresql+psycopg2://u:p@db:5432/traffic_db", pg_cfg.database_url())
try:
    Config({"database": {"type": "postgresql", "host": "db", "port": 5432,
                         "name": "d", "user": "u", "password": ""},
            "model": {}, "cameras": {"a": {"id": "c1", "source": 0}},
            "traffic_signals": {}}).database_url()
    check("postgres without a password fails loudly", False)
except ConfigError as e:
    check("postgres without a password fails loudly", True, str(e)[:55])

shutil.rmtree(TMP, ignore_errors=True)
print(f"\n{'='*62}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
