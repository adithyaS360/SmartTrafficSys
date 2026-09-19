"""
Database engine, session handling, and the buffered snapshot writer.

THE WRITE PATH, and why it is not just session.add(snapshot):

Frames arrive about 30 times a second per camera. A database round-trip per
frame is both far more write traffic than the data deserves and slow enough to
stall the detection loop - and stalling the detection loop is how the system
ends up making signal decisions on stale footage. So writes are buffered:

    process_once() -> TrafficSnapshot  (in memory, ~30/s)
              |
        SnapshotWriter.add()           (accumulates into a time bucket)
              |
        bucket closes (5s elapsed)     (aggregate ~150 frames into 1 row)
              |
        flush() every N rows or T secs (one batched INSERT)

That is roughly 150x fewer rows and orders of magnitude fewer round-trips, and
it loses nothing we actually read back.

SQLITE SPECIFICS, since that is what you are running:
- journal_mode=WAL lets the Flask dashboard read while the collector writes.
  Without it SQLite takes a whole-database lock on every write and the dashboard
  intermittently fails with 'database is locked'.
- foreign_keys=ON because SQLite does NOT enforce foreign keys by default. Your
  constraints are decorative until you turn this on, which means a bug that
  Postgres would reject gets quietly written instead.
- One writer. The camera threads only capture frames; all database writes happen
  on whichever thread drives the poll loop. SQLite tolerates many readers and one
  writer, and that is exactly the shape we have.
"""

import math
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from sqlalchemy import create_engine, delete, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, Camera, SignalEvent, TrafficSnapshotRow
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, url: str, echo: bool = False, pool_size: int = 10):
        self.url = url
        self.is_sqlite = url.startswith("sqlite")

        if self.is_sqlite:
            # Make sure the directory exists, or SQLite fails with a bare
            # "unable to open database file" that says nothing useful.
            db_path = url.replace("sqlite:///", "")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: the Flask request threads read from the
            # same engine as the collector loop. Safe here because writes are
            # serialised through a single writer.
            self.engine: Engine = create_engine(
                url, echo=echo, future=True,
                connect_args={"check_same_thread": False, "timeout": 15},
            )
            self._install_sqlite_pragmas()
        else:
            self.engine = create_engine(
                url, echo=echo, future=True,
                pool_size=pool_size, max_overflow=5,
                pool_pre_ping=True,   # drop dead connections instead of erroring
            )

        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        log.info("Database ready ({})", "sqlite" if self.is_sqlite else "postgresql")

    def _install_sqlite_pragmas(self) -> None:
        @event.listens_for(self.engine, "connect")
        def _set_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    def create_all(self) -> None:
        """
        Create tables directly from the models.

        Convenient for tests and a first run. For anything you care about keeping,
        use the Alembic migrations instead - create_all cannot ALTER an existing
        table, so once you have data it silently does nothing when the schema
        changes.
        """
        Base.metadata.create_all(self.engine)
        log.info("Schema created ({} tables)", len(Base.metadata.tables))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commits on success, rolls back on exception."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def sync_cameras(self, config) -> int:
        """
        Mirror the cameras from config.yaml into the cameras table.

        config.yaml stays the source of truth - this only makes the foreign keys
        resolvable and gives queries a human-readable name to join to.
        """
        count = 0
        with self.session() as s:
            for _, cam in config.cameras.items():
                row = s.get(Camera, cam["id"])
                if row is None:
                    row = Camera(id=cam["id"])
                    s.add(row)
                    count += 1
                row.name = cam.get("name", cam["id"])
                row.direction = cam.get("direction")
                row.pixels_per_meter = cam.get("pixels_per_meter")
                row.is_active = True
        log.info("Synced cameras from config ({} new)", count)
        return count

    def dispose(self) -> None:
        self.engine.dispose()


# ---------------------------------------------------------------------------
# Bucketed, batched writer
# ---------------------------------------------------------------------------

@dataclass
class _OpenBucket:
    """Frames accumulating for one (camera, time bucket) pair."""

    camera_id: str
    bucket_start: datetime
    vehicle_counts: List[float] = field(default_factory=list)
    pedestrian_counts: List[float] = field(default_factory=list)
    queue_lengths: List[int] = field(default_factory=list)
    flow_rates: List[float] = field(default_factory=list)
    dwells: List[float] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    processing_ms: List[float] = field(default_factory=list)
    breakdowns: List[Dict[str, int]] = field(default_factory=list)
    crossings: int = 0

    def add(self, snap) -> None:
        self.vehicle_counts.append(snap.vehicle_count)
        self.pedestrian_counts.append(snap.pedestrian_count)
        self.queue_lengths.append(snap.queue_length)
        self.flow_rates.append(snap.flow_rate)
        self.dwells.append(snap.avg_dwell_seconds)
        self.processing_ms.append(snap.processing_ms)
        self.crossings += getattr(snap, "crossings_delta", 0)
        if snap.avg_speed_kmh is not None:
            self.speeds.append(snap.avg_speed_kmh)
        if snap.class_breakdown:
            self.breakdowns.append(snap.class_breakdown)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def to_row(self, bucket_seconds: int) -> TrafficSnapshotRow:
        """
        Collapse the bucket into one row, choosing a statistic per field that
        preserves the meaning of that field (see models.py for the reasoning).
        """
        merged: Dict[str, float] = defaultdict(float)
        for breakdown in self.breakdowns:
            for cls, n in breakdown.items():
                merged[cls] += n
        n_breakdowns = max(len(self.breakdowns), 1)
        averaged = {cls: round(total / n_breakdowns, 2) for cls, total in merged.items()}

        return TrafficSnapshotRow(
            camera_id=self.camera_id,
            bucket_start=self.bucket_start,
            bucket_seconds=bucket_seconds,
            vehicle_count=round(self._mean(self.vehicle_counts), 2),
            pedestrian_count=round(self._mean(self.pedestrian_counts), 2),
            queue_length=max(self.queue_lengths) if self.queue_lengths else 0,
            crossings=self.crossings,
            flow_rate=round(self._mean(self.flow_rates), 2),
            avg_dwell_seconds=round(self._mean(self.dwells), 2),
            # None, not 0.0 - an uncalibrated camera has no speed, and a zero
            # would be indistinguishable from a genuine traffic jam.
            avg_speed_kmh=round(self._mean(self.speeds), 2) if self.speeds else None,
            class_breakdown=averaged or None,
            sample_count=len(self.vehicle_counts),
            avg_processing_ms=round(self._mean(self.processing_ms), 2),
        )


class SnapshotWriter:
    """
    Buffers TrafficSnapshots, aggregates them into time buckets, writes in batches.

    Usage:
        writer = SnapshotWriter(db, bucket_seconds=5)
        writer.add(snapshot)     # every frame
        writer.tick()            # once per loop - closes and flushes when due
        writer.close()           # on shutdown, flushes everything
    """

    def __init__(self,
                 db: Database,
                 bucket_seconds: int = 5,
                 batch_size: int = 50,
                 flush_interval: float = 10.0,
                 grace_seconds: float = 1.0):
        """
        Args:
            bucket_seconds: width of each aggregation window
            batch_size: rows to accumulate before writing
            flush_interval: write anyway after this long, so a quiet camera's
                data still reaches the database promptly
            grace_seconds: wait this long past a bucket's end before closing it,
                so a slightly late frame is not dropped into the next bucket
        """
        self.db = db
        self.bucket_seconds = bucket_seconds
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.grace_seconds = grace_seconds

        self._open: Dict[tuple, _OpenBucket] = {}
        self._pending: List[TrafficSnapshotRow] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self.rows_written = 0
        self.rows_rejected = 0

    def _bucket_start(self, ts: datetime) -> datetime:
        """Floor a timestamp to the start of its bucket."""
        epoch = ts.timestamp()
        floored = math.floor(epoch / self.bucket_seconds) * self.bucket_seconds
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def add(self, snap) -> None:
        """Route one snapshot into its bucket."""
        start = self._bucket_start(snap.timestamp)
        key = (snap.camera_id, start)
        with self._lock:
            bucket = self._open.get(key)
            if bucket is None:
                bucket = _OpenBucket(camera_id=snap.camera_id, bucket_start=start)
                self._open[key] = bucket
            bucket.add(snap)

    def tick(self) -> int:
        """
        Close any buckets whose window has elapsed and flush if due.
        Call once per poll loop iteration.

        Returns:
            number of rows written by this call
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.bucket_seconds + self.grace_seconds)

        with self._lock:
            closed = [key for key, b in self._open.items() if b.bucket_start <= cutoff]
            for key in closed:
                bucket = self._open.pop(key)
                self._pending.append(bucket.to_row(self.bucket_seconds))

        due = (len(self._pending) >= self.batch_size
               or (self._pending and time.time() - self._last_flush >= self.flush_interval))
        return self.flush() if due else 0

    def flush(self) -> int:
        """Write all pending rows. Returns the number written."""
        with self._lock:
            rows, self._pending = self._pending, []
        if not rows:
            return 0

        try:
            with self.db.session() as s:
                s.add_all(rows)
            written = len(rows)
        except IntegrityError:
            # Almost always the unique (camera_id, bucket_start) constraint after
            # a restart re-emitted a bucket. Fall back to row-by-row so one
            # duplicate does not discard the whole batch.
            written = self._insert_individually(rows)
        except Exception as exc:
            log.error("Snapshot flush failed, dropping {} row(s): {}", len(rows), exc)
            self.rows_rejected += len(rows)
            return 0

        self.rows_written += written
        self._last_flush = time.time()
        log.debug("Flushed {} snapshot row(s) (total {})", written, self.rows_written)
        return written

    def _insert_individually(self, rows: List[TrafficSnapshotRow]) -> int:
        written = 0
        for row in rows:
            try:
                with self.db.session() as s:
                    s.add(row)
                written += 1
            except IntegrityError:
                self.rows_rejected += 1
                log.debug("Duplicate bucket skipped: {} @ {}", row.camera_id, row.bucket_start)
            except Exception as exc:
                self.rows_rejected += 1
                log.warning("Row rejected: {}", exc)
        return written

    def close(self) -> int:
        """Force-close every open bucket and flush. Call on shutdown."""
        with self._lock:
            for key in list(self._open):
                self._pending.append(self._open.pop(key).to_row(self.bucket_seconds))
        written = self.flush()
        log.info("Writer closed: {} rows written, {} rejected", self.rows_written, self.rows_rejected)
        return written

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "open_buckets": len(self._open),
            "pending_rows": len(self._pending),
            "rows_written": self.rows_written,
            "rows_rejected": self.rows_rejected,
        }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TrafficRepository:
    """Read queries. Kept apart from the writer so the API layer gets a narrow surface."""

    def __init__(self, db: Database):
        self.db = db

    def latest(self, camera_id: str) -> Optional[TrafficSnapshotRow]:
        with self.db.session() as s:
            return s.scalar(
                select(TrafficSnapshotRow)
                .where(TrafficSnapshotRow.camera_id == camera_id)
                .order_by(TrafficSnapshotRow.bucket_start.desc())
                .limit(1)
            )

    def latest_all(self) -> Dict[str, TrafficSnapshotRow]:
        """Most recent bucket per camera - what the dashboard shows on load."""
        out: Dict[str, TrafficSnapshotRow] = {}
        with self.db.session() as s:
            for cam_id in s.scalars(select(Camera.id)):
                row = s.scalar(
                    select(TrafficSnapshotRow)
                    .where(TrafficSnapshotRow.camera_id == cam_id)
                    .order_by(TrafficSnapshotRow.bucket_start.desc())
                    .limit(1)
                )
                if row is not None:
                    out[cam_id] = row
        return out

    def series(self, camera_id: str, minutes: int = 60) -> List[TrafficSnapshotRow]:
        """Raw buckets for the last N minutes, oldest first - the dashboard chart."""
        since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        with self.db.session() as s:
            return list(s.scalars(
                select(TrafficSnapshotRow)
                .where(TrafficSnapshotRow.camera_id == camera_id,
                       TrafficSnapshotRow.bucket_start >= since)
                .order_by(TrafficSnapshotRow.bucket_start)
            ))

    def per_minute(self, camera_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Roll 5-second buckets up to per-minute rows - the LSTM's training set.

        THIS IS THE ONE PLACE THE TWO BACKENDS GENUINELY DIVERGE. Postgres
        truncates timestamps with date_trunc(); SQLite has no such function and
        needs strftime(). SQLAlchemy cannot paper over it, so we branch on the
        dialect. Keeping that branch in a single method is the point - the rest
        of the codebase stays backend-agnostic.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        if self.db.is_sqlite:
            minute = func.strftime("%Y-%m-%d %H:%M:00", TrafficSnapshotRow.bucket_start)
        else:
            minute = func.date_trunc("minute", TrafficSnapshotRow.bucket_start)

        with self.db.session() as s:
            rows = s.execute(
                select(
                    minute.label("minute"),
                    func.avg(TrafficSnapshotRow.vehicle_count).label("vehicle_count"),
                    func.max(TrafficSnapshotRow.queue_length).label("queue_length"),
                    func.sum(TrafficSnapshotRow.crossings).label("crossings"),
                    func.avg(TrafficSnapshotRow.avg_dwell_seconds).label("avg_dwell_seconds"),
                    func.sum(TrafficSnapshotRow.sample_count).label("sample_count"),
                )
                .where(TrafficSnapshotRow.camera_id == camera_id,
                       TrafficSnapshotRow.bucket_start >= since)
                .group_by(minute)
                .order_by(minute)
            ).all()

        return [
            {
                "minute": str(r.minute),
                "vehicle_count": round(float(r.vehicle_count or 0), 2),
                "queue_length": int(r.queue_length or 0),
                "crossings": int(r.crossings or 0),
                "avg_dwell_seconds": round(float(r.avg_dwell_seconds or 0), 2),
                "sample_count": int(r.sample_count or 0),
            }
            for r in rows
        ]

    def record_signal_event(self, **kwargs) -> SignalEvent:
        """Log one controller decision. Used by Phase 3."""
        kwargs.setdefault("timestamp", datetime.now(timezone.utc))
        event_row = SignalEvent(**kwargs)
        with self.db.session() as s:
            s.add(event_row)
        return event_row

    def signal_events(self, signal_id: str, hours: int = 24) -> List[SignalEvent]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.db.session() as s:
            return list(s.scalars(
                select(SignalEvent)
                .where(SignalEvent.signal_id == signal_id, SignalEvent.timestamp >= since)
                .order_by(SignalEvent.timestamp)
            ))

    def purge_older_than(self, days: int) -> int:
        """Apply the retention policy. Returns rows deleted."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self.db.session() as s:
            result = s.execute(
                delete(TrafficSnapshotRow).where(TrafficSnapshotRow.bucket_start < cutoff)
            )
        deleted = result.rowcount or 0
        if deleted:
            log.info("Purged {} snapshot row(s) older than {} days", deleted, days)
        return deleted


def build_database(config) -> Database:
    """Construct a Database from a loaded Config."""
    return Database(
        url=config.database_url(),
        echo=bool(config.get("database.echo", False)),
        pool_size=int(config.get("database.pool_size", 10)),
    )
