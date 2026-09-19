"""
Database schema.

WRITTEN TO RUN ON BOTH SQLITE AND POSTGRES.

You are developing on SQLite and will move to Postgres later. SQLAlchemy hides
most of that difference, but not all of it, and the parts it does not hide are
exactly the parts that break silently months later. Three of them are handled
explicitly here:

1. PRIMARY KEYS. SQLite only auto-increments a column declared INTEGER PRIMARY
   KEY - a BIGINT primary key does not auto-increment and inserts fail. So the
   id columns are BigInteger on Postgres and Integer on SQLite, via with_variant.

2. JSON. Postgres has JSONB, which is binary, faster and indexable. SQLite has
   only text JSON. with_variant gives each backend its best option from one
   declaration.

3. TIMESTAMPS - the one that actually bites. SQLite has no timezone-aware type.
   DateTime(timezone=True) is silently ignored, so a tz-aware datetime goes in
   and a NAIVE one comes back. Your code then compares an aware datetime to a
   naive one and raises TypeError, or worse, does arithmetic across a timezone
   offset and is quietly wrong by 5h30m. The UTCDateTime decorator below forces
   everything to UTC on the way in and re-attaches UTC on the way out, so both
   backends behave identically.

TIME BUCKETING - why we do not store one row per frame:
At 30fps one camera produces 2.6 million rows a day. Two cameras is 5 million.
Nothing reads data at that resolution: the signal controller decides every few
seconds and the LSTM will be trained on minute-level data. So frames are
aggregated into fixed buckets (5s by default) before being written. That is
roughly a 150x reduction with no loss of anything we actually use.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Portable column types
# ---------------------------------------------------------------------------

class UTCDateTime(TypeDecorator):
    """
    A timestamp that is always UTC-aware, on every backend.

    Storing naive local times is the single most common way a project like this
    ends up with data it cannot trust. Rejecting naive datetimes at the boundary
    means the mistake surfaces at the line that made it.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "Refusing to store a naive datetime. Use datetime.now(timezone.utc), "
                "not datetime.now()."
            )
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:          # SQLite hands back naive values
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


# BigInt on Postgres, plain Integer on SQLite so AUTOINCREMENT works.
PortableBigInt = BigInteger().with_variant(Integer, "sqlite")
# JSONB on Postgres (binary, indexable), generic JSON elsewhere.
PortableJSON = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class Camera(Base):
    """
    Registry of known cameras.

    config.yaml remains the source of truth; this table is synced from it at
    startup. It exists so snapshots can foreign-key to something real, so the
    dashboard can show a human name instead of 'cam_1', and so calibration
    values survive alongside the data they apply to.
    """

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    direction: Mapped[Optional[str]] = mapped_column(String(32))
    pixels_per_meter: Mapped[Optional[float]] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now(), nullable=False
    )

    snapshots: Mapped[list["TrafficSnapshotRow"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Camera {self.id} '{self.name}'>"


class TrafficSnapshotRow(Base):
    """
    One time bucket of traffic state for one camera.

    Each aggregate uses the statistic that is actually meaningful for it, which
    is not the same statistic in each case:

      vehicle_count     MEAN  - typical occupancy over the bucket
      queue_length      MAX   - the PEAK queue is what the controller must clear;
                                averaging it hides exactly the moment that matters
      crossings         SUM   - counts add; this is the LSTM's target variable
      avg_speed_kmh     MEAN of non-null values, or NULL if the camera is
                                uncalibrated. Never a fabricated zero.
      sample_count      how many frames fed this bucket - a data-quality signal.
                                A bucket built from 3 frames instead of 150 means
                                detection was falling behind, and any model
                                trained on it should know that.
    """

    __tablename__ = "traffic_snapshots"
    __table_args__ = (
        # Makes writes idempotent: re-running a bucket after a restart is rejected
        # rather than silently duplicated.
        UniqueConstraint("camera_id", "bucket_start", name="uq_snapshot_camera_bucket"),
        # The dashboard and the model trainer both query "one camera, a time
        # range, in order". This composite index serves exactly that.
        Index("ix_snapshot_camera_time", "camera_id", "bucket_start"),
        Index("ix_snapshot_time", "bucket_start"),
    )

    id: Mapped[int] = mapped_column(PortableBigInt, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )

    bucket_start: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    bucket_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    vehicle_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pedestrian_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    queue_length: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    crossings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    flow_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_dwell_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_speed_kmh: Mapped[Optional[float]] = mapped_column(Float)

    class_breakdown: Mapped[Optional[dict]] = mapped_column(PortableJSON)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_processing_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now(), nullable=False
    )

    camera: Mapped["Camera"] = relationship(back_populates="snapshots")

    def __repr__(self) -> str:
        return (
            f"<Snapshot {self.camera_id} @{self.bucket_start:%H:%M:%S} "
            f"veh={self.vehicle_count:.1f} q={self.queue_length} x={self.crossings}>"
        )


class SignalEvent(Base):
    """
    Every decision the signal controller makes.

    WHY THIS TABLE MATTERS MORE THAN IT LOOKS:
    The claim your project makes is "adaptive timing reduces waiting". That claim
    is unfalsifiable without a record of what the controller did and what it saw
    when it did it. With this table you can replay a day, compare measured dwell
    time under adaptive control against the fixed-time baseline, and put a real
    number in your report instead of an assertion.

    'reason' is a short machine-readable tag rather than free text, so the
    decisions can be counted and grouped: how often did we hit green_max? How
    often did the ML prediction override the queue rule?
    """

    __tablename__ = "signal_events"
    __table_args__ = (
        Index("ix_signal_event_time", "signal_id", "timestamp"),
    )

    id: Mapped[int] = mapped_column(PortableBigInt, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("cameras.id", ondelete="SET NULL")
    )

    timestamp: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    phase: Mapped[str] = mapped_column(String(16), nullable=False)        # green|yellow|red
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)

    # The state the controller saw when it decided. Needed to audit the decision.
    queue_at_decision: Mapped[Optional[int]] = mapped_column(Integer)
    flow_at_decision: Mapped[Optional[float]] = mapped_column(Float)
    predicted_flow: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<SignalEvent {self.signal_id} {self.phase} {self.duration_seconds}s ({self.reason})>"
