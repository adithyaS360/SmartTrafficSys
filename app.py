"""
Smart Traffic Management System - application entry point.

    python app.py --mode simulated          # demo: simulated junction, no camera needed
    python app.py --mode simulated --speed 10   # 10x faster, for a short demo
    python app.py --mode live               # real cameras from config.yaml

Then open http://127.0.0.1:5000

TWO MODES, SAME CODE PATH - and that is the point, not a convenience:

  live       cameras -> detector -> tracker -> snapshots -> controller -> database
  simulated  simulator ------------------------> snapshots -> controller -> database

Everything downstream of "snapshots" is identical. The controller, the writer,
the API and the dashboard cannot tell which mode produced the data, because they
all consume the same interface. That matters for two reasons: the demo exercises
the real system rather than a mock-up of it, and a bug in the shared path shows
up in both modes rather than hiding in the one you test less.

Simulated mode exists because the sample footage is a motorway with no junction
and no signal - there is nothing there to control. It lets the dashboard and the
controller be developed and shown now, and swaps out for real cameras by
changing one flag once you have junction footage.

THREADING: collection runs on a background thread, Flask serves on the main one.
They share state through TrafficSystem, whose mutable fields are guarded by a
lock. SQLite is safe here because all writes come from the collection thread
while Flask only reads, which is exactly the one-writer/many-readers shape WAL
mode is built for.
"""

import argparse
import signal as signal_module
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api import create_api, register_error_handlers
from src.database.db_handler import Database, SnapshotWriter, TrafficRepository
from src.database.models import Camera
from src.traffic_controller import (AdaptiveStrategy, FixedTimeStrategy,
                                    IntersectionController, MLStrategy, Phase)
from src.utils.config_loader import load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


SIM_PHASES = [
    Phase(id="ns", name="North-South", camera_ids=["north", "south"],
          min_green=10.0, max_green=50.0, yellow=3.0, all_red=2.0),
    Phase(id="ew", name="East-West", camera_ids=["east", "west"],
          min_green=10.0, max_green=50.0, yellow=3.0, all_red=2.0),
]


class TrafficSystem:
    """Owns the live state, the collection loop, and the controller."""

    def __init__(self,
                 config,
                 mode: str = "simulated",
                 strategy_name: str = "adaptive",
                 speed: float = 1.0,
                 model_path: Optional[str] = None):
        self.config = config
        self.mode = mode
        self.speed = max(speed, 0.1)
        self.running = False
        self.started_at = time.time()
        self.pending_override: Optional[str] = None

        self._lock = threading.Lock()
        self._latest: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frames = 0
        self._last_snapshot_at: Optional[float] = None
        self._sim_clock = 0.0

        self.db = Database(config.database_url())
        self.db.create_all()
        self.repository = TrafficRepository(self.db)
        self.writer = SnapshotWriter(
            self.db,
            bucket_seconds=int(config.get("data_storage.bucket_seconds", 5)),
            batch_size=int(config.get("data_storage.batch_size", 50)),
            flush_interval=float(config.get("data_storage.flush_interval", 10)),
        )

        self._forecaster = None
        self._forecast_horizon = 15
        if model_path:
            self._load_model(model_path)

        strategy = self._build_strategy(strategy_name)

        if mode == "simulated":
            from src.simulator import ApproachSim, TrafficSimulator
            self._approaches = {
                a.camera_id: a for a in [
                    ApproachSim("north", lanes=2, peak_vehicles_per_hour=1250),
                    ApproachSim("south", lanes=2, peak_vehicles_per_hour=1150),
                    ApproachSim("east", lanes=1, peak_vehicles_per_hour=420),
                    ApproachSim("west", lanes=1, peak_vehicles_per_hour=360),
                ]
            }
            self._names = {c: f"{c.title()} approach" for c in self._approaches}
            self.controller = IntersectionController(
                "sim_signal", SIM_PHASES, strategy, repository=self.repository)
            self._register_cameras(self._approaches.keys())
            self.fleet = None
        else:
            from src.data_collector import MonitorFleet
            self.fleet = MonitorFleet(config)
            self._names = {m.camera_id: m.name for m in self.fleet.monitors.values()}
            self.db.sync_cameras(config)
            phases = self._live_phases()
            self.controller = (IntersectionController(
                config.get("traffic_signals.signal_1.id", "sig_1"),
                phases, strategy, repository=self.repository)
                if len(phases) >= 2 else None)
            if self.controller is None:
                log.warning(
                    "Only {} signal phase(s) configured, so no controller is running. "
                    "An intersection needs at least two phases - see the comment above "
                    "traffic_signals in config.yaml. Detection and storage still work.",
                    len(phases))
            self._approaches = {}

    # ---- setup helpers ----------------------------------------------

    def _build_strategy(self, name: str):
        if name == "fixed":
            return FixedTimeStrategy()
        if name == "ml":
            if self._forecaster is None:
                log.warning("Strategy 'ml' requested but no model is loaded; "
                            "falling back to adaptive.")
                return AdaptiveStrategy()
            return MLStrategy(predictor=lambda cam, snaps: self._predict(cam))
        return AdaptiveStrategy()

    def _live_phases(self) -> List[Phase]:
        from src.traffic_controller import build_phases_from_config
        return build_phases_from_config(self.config)

    def _register_cameras(self, ids) -> None:
        with self.db.session() as s:
            for cam in ids:
                if s.get(Camera, cam) is None:
                    s.add(Camera(id=cam, name=self._names.get(cam, cam), direction=cam))

    def _load_model(self, path: str) -> None:
        try:
            import numpy as np
            from tensorflow.keras.models import load_model
            self._forecaster = load_model(path)
            scaler = np.load(str(Path(path).with_suffix("")) + ".scaler.npz")
            self._scaler = scaler
            self._forecast_horizon = int(scaler["horizon"])
            log.info("Loaded forecasting model from {}", path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load model '{}': {}. Running without forecasts.", path, exc)
            self._forecaster = None

    # ---- the collection loop ----------------------------------------

    def start(self) -> "TrafficSystem":
        if self.running:
            return self
        if self.fleet is not None:
            self.fleet.start()
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="collector", daemon=True)
        self._thread.start()
        log.info("System started in {} mode", self.mode)
        return self

    def _loop(self) -> None:
        import random
        rng = random.Random(42)
        dt = 1.0
        start_hour = 7.0

        while not self._stop.is_set():
            began = time.time()

            if self.mode == "simulated":
                snapshots = self._step_simulation(dt, start_hour, rng)
            else:
                snapshots = self._step_live()

            if snapshots:
                with self._lock:
                    self._latest.update(snapshots)
                    self._last_snapshot_at = time.time()
                    self._frames += len(snapshots)
                for snap in snapshots.values():
                    self.writer.add(snap)

            if self.controller is not None:
                with self._lock:
                    observed = dict(self._latest)
                self.controller.tick(self.clock(), observed)
                self.pending_override = self.controller.pending_override

            self.writer.tick()

            # Pace the loop. In simulated mode `speed` compresses time so a
            # demo does not require sitting through a real rush hour.
            elapsed = time.time() - began
            time.sleep(max(0.0, dt / self.speed - elapsed))

        self.writer.close()
        log.info("Collection loop stopped")

    def _step_simulation(self, dt: float, start_hour: float, rng) -> Dict[str, Any]:
        from src.simulator import SimSnapshot
        hour = (start_hour + self._sim_clock / 3600.0) % 24.0
        for cam, approach in self._approaches.items():
            approach.step(dt, hour, self.controller.is_green_for(cam), rng)
        self._sim_clock += dt

        stamp = datetime.now(timezone.utc)
        return {
            cam: SimSnapshot(
                camera_id=cam, timestamp=stamp,
                queue_length=int(round(a.queue)),
                flow_rate=round(a.observed_flow_per_minute(dt), 2),
                vehicle_count=round(a.queue, 2),
                crossings_delta=a.take_crossings(),
                avg_dwell_seconds=round(a.average_delay, 2),
            )
            for cam, a in self._approaches.items()
        }

    def _step_live(self) -> Dict[str, Any]:
        snaps = self.fleet.poll() if self.fleet else []
        return {s.camera_id: s for s in snaps}

    def clock(self) -> float:
        """The controller's clock: simulated seconds, or wall seconds when live."""
        return self._sim_clock if self.mode == "simulated" else time.time()

    def stop(self) -> None:
        self._stop.set()
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self.fleet is not None:
            self.fleet.stop()
        self.db.dispose()

    # ---- read API used by the web layer ------------------------------

    def camera_ids(self) -> List[str]:
        return sorted(self._approaches) if self.mode == "simulated" else sorted(self._names)

    def camera_name(self, cam: str) -> str:
        return self._names.get(cam, cam)

    def latest_snapshots(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def totals(self) -> Dict[str, Any]:
        with self._lock:
            snaps = list(self._latest.values())
        return {
            "queue_length": sum(int(getattr(s, "queue_length", 0)) for s in snaps),
            "flow_rate": round(sum(float(getattr(s, "flow_rate", 0)) for s in snaps), 1),
            "vehicle_count": round(sum(float(getattr(s, "vehicle_count", 0)) for s in snaps), 1),
        }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            last = self._last_snapshot_at
            frames = self._frames
        return {
            "uptime_seconds": time.time() - self.started_at,
            "seconds_since_last_snapshot": round(time.time() - last, 1) if last else None,
            "frames_processed": frames,
            "rows_written": self.writer.rows_written,
            "cameras": len(self.camera_ids()),
        }

    def request_override(self, phase_id: Optional[str]) -> bool:
        if self.controller is None:
            return False
        accepted = self.controller.request_phase(phase_id)
        self.pending_override = self.controller.pending_override
        return accepted

    def _predict(self, camera_id: str) -> Optional[float]:
        if self._forecaster is None:
            return None
        try:
            import numpy as np
            rows = self.repository.per_minute(camera_id, hours=3)
            lookback = 30
            if len(rows) < lookback:
                return None
            from src.models.features import FEATURE_COLUMNS
            window = np.array([[float(r[c]) for c in FEATURE_COLUMNS]
                               for r in rows[-lookback:]], dtype=np.float32)
            scaled = (window - self._scaler["feature_means"]) / self._scaler["feature_stds"]
            out = self._forecaster.predict(scaled[None, ...], verbose=0).flatten()[0]
            return float(out * self._scaler["target_std"] + self._scaler["target_mean"])
        except Exception as exc:  # noqa: BLE001
            log.debug("Forecast failed for {}: {}", camera_id, exc)
            return None

    def forecast(self, camera_id: str) -> Optional[Dict[str, Any]]:
        value = self._predict(camera_id)
        if value is None:
            return None
        return {"predicted_flow_per_minute": round(value, 2),
                "horizon_minutes": self._forecast_horizon}


def create_app(system: TrafficSystem) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.register_blueprint(create_api(system))
    register_error_handlers(app)
    # Templates are cached unless this is on, so an edit to dashboard.html would
    # not appear until a restart - which is a confusing five minutes the first
    # time it happens.
    app.jinja_env.auto_reload = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html",
                               mode=system.mode,
                               signal_id=system.controller.signal_id if system.controller else None)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("simulated", "live"), default="simulated")
    parser.add_argument("--strategy", choices=("adaptive", "fixed", "ml"), default="adaptive")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="simulated-time multiplier (simulated mode only)")
    parser.add_argument("--model", default=None, help="path to a saved .keras model")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(level="INFO")
    config = load_config(args.config, env_file=".env")

    system = TrafficSystem(config, mode=args.mode, strategy_name=args.strategy,
                           speed=args.speed, model_path=args.model).start()

    def shutdown(_sig, _frame):
        log.info("Shutting down")
        system.stop()
        sys.exit(0)

    signal_module.signal(signal_module.SIGINT, shutdown)
    signal_module.signal(signal_module.SIGTERM, shutdown)

    app = create_app(system)
    host = args.host or config.get("webapp.host", "127.0.0.1")
    port = args.port or int(config.get("webapp.port", 5000))
    print(f"\n  Dashboard: http://127.0.0.1:{port}\n  Mode: {args.mode} "
          f"| strategy: {args.strategy}"
          f"{f' | speed x{args.speed:g}' if args.mode == 'simulated' else ''}\n")
    # use_reloader off: it forks a second process, which would start a second
    # collection thread writing to the same database.
    app.run(host=host, port=port, debug=args.debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
