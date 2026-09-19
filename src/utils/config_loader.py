"""
Configuration loader.

WHY THIS EXISTS:
config.yaml contains `${DB_PASSWORD}` style placeholders. Raw yaml.safe_load
returns those as the literal string "${DB_PASSWORD}", which would then be sent
to Postgres as the password and fail with a confusing auth error. This module
substitutes environment variables at load time and validates the result, so a
misconfiguration fails immediately at startup with a clear message rather than
three layers deep at runtime.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

log = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when the config file is missing, malformed, or incomplete."""


def _substitute_env(value: Any) -> Any:
    """
    Recursively replace ${VAR} and ${VAR:-default} with environment values.
    Walks dicts and lists so placeholders work at any depth.
    """
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            var_name, default = match.group(1), match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ConfigError(
                f"Environment variable '{var_name}' is referenced in config.yaml "
                f"but is not set. Copy .env.example to .env and fill it in."
            )
        return _ENV_PATTERN.sub(replace, value)
    return value


class Config:
    """Dict-backed config with dotted-path access and validation."""

    REQUIRED_SECTIONS = ("database", "model", "cameras", "traffic_signals")

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._validate()

    # ---- access -------------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        """
        Fetch a nested value with a dotted path:
            config.get("model.confidence_threshold")
            config.get("cameras.intersection_1.source")
        """
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        """Same as get(), but raises if the key is absent. Use for values with no sane default."""
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigError(f"Required config key '{path}' is missing from config.yaml")
        return value

    @property
    def cameras(self) -> Dict[str, Dict[str, Any]]:
        """All configured cameras, keyed by their config name."""
        return self._data.get("cameras", {})

    @property
    def signals(self) -> Dict[str, Dict[str, Any]]:
        """All configured traffic signals, keyed by their config name."""
        return self._data.get("traffic_signals", {})

    def database_url(self) -> str:
        """
        Build a SQLAlchemy connection URL from the database section.

        This method is the ONLY place that knows which backend is in use.
        Everything downstream receives a URL and does not care.
        """
        db = self._data["database"]
        backend = db.get("type", "sqlite")

        if backend == "sqlite":
            return f"sqlite:///{db.get('path', 'data/traffic.db')}"

        if backend == "postgresql":
            password = db.get("password") or ""
            if not password:
                raise ConfigError(
                    "database.type is 'postgresql' but no password is set. "
                    "Put DB_PASSWORD in your .env file, or switch database.type "
                    "back to 'sqlite' for local development."
                )
            return (
                f"postgresql+psycopg2://{db['user']}:{password}"
                f"@{db['host']}:{db['port']}/{db['name']}"
            )

        raise ConfigError(
            f"Unknown database.type '{backend}'. Expected 'sqlite' or 'postgresql'."
        )

    # ---- validation ---------------------------------------------------

    # (path, low, high, note) - checked when the key is present.
    NUMERIC_BOUNDS = (
        ("model.confidence_threshold", 0.0, 1.0,
         "a probability; 0.5 is typical. If you meant 50%, write 0.5"),
        ("model.iou_threshold", 0.0, 1.0, "a probability; 0.45 is typical"),
        ("data_storage.bucket_seconds", 1, 3600, "seconds per aggregation window"),
        ("data_storage.batch_size", 1, 100_000, "rows buffered before a write"),
        ("data_storage.flush_interval", 0.1, 3600, "seconds"),
        ("data_storage.retention_days", 1, 3650, "days of history to keep"),
        ("traffic_control.update_interval", 0.1, 600, "seconds between decisions"),
        ("webapp.port", 1, 65535, "TCP port"),
    )

    def _validate(self) -> None:
        missing = [s for s in self.REQUIRED_SECTIONS if s not in self._data]
        if missing:
            raise ConfigError(f"config.yaml is missing required section(s): {', '.join(missing)}")

        if not self._data["cameras"]:
            raise ConfigError("At least one camera must be defined under 'cameras'.")

        self._validate_numeric_bounds()
        seen_ids = self._validate_cameras()
        self._validate_signals(seen_ids)

        log.debug("Configuration validated: {} camera(s), {} signal(s)",
                  len(self._data["cameras"]), len(self._data.get("traffic_signals", {})))

    def _validate_numeric_bounds(self) -> None:
        """
        Range-check the numbers.

        Structural validation catches a missing key. It does not catch
        confidence_threshold: 50 - which parses fine, is silently clamped or
        matches nothing, and leaves you staring at a detector that finds no
        vehicles with nothing in the log to explain why. Off-by-100 on a
        probability is the single most common config mistake here.
        """
        for path, low, high, note in self.NUMERIC_BOUNDS:
            value = self.get(path)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(
                    f"config.yaml: '{path}' must be a number, got {value!r} "
                    f"({type(value).__name__}). Expected {note}."
                )
            if not (low <= value <= high):
                raise ConfigError(
                    f"config.yaml: '{path}' is {value}, outside the valid range "
                    f"{low} to {high}. Expected {note}."
                )

    def _validate_cameras(self) -> set:
        seen_ids = set()
        for name, cam in self._data["cameras"].items():
            if not isinstance(cam, dict):
                raise ConfigError(f"Camera '{name}' must be a mapping of settings, got {type(cam).__name__}.")
            for field in ("id", "source"):
                if field not in cam:
                    raise ConfigError(f"Camera '{name}' is missing required field '{field}'.")
            if cam["id"] in seen_ids:
                raise ConfigError(f"Duplicate camera id '{cam['id']}' - ids must be unique.")
            seen_ids.add(cam["id"])

            self._validate_roi(name, cam.get("roi"))
            self._validate_counting_line(name, cam.get("counting_line"))

            ppm = cam.get("pixels_per_meter")
            if ppm is not None and (not isinstance(ppm, (int, float)) or ppm <= 0):
                raise ConfigError(
                    f"Camera '{name}': pixels_per_meter must be a positive number or null, "
                    f"got {ppm!r}. Leave it null if the camera is not calibrated - speeds "
                    f"are then reported as uncalibrated rather than as a wrong number."
                )
        return seen_ids

    @staticmethod
    def _validate_roi(name: str, roi) -> None:
        if roi is None:
            return
        if not isinstance(roi, dict):
            raise ConfigError(f"Camera '{name}': roi must be a mapping with x1, y1, x2, y2.")
        for key in ("x1", "y1", "x2", "y2"):
            if key in roi and not isinstance(roi[key], int):
                raise ConfigError(f"Camera '{name}': roi.{key} must be a whole number of pixels, got {roi[key]!r}.")
        # An inverted box silently produces an empty crop and zero detections.
        if roi.get("x2", 1) <= roi.get("x1", 0):
            raise ConfigError(
                f"Camera '{name}': roi.x2 ({roi.get('x2')}) must be greater than "
                f"roi.x1 ({roi.get('x1')}) - the box has no width.")
        if roi.get("y2", 1) <= roi.get("y1", 0):
            raise ConfigError(
                f"Camera '{name}': roi.y2 ({roi.get('y2')}) must be greater than "
                f"roi.y1 ({roi.get('y1')}) - the box has no height.")

    @staticmethod
    def _validate_counting_line(name: str, line) -> None:
        if line is None:
            return
        shape_help = (f"Camera '{name}': counting_line must be two points, "
                      f"[[x1, y1], [x2, y2]]. Generate one with "
                      f"'python tools/calibrate.py <video>'.")
        if not isinstance(line, (list, tuple)) or len(line) != 2:
            raise ConfigError(f"{shape_help} Got {line!r}.")
        for point in line:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(isinstance(c, int) for c in point)):
                raise ConfigError(f"{shape_help} Point {point!r} is not [x, y] integers.")
        if tuple(line[0]) == tuple(line[1]):
            raise ConfigError(
                f"Camera '{name}': counting_line's two points are identical, so the "
                f"line has zero length and nothing can ever cross it.")

    def _validate_signals(self, seen_ids: set) -> None:
        for name, sig in self._data.get("traffic_signals", {}).items():
            cam_id = sig.get("camera_id")
            if cam_id and cam_id not in seen_ids:
                raise ConfigError(
                    f"Signal '{name}' references camera_id '{cam_id}', "
                    f"which is not defined. Known cameras: {sorted(seen_ids)}"
                )
            timings = sig.get("timings", {})
            gmin, gmax = timings.get("green_min"), timings.get("green_max")
            if gmin is not None and gmax is not None and gmax < gmin:
                raise ConfigError(
                    f"Signal '{name}': green_max ({gmax}s) is below green_min ({gmin}s) "
                    f"- the phase could never run.")
            if gmin is not None and gmin < 5:
                raise ConfigError(
                    f"Signal '{name}': green_min of {gmin}s is unsafe. Drivers need time "
                    f"to perceive the change and clear the stop line; 5s is the floor.")
            if timings.get("yellow") is not None and timings["yellow"] < 3:
                raise ConfigError(
                    f"Signal '{name}': yellow of {timings['yellow']}s is unsafe. Yellow "
                    f"must cover driver reaction plus stopping distance; 3s is the minimum.")


def load_config(path: str = "config/config.yaml",
                env_file: Optional[str] = ".env") -> Config:
    """
    Load .env, then config.yaml with environment substitution applied.

    Args:
        path: path to config.yaml
        env_file: path to the .env file (skipped silently if absent)
    """
    if env_file and Path(env_file).exists():
        load_dotenv(env_file)
        log.debug("Loaded environment from {}", env_file)

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"Config file not found at '{config_path.resolve()}'. "
            f"Run the app from the project root, or pass an explicit path."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigError(f"'{path}' did not parse to a mapping - check the YAML syntax.")

    return Config(_substitute_env(raw))
