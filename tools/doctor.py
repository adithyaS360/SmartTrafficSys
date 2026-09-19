"""
Check that this installation is actually working, and say what to do if not.

    python tools/doctor.py

WHY THIS EXISTS: the failure mode of a project like this is not a crash, it is
coming back after three weeks and finding that something does not run, with no
memory of how it was set up. Every check below reports what it found and, when
something is wrong, the exact command that fixes it.

Run it after cloning, after changing config.yaml, and whenever anything behaves
oddly. It touches nothing - it only reads.

Exit code: 0 all clear, 1 warnings only, 2 something is broken.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(status: str, title: str, detail: str = "", fix: str = "") -> None:
    results.append((status, title, detail, fix))


def section(name: str) -> None:
    print(f"\n{name}")
    print("-" * len(name))


def show(status: str, title: str, detail: str = "", fix: str = "") -> None:
    record(status, title, detail, fix)
    print(f"  [{status}] {title}" + (f"  {detail}" if detail else ""))
    if fix and status != OK:
        for line in fix.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------

def check_python() -> None:
    section("Python")
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) < (3, 9):
        show(FAIL, "Python version", version,
             "This project needs Python 3.9 or newer. Install a newer Python.")
    elif (major, minor) >= (3, 13):
        show(WARN, "Python version", version,
             "TensorFlow often lags the newest Python by months. If\n"
             "'pip install -r requirements-ml.txt' finds no build, that is why.\n"
             "The rest of the project is unaffected; use --model gbr for Phase 3b.")
    else:
        show(OK, "Python version", version)


def check_dependencies() -> None:
    section("Dependencies")
    core = {
        "cv2": "opencv-python", "numpy": "numpy", "scipy": "scipy",
        "ultralytics": "ultralytics", "sqlalchemy": "SQLAlchemy",
        "alembic": "alembic", "flask": "Flask", "yaml": "PyYAML",
        "dotenv": "python-dotenv", "loguru": "loguru",
    }
    missing = []
    for module, package in core.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        show(FAIL, "Core packages", f"missing: {', '.join(missing)}",
             "pip install -r requirements.txt")
    else:
        show(OK, "Core packages", f"all {len(core)} present")

    optional = {"tensorflow": "TensorFlow (LSTM)", "sklearn": "scikit-learn (gradient boosting)"}
    for module, label in optional.items():
        try:
            importlib.import_module(module)
            show(OK, f"Optional: {label}", "installed")
        except ImportError:
            show(WARN, f"Optional: {label}", "not installed",
                 "Only needed for Phase 3b prediction:\n  pip install -r requirements-ml.txt")


def check_config():
    section("Configuration")
    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        show(FAIL, "config/config.yaml", "not found",
             "The file should be in the repository. Restore it from git.")
        return None

    if not (ROOT / ".env").exists():
        show(WARN, ".env", "not found",
             "Only required for PostgreSQL. On SQLite the defaults are fine.\n"
             "  copy .env.example .env")
    else:
        show(OK, ".env", "present")

    try:
        from src.utils.config_loader import load_config
        config = load_config(str(config_path), env_file=str(ROOT / ".env"))
    except Exception as exc:  # noqa: BLE001
        show(FAIL, "config.yaml validates", str(exc)[:160],
             "Fix the message above, then run this again.")
        return None

    show(OK, "config.yaml validates",
         f"{len(config.cameras)} camera(s), {len(config.signals)} signal phase(s)")

    backend = config.get("database.type", "sqlite")
    show(OK, "Database backend", backend)

    # Calibration: the two values that cannot be guessed.
    for name, cam in config.cameras.items():
        issues = []
        if not cam.get("counting_line"):
            issues.append("no counting_line (flow rate will be meaningless)")
        if cam.get("pixels_per_meter") in (None, 0):
            issues.append("no pixels_per_meter (speeds reported as uncalibrated)")
        if issues:
            show(WARN, f"Camera '{cam['id']}' calibration", "; ".join(issues),
                 f"python tools/calibrate.py {cam.get('source')} --name {name}")
        else:
            show(OK, f"Camera '{cam['id']}' calibration", "counting line and scale set")

    if len(config.signals) < 2:
        show(WARN, "Signal phases", f"{len(config.signals)} configured",
             "An intersection needs at least two phases, or no controller runs in\n"
             "--mode live. This is expected until you have junction footage;\n"
             "'python app.py --mode simulated' is unaffected.")
    else:
        show(OK, "Signal phases", f"{len(config.signals)} configured")

    return config


def check_database(config) -> None:
    section("Database")
    if config is None:
        show(WARN, "Database", "skipped - config did not load")
        return
    try:
        from sqlalchemy import inspect, text
        from src.database.db_handler import Database, TrafficRepository
        db = Database(config.database_url())
        tables = set(inspect(db.engine).get_table_names())
    except Exception as exc:  # noqa: BLE001
        show(FAIL, "Database reachable", str(exc)[:140],
             "For SQLite this usually means the data/ folder is missing.\n"
             "For PostgreSQL, check the server is running and .env is correct.")
        return

    show(OK, "Database reachable", config.database_url().split("://")[0])

    expected = {"cameras", "traffic_snapshots", "signal_events"}
    if not expected <= tables:
        show(FAIL, "Schema applied", f"missing: {sorted(expected - tables)}",
             "alembic upgrade head")
        return
    show(OK, "Schema applied", f"{len(expected)} tables")

    if "alembic_version" not in tables:
        show(WARN, "Migration tracking", "no alembic_version table",
             "The schema was created with create_all() rather than migrations.\n"
             "That works, but schema changes later will not apply cleanly:\n"
             "  alembic stamp head")
    else:
        show(OK, "Migration tracking", "alembic_version present")

    try:
        with db.session() as s:
            rows = s.execute(text("SELECT COUNT(*) FROM traffic_snapshots")).scalar() or 0
            cams = s.execute(text("SELECT COUNT(*) FROM cameras")).scalar() or 0
            span = s.execute(text(
                "SELECT MIN(bucket_start), MAX(bucket_start) FROM traffic_snapshots")).first()
    except Exception as exc:  # noqa: BLE001
        show(WARN, "Reading data", str(exc)[:120])
        return

    if rows == 0:
        show(WARN, "Stored history", "empty",
             "Nothing to train a model on yet. Either run the system, or:\n"
             "  python tools/simulate.py generate --days 21")
    else:
        detail = f"{rows:,} rows across {cams} camera(s)"
        if span and span[0]:
            detail += f", {str(span[0])[:16]} to {str(span[1])[:16]}"
        show(OK, "Stored history", detail)
        # The model needs roughly one seasonal period plus a window.
        if rows < 2000:
            show(WARN, "Enough data to train?", f"{rows:,} rows is thin",
                 "Phase 3b needs at least a day or two of history:\n"
                 "  python tools/simulate.py generate --days 21")
    db.dispose()


def check_assets() -> None:
    section("Model and footage")
    weights = list(ROOT.glob("*.pt")) + list((ROOT / "models").glob("*.pt"))
    if weights:
        size = weights[0].stat().st_size / 1e6
        show(OK, "YOLO weights", f"{weights[0].name} ({size:.1f} MB)")
    else:
        show(WARN, "YOLO weights", "not downloaded yet",
             "ultralytics fetches them automatically on first detection run,\n"
             "so this resolves itself - but it needs internet the first time.")

    videos = [p for p in (ROOT / "data").glob("*")
              if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if videos:
        listing = ", ".join(f"{v.name} ({v.stat().st_size / 1e6:.0f} MB)" for v in videos[:4])
        show(OK, "Footage", listing)
        show(WARN, "Footage assessed?", "run the checker before relying on a clip",
             f"python tools/check_footage.py data/{videos[0].name}")
    else:
        show(WARN, "Footage", "no video files in data/",
             "Needed only for --mode live. See data/README.md.\n"
             "'python app.py --mode simulated' works without any footage.")


def check_tests() -> None:
    section("Tests")
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if not files:
        show(FAIL, "Test suite", "no tests found", "The tests/ folder should not be empty.")
        return
    show(OK, "Test suite", f"{len(files)} file(s): {', '.join(f.stem for f in files)}")
    show(OK, "Run them with", "python tests/run_all.py")


def main() -> None:
    # Quiet the application's own logging: this tool's output IS the report,
    # and an INFO line about the database connecting in the middle of it just
    # makes the result harder to read.
    try:
        from src.utils.logger import setup_logging
        setup_logging(level="ERROR")
    except Exception:  # noqa: BLE001
        pass

    print("=" * 66)
    print("  Smart Traffic Management System - installation check")
    print("=" * 66)

    check_python()
    check_dependencies()
    config = check_config()
    check_database(config)
    check_assets()
    check_tests()

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]

    print("\n" + "=" * 66)
    print(f"  {len(results) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warning(s), {len(fails)} failure(s)")
    if fails:
        print("\n  BROKEN - fix these first:")
        for _, title, detail, _ in fails:
            print(f"    - {title}: {detail}")
    elif warns:
        print("\n  WORKING, with warnings. Most are expected before you have")
        print("  real footage; none of them stop 'python app.py --mode simulated'.")
    else:
        print("\n  All clear.")
    print("=" * 66 + "\n")
    sys.exit(2 if fails else (1 if warns else 0))


if __name__ == "__main__":
    main()
