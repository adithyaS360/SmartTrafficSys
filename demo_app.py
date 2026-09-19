"""
Public demo server: fixed-time vs adaptive signal control, side by side.

    python demo_app.py                  # local, http://127.0.0.1:5001
    gunicorn -w 1 -t 120 demo_app:app   # production (ONE worker - see below)

DEPLOYMENT NOTES, because each of these is a way it would otherwise break:

ONE WORKER, ALWAYS. The simulation runs in a background thread inside the web
process. Under `gunicorn -w 4` you would get four independent simulations, four
controllers disagreeing about the current phase, and a page showing whichever
worker happened to answer that request. Nothing would crash; the numbers would
just quietly stop meaning anything. Threads, not workers, is the right dial
here - the work is a fraction of a millisecond per tick and the rest is idle.

NO DATABASE. Everything lives in memory and the demo seeds itself at startup.
That is not laziness: free hosting gives you an ephemeral filesystem (so SQLite
is wiped on every restart) and a free Postgres that expires after thirty days
(so the demo would silently die a month after you shared the link). Holding no
state at all is the only option that still works in ninety days without
maintenance.

COLD STARTS. Free instances sleep after ~15 minutes idle and take up to a
minute to wake. The first visitor after a nap would otherwise land on an empty
chart and a 0.0s delay reading, which looks broken. So boot runs a few
simulated minutes instantly - the simulator does an hour in well under a second
- and the page is immediately worth looking at.
"""

import os
import threading
import time

from flask import Flask, jsonify, render_template, request

from src.demo import DEMAND_PRESETS, ParallelDemo
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# Simulated seconds per real second. A signal cycle is ~30 simulated seconds;
# at 8x a visitor sees a full cycle in about four seconds and a queue build and
# clear inside fifteen. Real time would be accurate and unwatchable.
SPEED = float(os.environ.get("DEMO_SPEED", "8"))
# Simulated minutes to run at startup so the page is never empty on arrival.
WARMUP_MINUTES = float(os.environ.get("DEMO_WARMUP_MINUTES", "12"))

demo = ParallelDemo()
_lock = threading.Lock()
_started = False


def warmup() -> None:
    """Run some simulated time instantly so a cold start looks alive."""
    began = time.time()
    steps = int(WARMUP_MINUTES * 60 / demo.dt)
    with _lock:
        for _ in range(steps):
            demo.tick()
    log.info("Warmed up {:.0f} simulated minutes in {:.2f}s",
             WARMUP_MINUTES, time.time() - began)


def run_loop() -> None:
    """Advance the simulation continuously in the background."""
    interval = demo.dt / SPEED
    while True:
        began = time.time()
        with _lock:
            demo.tick()
            demo.maybe_idle_reset()
        time.sleep(max(0.0, interval - (time.time() - began)))


def ensure_started() -> None:
    """
    Start the simulation exactly once, on first use.

    Done lazily rather than at import so that `gunicorn --preload`, a health
    probe, or an import from a test does not each spawn its own loop.
    """
    global _started
    if _started:
        return
    with _lock:
        if _started:
            return
        _started = True
    warmup()
    threading.Thread(target=run_loop, name="demo-sim", daemon=True).start()
    log.info("Demo simulation running at {}x", SPEED)


app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def index():
    ensure_started()
    return render_template("demo.html", speed=SPEED)


@app.get("/api/demo")
def state():
    ensure_started()
    with _lock:
        return jsonify({"data": demo.snapshot()})


@app.post("/api/demo/demand")
def set_demand():
    """Change the traffic level. Resets both junctions so the comparison stays fair."""
    ensure_started()
    body = request.get_json(silent=True) or {}
    level = body.get("demand")
    if level not in DEMAND_PRESETS:
        return jsonify({"error": {
            "code": "bad_demand",
            "message": f"Unknown demand '{level}'. Valid: {sorted(DEMAND_PRESETS)}",
        }}), 400
    with _lock:
        demo.reset(demand=level)
        demo.touch()
    warmup()
    with _lock:
        return jsonify({"data": demo.snapshot()})


@app.post("/api/demo/reset")
def reset():
    ensure_started()
    with _lock:
        demo.reset()
        demo.touch()
    warmup()
    with _lock:
        return jsonify({"data": demo.snapshot()})


@app.get("/api/health")
def health():
    """Render's health check hits this. Kept trivial so it never wakes the sim."""
    return jsonify({"status": "ok", "running": _started, "speed": SPEED}), 200


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": {"code": "not_found", "message": "No such endpoint"}}), 404


@app.errorhandler(500)
def server_error(exc):
    log.exception("Unhandled error: {}", exc)
    return jsonify({"error": {"code": "internal_error", "message": "Something went wrong"}}), 500


if __name__ == "__main__":
    setup_logging(level="INFO")
    port = int(os.environ.get("PORT", "5001"))
    print(f"\n  Demo: http://127.0.0.1:{port}   (running at {SPEED:g}x speed)\n")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
else:
    setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
