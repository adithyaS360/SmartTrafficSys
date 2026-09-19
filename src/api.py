"""
REST API over the live traffic system.

DESIGN DECISIONS WORTH KNOWING:

1. THE OVERRIDE ENDPOINT DOES NOT SET THE SIGNAL. It requests a phase change,
   which the state machine grants at the next safe moment - after min_green has
   elapsed and by way of yellow and all-red like any other transition. An HTTP
   endpoint that could put a signal straight to green would be a collision
   waiting for anyone with curl and an afternoon. The API is a client of the
   controller, never a way around it.

2. EVERY RESPONSE HAS THE SAME SHAPE: {"data": ..., "meta": ...} on success,
   {"error": {"code", "message"}} on failure. A front end that has to guess
   whether it received a list, an object or an HTML error page grows defensive
   code everywhere. One envelope means one parsing path.

3. ERRORS RETURN JSON, NOT HTML. Flask's default 404 and 500 handlers return
   HTML pages, so a fetch() gets a parse error instead of a diagnosis. The
   handlers below are registered for exactly that reason.

4. QUERY PARAMETERS ARE BOUNDED. `minutes` is clamped rather than trusted:
   ?minutes=99999999 would otherwise pull every row in the database into memory
   and serialise it. That is not a hypothetical - it is the first thing anyone
   tries when they see a number in a URL.

NOT IMPLEMENTED, DELIBERATELY: authentication. Every endpoint here is open, which
is fine on localhost and unacceptable the moment this is exposed to a network -
the override endpoint in particular. Say so in the report rather than leaving a
reader to assume it was overlooked; "out of scope, and here is what it would
need" reads as judgement, silence reads as a gap.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_MINUTES = 60 * 24 * 7          # one week - the ceiling for any range query
MAX_HOURS = 24 * 90


def ok(data: Any, **meta) -> tuple:
    payload = {"data": data, "meta": {"generated_at": datetime.now(timezone.utc).isoformat(), **meta}}
    return jsonify(payload), 200


def fail(code: str, message: str, status: int = 400) -> tuple:
    return jsonify({"error": {"code": code, "message": message}}), status


def _clamp(value: Optional[str], default: int, maximum: int, name: str) -> int:
    """Parse and bound an integer query parameter, or raise ValueError."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer, got '{value}'")
    if parsed < 1:
        raise ValueError(f"'{name}' must be at least 1, got {parsed}")
    return min(parsed, maximum)


def _snapshot_json(snap) -> Optional[Dict[str, Any]]:
    if snap is None:
        return None
    stamp = getattr(snap, "timestamp", None) or getattr(snap, "bucket_start", None)
    return {
        "camera_id": snap.camera_id,
        "timestamp": stamp.isoformat() if stamp else None,
        "vehicle_count": round(float(getattr(snap, "vehicle_count", 0)), 2),
        "pedestrian_count": round(float(getattr(snap, "pedestrian_count", 0)), 2),
        "queue_length": int(getattr(snap, "queue_length", 0)),
        "flow_rate": round(float(getattr(snap, "flow_rate", 0)), 2),
        "avg_dwell_seconds": round(float(getattr(snap, "avg_dwell_seconds", 0)), 2),
        # Stays null when the camera is uncalibrated. The dashboard renders
        # "uncalibrated" rather than a zero, because a zero would read as a jam.
        "avg_speed_kmh": (round(float(snap.avg_speed_kmh), 1)
                          if getattr(snap, "avg_speed_kmh", None) is not None else None),
        "class_breakdown": getattr(snap, "class_breakdown", None) or {},
    }


def create_api(system) -> Blueprint:
    """
    Build the API blueprint around a running TrafficSystem.

    `system` is passed in rather than imported so the API can be tested against
    a stub, and so there is exactly one owner of the live state.
    """
    api = Blueprint("api", __name__, url_prefix="/api")

    # ---- health ------------------------------------------------------

    @api.get("/health")
    def health():
        """
        Liveness plus the things that actually go wrong.

        A health check that only says "the web server is up" is close to
        useless: the web server being up is the one thing you already know,
        because it answered. What matters is whether frames are still arriving
        and whether rows are still being written - the two failures that leave
        the dashboard looking perfectly healthy while showing stale data.
        """
        stats = system.stats()
        stale = stats["seconds_since_last_snapshot"]
        healthy = system.running and (stale is None or stale < 30)
        # ok() already returns (body, status); rebuild rather than nesting it,
        # because a degraded system must answer 503 for a monitor to notice.
        body, _ = ok({
            "status": "healthy" if healthy else "degraded",
            "running": system.running,
            "mode": system.mode,
            "uptime_seconds": round(stats["uptime_seconds"], 1),
            "seconds_since_last_snapshot": stale,
            "frames_processed": stats["frames_processed"],
            "rows_written": stats["rows_written"],
            "cameras": stats["cameras"],
        })
        return body, (200 if healthy else 503)

    # ---- live state --------------------------------------------------

    @api.get("/status")
    def status():
        """Everything the dashboard needs for one render."""
        controller = system.controller
        signal = None
        if controller is not None:
            signal = {
                "signal_id": controller.signal_id,
                "state": controller.state.value,
                "current_phase": controller.current_phase.id,
                "phase_name": controller.current_phase.name or controller.current_phase.id,
                "elapsed_seconds": round(controller.elapsed(system.clock()), 1),
                "min_green": controller.current_phase.min_green,
                "max_green": controller.current_phase.max_green,
                "strategy": controller.strategy.name,
                "approaches": {cam: controller.signal_for(cam)
                               for cam in system.camera_ids()},
                "pending_override": system.pending_override,
            }
        return ok({
            "signal": signal,
            "cameras": {cam: _snapshot_json(snap)
                        for cam, snap in system.latest_snapshots().items()},
            "totals": system.totals(),
        })

    @api.get("/cameras")
    def cameras():
        return ok([{"id": cam, "name": system.camera_name(cam)} for cam in system.camera_ids()])

    @api.get("/cameras/<camera_id>/series")
    def series(camera_id: str):
        """Recent buckets for one camera - the dashboard's chart."""
        if camera_id not in system.camera_ids():
            return fail("unknown_camera",
                        f"No camera '{camera_id}'. Known: {sorted(system.camera_ids())}", 404)
        try:
            minutes = _clamp(request.args.get("minutes"), 60, MAX_MINUTES, "minutes")
        except ValueError as exc:
            return fail("bad_request", str(exc), 400)

        rows = system.repository.series(camera_id, minutes=minutes)
        return ok([{
            "timestamp": r.bucket_start.isoformat(),
            "vehicle_count": r.vehicle_count,
            "queue_length": r.queue_length,
            "crossings": r.crossings,
            "flow_rate": r.flow_rate,
            "avg_dwell_seconds": r.avg_dwell_seconds,
            "avg_speed_kmh": r.avg_speed_kmh,
        } for r in rows], camera_id=camera_id, minutes=minutes, count=len(rows))

    @api.get("/cameras/<camera_id>/history")
    def history(camera_id: str):
        """Per-minute roll-up - the same view the model trains on."""
        if camera_id not in system.camera_ids():
            return fail("unknown_camera", f"No camera '{camera_id}'", 404)
        try:
            hours = _clamp(request.args.get("hours"), 24, MAX_HOURS, "hours")
        except ValueError as exc:
            return fail("bad_request", str(exc), 400)
        rows = system.repository.per_minute(camera_id, hours=hours)
        return ok(rows, camera_id=camera_id, hours=hours, count=len(rows))

    # ---- signals -----------------------------------------------------

    @api.get("/signals/<signal_id>/events")
    def events(signal_id: str):
        try:
            hours = _clamp(request.args.get("hours"), 1, MAX_HOURS, "hours")
        except ValueError as exc:
            return fail("bad_request", str(exc), 400)
        rows = system.repository.signal_events(signal_id, hours=hours)
        return ok([{
            "timestamp": e.timestamp.isoformat(),
            "phase": e.phase,
            "duration_seconds": e.duration_seconds,
            "reason": e.reason,
            "queue_at_decision": e.queue_at_decision,
            "flow_at_decision": e.flow_at_decision,
        } for e in rows], signal_id=signal_id, hours=hours, count=len(rows))

    @api.get("/signals/<signal_id>/decisions")
    def decisions(signal_id: str):
        """In-memory decision log - available even with no database configured."""
        controller = system.controller
        if controller is None or controller.signal_id != signal_id:
            return fail("unknown_signal", f"No signal '{signal_id}'", 404)
        try:
            limit = _clamp(request.args.get("limit"), 50, 500, "limit")
        except ValueError as exc:
            return fail("bad_request", str(exc), 400)
        recent = controller.decisions[-limit:]
        return ok([{
            "timestamp": d.timestamp.isoformat(),
            "phase_id": d.phase_id,
            "state": d.state.value,
            "duration_seconds": d.duration_seconds,
            "reason": d.reason,
            "queue_at_decision": d.queue_at_decision,
            "flow_at_decision": d.flow_at_decision,
        } for d in reversed(recent)],
            signal_id=signal_id,
            reason_counts=controller.reason_counts(),
            total_decisions=len(controller.decisions))

    @api.post("/signals/<signal_id>/override")
    def override(signal_id: str):
        """
        Request that a given phase be served next.

        THE REQUEST IS QUEUED, NOT APPLIED. The controller grants it at the next
        safe opportunity: after min_green on the running phase, through yellow
        and all-red like any other transition. There is deliberately no way to
        force an immediate change through this API - see the module docstring.
        """
        controller = system.controller
        if controller is None or controller.signal_id != signal_id:
            return fail("unknown_signal", f"No signal '{signal_id}'", 404)

        body = request.get_json(silent=True) or {}
        phase_id = body.get("phase_id")
        if not phase_id:
            return fail("bad_request",
                        "Body must be JSON containing 'phase_id', e.g. {\"phase_id\": \"ew\"}")

        known = [p.id for p in controller.phases]
        if phase_id not in known:
            return fail("unknown_phase", f"No phase '{phase_id}'. Known: {known}", 404)

        accepted = system.request_override(phase_id)
        return ok({
            "accepted": accepted,
            "requested_phase": phase_id,
            "current_phase": controller.current_phase.id,
            "note": ("Queued. It will be served after the running phase reaches "
                     "min_green and clears through yellow and all-red."),
        })

    @api.delete("/signals/<signal_id>/override")
    def clear_override(signal_id: str):
        system.request_override(None)
        return ok({"accepted": True, "requested_phase": None})

    # ---- prediction --------------------------------------------------

    @api.get("/cameras/<camera_id>/forecast")
    def forecast(camera_id: str):
        """
        Predicted flow at the model's horizon, when a model is loaded.

        Returns 503 rather than a fabricated number when no model is available.
        A dashboard showing a confident forecast produced by nothing is worse
        than one showing "no model loaded".
        """
        if camera_id not in system.camera_ids():
            return fail("unknown_camera", f"No camera '{camera_id}'", 404)
        prediction = system.forecast(camera_id)
        if prediction is None:
            return fail("no_model",
                        "No forecasting model is loaded. Train one with "
                        "'python tools/train.py --save models/flow' and restart "
                        "with --model models/flow.keras", 503)
        return ok({"camera_id": camera_id, **prediction})

    @api.errorhandler(500)
    def server_error(exc):
        log.exception("Unhandled API error: {}", exc)
        return fail("internal_error", "Something went wrong; check the server log", 500)

    return api


def register_error_handlers(app) -> None:
    """
    Install JSON error handlers at APPLICATION level.

    WHY NOT ON THE BLUEPRINT: a blueprint's 404 handler only fires for a 404
    raised inside one of its own views. A request to a URL that matches no route
    at all never reaches any blueprint, so routing 404s fall through to Flask's
    default HTML error page - and a fetch() then fails on JSON.parse with a
    message about unexpected '<', which says nothing about the real problem.

    That made the blueprint quietly dependent on the host app registering its
    own handler. It did, so live testing passed and only the unit test, which
    builds a bare app, caught it. Registering here makes create_api() complete
    on its own.
    """

    @app.errorhandler(404)
    def _not_found(_):
        return fail("not_found", "No such endpoint", 404)

    @app.errorhandler(405)
    def _bad_method(_):
        return fail("method_not_allowed", "Wrong HTTP method for this endpoint", 405)

    @app.errorhandler(500)
    def _server_error(exc):
        log.exception("Unhandled error: {}", exc)
        return fail("internal_error", "Something went wrong; check the server log", 500)
