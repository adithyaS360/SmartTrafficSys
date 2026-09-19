"""
Verification of Phase 4: the REST API contract and override safety.

Uses Flask's test client against a stub system, so no server, camera or
database is needed. The one test that matters most is the last group: that the
override endpoint cannot make the signal do anything the state machine would
not have allowed on its own.
"""
import sys, types, json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = types.ModuleType("cv2"); cv2.__getattr__ = lambda n: (lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0; cv2.CAP_PROP_BUFFERSIZE = 38; cv2.CAP_PROP_POS_FRAMES = 1
cv2.VideoCapture = object
sys.modules["cv2"] = cv2
ultra = types.ModuleType("ultralytics"); ultra.YOLO = object
sys.modules["ultralytics"] = ultra

from flask import Flask

from src.api import create_api, register_error_handlers
from src.traffic_controller import (AdaptiveStrategy, IntersectionController,
                                    Phase, SignalState)

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))


class StubRepo:
    def series(self, camera_id, minutes=60): return []
    def per_minute(self, camera_id, hours=24): return []
    def signal_events(self, signal_id, hours=1): return []
    def record_signal_event(self, **kw): return None


class StubSystem:
    """Minimal stand-in exposing the surface the API consumes."""

    def __init__(self, stale=0.0, running=True, with_controller=True):
        self.mode = "simulated"
        self.running = running
        self.repository = StubRepo()
        self._stale = stale
        self._t = 0.0
        self.pending_override = None
        phases = [Phase(id="ns", name="North-South", camera_ids=["north", "south"],
                        min_green=10.0, max_green=40.0, yellow=3.0, all_red=2.0),
                  Phase(id="ew", name="East-West", camera_ids=["east", "west"],
                        min_green=10.0, max_green=40.0, yellow=3.0, all_red=2.0)]
        self.controller = (IntersectionController("sig_1", phases, AdaptiveStrategy())
                           if with_controller else None)

    def clock(self): return self._t
    def camera_ids(self): return ["east", "north", "south", "west"]
    def camera_name(self, c): return c.title()
    def latest_snapshots(self):
        return {c: types.SimpleNamespace(
            camera_id=c, timestamp=datetime.now(timezone.utc), vehicle_count=3.0,
            pedestrian_count=0.0, queue_length=4, flow_rate=9.5,
            avg_dwell_seconds=7.5, avg_speed_kmh=None, class_breakdown={"car": 3})
            for c in self.camera_ids()}
    def totals(self): return {"queue_length": 16, "flow_rate": 38.0, "vehicle_count": 12.0}
    def stats(self):
        return {"uptime_seconds": 12.0, "seconds_since_last_snapshot": self._stale,
                "frames_processed": 400, "rows_written": 9, "cameras": 4}
    def request_override(self, pid):
        ok = self.controller.request_phase(pid) if self.controller else False
        self.pending_override = self.controller.pending_override if self.controller else None
        return ok
    def forecast(self, camera_id): return None


def client_for(system):
    app = Flask(__name__)
    app.register_blueprint(create_api(system))
    register_error_handlers(app)
    app.config["TESTING"] = True
    return app.test_client()


def body(res):
    return json.loads(res.data)


print("\n=== 1. Every response uses the same envelope ===")
c = client_for(StubSystem())
for path in ("/api/status", "/api/cameras", "/api/health"):
    b = body(c.get(path))
    check(f"{path} returns data+meta", "data" in b and "meta" in b, f"keys {sorted(b)}")
b = body(c.get("/api/cameras/nope/series"))
check("errors return an error object with a code",
      "error" in b and {"code", "message"} <= set(b["error"]), f"{b}")


print("\n=== 2. Health reports 503 when data has gone stale ===")
check("fresh system is 200 healthy",
      client_for(StubSystem(stale=1.0)).get("/api/health").status_code == 200)
res = client_for(StubSystem(stale=120.0)).get("/api/health")
check("stale system is 503 degraded",
      res.status_code == 503 and body(res)["data"]["status"] == "degraded",
      f"status {res.status_code}")
check("a stopped system is degraded even with fresh data",
      client_for(StubSystem(stale=0.0, running=False)).get("/api/health").status_code == 503)


print("\n=== 3. Query parameters are validated and bounded ===")
c = client_for(StubSystem())
res = c.get("/api/cameras/north/series?minutes=abc")
check("non-integer rejected with 400", res.status_code == 400
      and "must be an integer" in body(res)["error"]["message"])
res = c.get("/api/cameras/north/series?minutes=0")
check("zero rejected", res.status_code == 400, body(res).get("error", {}).get("message", "")[:40])
res = c.get("/api/cameras/north/series?minutes=99999999")
check("absurd range clamped, not executed", body(res)["meta"]["minutes"] == 60 * 24 * 7,
      f"clamped to {body(res)['meta']['minutes']}")
check("default applies when omitted",
      body(c.get("/api/cameras/north/series"))["meta"]["minutes"] == 60)


print("\n=== 4. Unknown resources 404 as JSON, never HTML ===")
for path in ("/api/cameras/ghost/series", "/api/cameras/ghost/history",
             "/api/cameras/ghost/forecast", "/api/nope"):
    res = c.get(path)
    is_json = res.headers.get("Content-Type", "").startswith("application/json")
    check(f"{path} -> JSON {res.status_code}", res.status_code in (404, 503) and is_json)


print("\n=== 5. Forecast refuses to invent a number ===")
res = c.get("/api/cameras/north/forecast")
check("no model returns 503, not a fabricated value",
      res.status_code == 503 and body(res)["error"]["code"] == "no_model")
check("and says how to fix it", "train.py" in body(res)["error"]["message"])


print("\n=== 6. Override is validated ===")
res = c.post("/api/signals/sig_1/override", json={})
check("missing phase_id rejected", res.status_code == 400)
res = c.post("/api/signals/sig_1/override", json={"phase_id": "bogus"})
check("unknown phase rejected", res.status_code == 404
      and body(res)["error"]["code"] == "unknown_phase")
res = c.post("/api/signals/wrong/override", json={"phase_id": "ns"})
check("unknown signal rejected", res.status_code == 404)


print("\n=== 7. THE IMPORTANT ONE: override cannot force an unsafe change ===")
system = StubSystem()
c = client_for(system)
ctrl = system.controller
snaps = {cam: types.SimpleNamespace(queue_length=6, flow_rate=12.0)
         for cam in system.camera_ids()}

ctrl.tick(0.0, snaps)                       # start the cycle on 'ns'
start_phase = ctrl.current_phase.id
res = c.post("/api/signals/sig_1/override", json={"phase_id": "ew"})
check("request accepted", body(res)["data"]["accepted"] is True)
check("but the running phase has NOT changed", ctrl.current_phase.id == start_phase,
      f"still {ctrl.current_phase.id}")
check("and the signal is still green, not jumped", ctrl.state is SignalState.GREEN)

# Step forward and record every state the machine passes through.
seen = []
for step in range(1, 400):
    system._t = step * 0.25
    ctrl.tick(system._t, snaps)
    seen.append((round(system._t, 2), ctrl.current_phase.id, ctrl.state))
    if ctrl.current_phase.id == "ew" and ctrl.state is SignalState.GREEN:
        break

switch_time = seen[-1][0]
check("the override is eventually honoured", ctrl.current_phase.id == "ew",
      f"switched at t={switch_time}s")
check("it waited for min_green first", switch_time >= 10.0,
      f"switched at {switch_time}s, min_green=10s")

states = [s for _, _, s in seen]
check("it passed through YELLOW", SignalState.YELLOW in states)
check("it passed through ALL_RED", SignalState.ALL_RED in states)
check("it NEVER went green->all_red directly",
      not any(a is SignalState.GREEN and b is SignalState.ALL_RED
              for a, b in zip(states, states[1:])))

yellow_first = states.index(SignalState.YELLOW)
allred_first = states.index(SignalState.ALL_RED)
check("yellow came before all-red, in that order", yellow_first < allred_first,
      f"yellow at index {yellow_first}, all-red at {allred_first}")

check("the override cleared itself after being served",
      ctrl.pending_override is None)

res = c.delete("/api/signals/sig_1/override")
check("override can be cleared", body(res)["data"]["requested_phase"] is None)


print("\n=== 8. Degrades cleanly with no controller configured ===")
c2 = client_for(StubSystem(with_controller=False))
b = body(c2.get("/api/status"))
check("status still serves camera data", b["data"]["signal"] is None
      and len(b["data"]["cameras"]) == 4)
check("override on a signal-less system 404s",
      c2.post("/api/signals/sig_1/override", json={"phase_id": "ns"}).status_code == 404)

print(f"\n{'='*62}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 62)
sys.exit(1 if FAIL else 0)
