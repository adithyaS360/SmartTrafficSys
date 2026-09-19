"""
Verification of Phase 3: signal safety invariants, control behaviour, and the
simulator's traffic model.

The safety tests matter more than the performance ones. A controller that times
badly is a worse controller; one that can show green to conflicting approaches
is a hazard. Those properties are asserted here against an ADVERSARIAL strategy
that actively tries to violate them, because a strategy that behaves itself
proves nothing about whether the machine would stop one that did not.
"""
import sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = types.ModuleType("cv2"); cv2.__getattr__ = lambda n: (lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0; cv2.CAP_PROP_BUFFERSIZE = 38; cv2.CAP_PROP_POS_FRAMES = 1
cv2.VideoCapture = object
sys.modules["cv2"] = cv2
ultra = types.ModuleType("ultralytics"); ultra.YOLO = object
sys.modules["ultralytics"] = ultra

from src.simulator import (ApproachSim, TrafficSimulator, compare_strategies,
                           diurnal_multiplier, _poisson)
from src.traffic_controller import (AdaptiveStrategy, ControlStrategy, Decision,
                                    FixedTimeStrategy, IntersectionController,
                                    Phase, SignalState)

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))

def two_phases(min_green=10.0, max_green=40.0):
    return [Phase(id="ns", camera_ids=["north", "south"], min_green=min_green,
                  max_green=max_green, yellow=3.0, all_red=2.0),
            Phase(id="ew", camera_ids=["east", "west"], min_green=min_green,
                  max_green=max_green, yellow=3.0, all_red=2.0)]

class AlwaysStop(ControlStrategy):
    """Adversarial: demands the phase end on every single tick."""
    name = "always_stop"
    def should_extend(self, phase, elapsed, snapshots): return (False, "adversarial_stop")

class AlwaysExtend(ControlStrategy):
    """Adversarial: never lets a phase end."""
    name = "always_extend"
    def should_extend(self, phase, elapsed, snapshots): return (True, "adversarial_extend")


print("\n=== 1. Unsafe configurations are rejected at construction ===")
for label, kwargs in [
    ("min_green below 5s", dict(id="p", camera_ids=["a"], min_green=2.0)),
    ("yellow below 3s", dict(id="p", camera_ids=["a"], yellow=1.0)),
    ("max_green below min_green", dict(id="p", camera_ids=["a"], min_green=20.0, max_green=10.0)),
    ("phase serving no approach", dict(id="p", camera_ids=[])),
]:
    try:
        Phase(**kwargs); check(label + " rejected", False)
    except ValueError as e:
        check(label + " rejected", True, str(e)[:52])

try:
    IntersectionController("s", two_phases()[:1], FixedTimeStrategy())
    check("single-phase intersection rejected", False)
except ValueError as e:
    check("single-phase intersection rejected", True, str(e)[:52])

try:
    IntersectionController("s", [Phase(id="a", camera_ids=["north", "east"]),
                                 Phase(id="b", camera_ids=["east", "west"])],
                           FixedTimeStrategy())
    check("approach in two phases rejected", False)
except ValueError as e:
    check("approach in two phases rejected", True, str(e)[:52])


print("\n=== 2. Green NEVER becomes red without yellow (adversarial strategy) ===")
ctrl = IntersectionController("s", two_phases(), AlwaysStop(), decision_interval=0.5)
seq, per_cam = [], {c: [] for c in ("north", "south", "east", "west")}
for t in range(0, 4000):
    ctrl.tick(t * 0.5, {})
    seq.append(ctrl.state)
    for c in per_cam:
        per_cam[c].append(ctrl.signal_for(c))

illegal = [(a, b) for a, b in zip(seq, seq[1:])
           if a is SignalState.GREEN and b is SignalState.ALL_RED]
check("GREEN -> ALL_RED never occurs", not illegal, f"{len(illegal)} violations")

bad = []
for cam, states in per_cam.items():
    for a, b in zip(states, states[1:]):
        if a == "green" and b == "red":
            bad.append(cam)
check("no approach goes green -> red directly", not bad, f"{set(bad)}")

order_ok = all(b in {"yellow", "green"} for a, b in zip(seq, seq[1:]) if a is SignalState.GREEN)
check("GREEN may only be followed by YELLOW or GREEN", order_ok)


print("\n=== 3. Exactly one phase is green at any moment ===")
ctrl3 = IntersectionController("s", two_phases(), AlwaysExtend())
violations = 0
for t in range(3000):
    ctrl3.tick(t * 0.5, {})
    greens = sum(1 for c in ("north", "south", "east", "west") if ctrl3.is_green_for(c))
    # north+south share a phase, so 0 or 2 approaches green - never a mix of phases
    if greens not in (0, 2):
        violations += 1
check("conflicting approaches never green together", violations == 0, f"{violations} ticks bad")


print("\n=== 4. min_green holds even when the strategy says stop immediately ===")
ctrl4 = IntersectionController("s", two_phases(min_green=12.0), AlwaysStop(), decision_interval=0.1)
for t in range(2000):
    ctrl4.tick(t * 0.1, {})
greens = [d for d in ctrl4.decisions if d.state is SignalState.YELLOW]
shortest = min((d.duration_seconds for d in greens), default=0)
check("no green shorter than min_green", shortest >= 12.0 - 1e-6,
      f"shortest was {shortest}s, min_green=12.0")


print("\n=== 5. max_green holds even when the strategy never stops ===")
ctrl5 = IntersectionController("s", two_phases(max_green=25.0), AlwaysExtend(), decision_interval=0.1)
for t in range(4000):
    ctrl5.tick(t * 0.1, {})
greens5 = [d for d in ctrl5.decisions if d.state is SignalState.YELLOW]
longest = max((d.duration_seconds for d in greens5), default=0)
check("no green longer than max_green", longest <= 25.0 + 0.2,
      f"longest was {longest}s, max_green=25.0")
check("an extend-forever strategy still yields the junction", len(greens5) > 5,
      f"{len(greens5)} handovers occurred")
check("termination reason recorded as max_green",
      all(d.reason == "max_green" for d in greens5), f"{set(d.reason for d in greens5)}")


print("\n=== 6. Both phases get served (no starvation) ===")
served = {}
for d in ctrl5.decisions:
    if d.state is SignalState.GREEN:
        served[d.phase_id] = served.get(d.phase_id, 0) + 1
check("every phase received green", len(served) == 2, f"{served}")
ratio = min(served.values()) / max(served.values()) if served else 0
check("phases served about equally", ratio > 0.8, f"ratio {ratio:.2f}")


print("\n=== 7. Decisions carry the context needed to audit them ===")
ctrl7 = IntersectionController("s", two_phases(), AdaptiveStrategy())
snaps = {"north": types.SimpleNamespace(queue_length=9, flow_rate=14.0),
         "south": types.SimpleNamespace(queue_length=4, flow_rate=6.0),
         "east": types.SimpleNamespace(queue_length=1, flow_rate=2.0),
         "west": types.SimpleNamespace(queue_length=0, flow_rate=1.0)}
for t in range(400):
    ctrl7.tick(t * 0.5, snaps)
d0 = ctrl7.decisions[0]
check("queue recorded on the decision", d0.queue_at_decision == 13, f"{d0.queue_at_decision}")
check("flow recorded on the decision", abs(d0.flow_at_decision - 20.0) < 0.01, f"{d0.flow_at_decision}")
check("reasons are countable, not free text", set(ctrl7.reason_counts()) <= {
    "queue_present", "queue_cleared", "gap_out", "max_green", "starvation_guard"},
    f"{ctrl7.reason_counts()}")


print("\n=== 8. Diurnal demand curve is normalised and shaped ===")
peak = max(diurnal_multiplier(h / 60.0) for h in range(24 * 60))
check("curve peaks at exactly 1.0", abs(peak - 1.0) < 1e-9, f"{peak}")
check("03:00 is quiet", diurnal_multiplier(3.0) < 0.25, f"{diurnal_multiplier(3.0):.3f}")
check("09:00 morning peak is busy", diurnal_multiplier(9.0) > 0.7, f"{diurnal_multiplier(9.0):.3f}")
check("18:30 is the daily maximum", diurnal_multiplier(18.5) > diurnal_multiplier(9.0))


print("\n=== 9. Poisson arrivals have the right mean and variance ===")
import random as _r
rng = _r.Random(1)
draws = [_poisson(4.0, rng) for _ in range(20000)]
mean = sum(draws) / len(draws)
var = sum((d - mean) ** 2 for d in draws) / len(draws)
check("mean matches the rate", abs(mean - 4.0) < 0.1, f"mean={mean:.3f}")
# For a Poisson process variance equals the mean; that is what makes arrivals
# bursty rather than evenly spaced, which is what makes queues form at all.
check("variance equals mean (genuinely Poisson)", abs(var - 4.0) < 0.2, f"var={var:.3f}")


print("\n=== 10. Saturation accounting is correct ===")
a = ApproachSim("x", lanes=2, peak_vehicles_per_hour=1200, saturation_flow_per_lane=1800)
check("capacity = lanes x saturation x green share",
      abs(a.capacity_per_hour(0.5) - 1800.0) < 1e-6, f"{a.capacity_per_hour(0.5)}")
check("v/c ratio computed correctly",
      abs(a.degree_of_saturation(0.5) - 1200 / 1800) < 1e-6,
      f"{a.degree_of_saturation(0.5):.3f}")
check("oversaturation detected", ApproachSim("y", lanes=1, peak_vehicles_per_hour=2000)
      .degree_of_saturation(0.5) > 1.0)


print("\n=== 11. Simulation is reproducible, and seed actually varies it ===")
def approaches():
    return [ApproachSim("north", lanes=2, peak_vehicles_per_hour=1250),
            ApproachSim("south", lanes=2, peak_vehicles_per_hour=1150),
            ApproachSim("east", lanes=1, peak_vehicles_per_hour=420),
            ApproachSim("west", lanes=1, peak_vehicles_per_hour=360)]

r1 = compare_strategies(approaches, two_phases(max_green=50), {"a": AdaptiveStrategy()},
                        hours=2.0, seed=7)["a"]
r2 = compare_strategies(approaches, two_phases(max_green=50), {"a": AdaptiveStrategy()},
                        hours=2.0, seed=7)["a"]
r3 = compare_strategies(approaches, two_phases(max_green=50), {"a": AdaptiveStrategy()},
                        hours=2.0, seed=99)["a"]
check("same seed gives identical results", r1.average_delay == r2.average_delay,
      f"{r1.average_delay} vs {r2.average_delay}")
check("different seed gives different traffic", r1.average_delay != r3.average_delay,
      f"{r1.average_delay} vs {r3.average_delay}")


print("\n=== 12. Adaptive control beats every fixed-time schedule ===")
res = compare_strategies(approaches, two_phases(max_green=50), {
    "fixed-25": FixedTimeStrategy(25), "fixed-35": FixedTimeStrategy(35),
    "fixed-50": FixedTimeStrategy(50), "adaptive": AdaptiveStrategy(),
}, hours=12.0, seed=42)
ad = res["adaptive"].average_delay
best_fixed = min(r.average_delay for k, r in res.items() if k.startswith("fixed"))
for k, r in res.items():
    print(f"        {k:<9} {r.average_delay:6.1f}s  served {r.total_served:6.0f}  peak q {r.peak_queue:5.1f}")
check("adaptive has the lowest delay of all schedules",
      all(ad < r.average_delay for k, r in res.items() if k.startswith("fixed")),
      f"adaptive {ad}s vs best fixed {best_fixed}s")
improvement = (best_fixed - ad) / best_fixed * 100
check("improvement is material (>15%)", improvement > 15, f"{improvement:.1f}%")
check("adaptive serves at least as much traffic",
      res["adaptive"].total_served >= max(r.total_served for k, r in res.items()
                                          if k.startswith("fixed")) * 0.99,
      "throughput not sacrificed for delay")
check("adaptive holds shorter queues",
      res["adaptive"].peak_queue < min(r.peak_queue for k, r in res.items()
                                       if k.startswith("fixed")),
      f"peak {res['adaptive'].peak_queue} vs {min(r.peak_queue for k,r in res.items() if k.startswith('fixed'))}")


print("\n=== 13. Delay accounting balances ===")
sim = TrafficSimulator(approaches(), IntersectionController(
    "s", two_phases(max_green=50), AdaptiveStrategy()), seed=5)
out = sim.run(hours=3.0, collect_snapshots=True)
check("vehicles served does not exceed vehicles arrived",
      out.total_served <= out.total_arrived + 1e-6,
      f"served {out.total_served} arrived {out.total_arrived}")
check("nearly all arrivals are cleared when undersaturated",
      out.total_served / out.total_arrived > 0.95,
      f"{out.total_served / out.total_arrived:.3f}")
check("snapshots were produced for every approach",
      len({s.camera_id for s in out.snapshots}) == 4,
      f"{len(out.snapshots)} snapshots")
check("snapshot fields match TrafficSnapshot's interface",
      all(hasattr(out.snapshots[0], f) for f in
          ("camera_id", "timestamp", "queue_length", "flow_rate", "crossings_delta")))

print(f"\n{'='*64}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
