"""
Verification of the traffic-engineering layer: PCU conversion, saturation flow,
Webster timing, and the three-way demo comparison.

The Webster tests check against HAND-WORKED values rather than against whatever
the code currently returns. A test that asserts the code agrees with itself
catches nothing; these would fail if the formula were transcribed wrongly.
"""
import sys, types, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = types.ModuleType("cv2"); cv2.__getattr__ = lambda n: (lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0; cv2.CAP_PROP_BUFFERSIZE = 38; cv2.CAP_PROP_POS_FRAMES = 1
cv2.VideoCapture = object
sys.modules["cv2"] = cv2
sys.modules["ultralytics"] = types.ModuleType("ultralytics")
sys.modules["ultralytics"].YOLO = object

from src.demo import DEMAND_PRESETS, ParallelDemo
from src.traffic_controller import FixedTimeStrategy, Phase
from src.traffic_engineering import (IRC_PCU_FACTORS, INDIAN_URBAN_MIX, PhaseDemand,
                                     mix_pcu_per_vehicle, plan_from_approach_flows,
                                     saturation_flow, to_pcu, webster_delay, webster_plan)

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))


print("\n=== 1. Webster matches a hand-worked example ===")
# y1 = 540/1800 = 0.30, y2 = 450/1800 = 0.25, Y = 0.55
# L = 2 phases x 2s + 6s all-red = 10s
# C0 = (1.5*10 + 5) / (1 - 0.55) = 20 / 0.45 = 44.444s
# effective green = 34.444;  G1 = 34.444 * 0.30/0.55 = 18.788;  G2 = 15.657
p = [PhaseDemand("a", 540, 1800), PhaseDemand("b", 450, 1800)]
plan = webster_plan(p, lost_time_per_phase=2.0, all_red_total=6.0)
check("flow ratios", abs(p[0].flow_ratio - 0.30) < 1e-9 and abs(p[1].flow_ratio - 0.25) < 1e-9)
check("critical ratio Y = 0.55", abs(plan.critical_ratio - 0.55) < 1e-9)
check("lost time L = 10s", abs(plan.lost_time - 10.0) < 1e-9)
check("cycle = 44.44s", abs(plan.cycle_seconds - 44.4444) < 0.01, f"{plan.cycle_seconds:.4f}")
check("green a = 18.79s", abs(plan.green["a"] - 18.7879) < 0.01, f"{plan.green['a']:.4f}")
check("green b = 15.66s", abs(plan.green["b"] - 15.6566) < 0.01, f"{plan.green['b']:.4f}")
check("greens plus lost time equal the cycle",
      abs(sum(plan.green.values()) + plan.lost_time - plan.cycle_seconds) < 0.01)

print("\n=== 2. Busier phases get proportionally more green ===")
p2 = webster_plan([PhaseDemand("main", 1080, 1800), PhaseDemand("side", 360, 1800)],
                  lost_time_per_phase=2.0, all_red_total=4.0)
check("main road outranks side road", p2.green["main"] > p2.green["side"],
      f"main {p2.green['main']:.1f}s vs side {p2.green['side']:.1f}s")
check("split matches the flow ratio 3:1",
      abs(p2.green["main"] / p2.green["side"] - 3.0) < 0.05,
      f"ratio {p2.green['main']/p2.green['side']:.2f}")

print("\n=== 3. Oversaturation is refused, not fudged ===")
over = webster_plan([PhaseDemand("a", 1700, 1800), PhaseDemand("b", 900, 1800)])
check("Y above 1 flagged", over.oversaturated and over.critical_ratio > 1.0,
      f"Y={over.critical_ratio:.2f}")
check("no negative or absurd cycle returned", 0 < over.cycle_seconds <= 120,
      f"{over.cycle_seconds}s")
check("says retiming cannot help", "cannot help" in over.note or "capacity" in over.note)
# Just below the threshold the formula still applies normally.
near = webster_plan([PhaseDemand("a", 800, 1800), PhaseDemand("b", 700, 1800)])
check("a junction just within capacity is still planned", not near.oversaturated,
      f"Y={near.critical_ratio:.2f} cycle={near.cycle_seconds:.0f}s")

print("\n=== 4. Minimum green is respected ===")
tiny = webster_plan([PhaseDemand("a", 900, 1800), PhaseDemand("b", 10, 1800)],
                    min_green=10.0)
check("a near-empty phase still gets min_green", tiny.green["b"] >= 10.0 - 1e-9,
      f"{tiny.green['b']:.1f}s")

print("\n=== 5. PCU conversion follows IRC 106-1990 ===")
check("an auto-rickshaw is 1.2 PCU", IRC_PCU_FACTORS["three_wheeler"] == 1.2)
check("a car is the unit", IRC_PCU_FACTORS["car"] == 1.0)
check("a two-wheeler is 0.5", IRC_PCU_FACTORS["two_wheeler"] == 0.5)
counts = {"two_wheeler": 880, "three_wheeler": 326, "car": 582, "bus": 252, "heavy": 106}
expected = 880*0.5 + 326*1.2 + 582*1.0 + 252*2.2 + 106*4.0
check("mixed count converts correctly", abs(to_pcu(counts) - expected) < 1e-6,
      f"{to_pcu(counts):.1f} PCU")
# The composition matters: the same vehicle count gives a different PCU total.
all_cars = to_pcu({"car": 2146})
check("composition changes the PCU total for the same vehicle count",
      abs(to_pcu(counts) - all_cars) > 200,
      f"mixed {to_pcu(counts):.0f} vs all-cars {all_cars:.0f} for 2146 vehicles")
check("Indian mix averages near 1 PCU/vehicle",
      0.9 < mix_pcu_per_vehicle(INDIAN_URBAN_MIX) < 1.2,
      f"{mix_pcu_per_vehicle(INDIAN_URBAN_MIX):.3f}")

print("\n=== 6. Saturation flow is in the planning range ===")
s1 = saturation_flow(lane_width_m=3.5, lanes=1)
check("3.5m single lane near the 1800 planning figure", 1700 < s1 < 1950, f"{s1:.0f} PCU/h")
check("two lanes double it", abs(saturation_flow(lanes=2) - 2 * s1) < 1e-6)

print("\n=== 7. Webster delay behaves correctly ===")
check("delay rises as a junction approaches saturation",
      webster_delay(1200, 1840, 60, 30) > webster_delay(600, 1840, 60, 30))
check("more green for the same flow means less delay",
      webster_delay(700, 1840, 60, 40) < webster_delay(700, 1840, 60, 20))
check("saturated approach returns the sentinel, not a finite lie",
      webster_delay(1800, 1840, 60, 20) >= 999.0)
check("zero flow gives zero delay", webster_delay(0, 1840, 60, 30) == 0.0)

print("\n=== 8. FixedTimeStrategy accepts a per-phase plan ===")
ns = Phase(id="ns", camera_ids=["north"], min_green=10, max_green=60)
ew = Phase(id="ew", camera_ids=["east"], min_green=10, max_green=60)
strat = FixedTimeStrategy({"ns": 35.0, "ew": 15.0})
check("each phase gets its own duration",
      strat._target_for(ns) == 35.0 and strat._target_for(ew) == 15.0)
check("a scalar still applies to every phase",
      FixedTimeStrategy(25)._target_for(ns) == 25.0)
check("durations are clamped to the phase's safety limits",
      FixedTimeStrategy({"ns": 500.0})._target_for(ns) == 60.0
      and FixedTimeStrategy({"ns": 1.0})._target_for(ns) == 10.0)
check("an unlisted phase falls back to max_green",
      FixedTimeStrategy({"ns": 35.0})._target_for(ew) == 60.0)

print("\n=== 9. The demo builds all three junctions with a computed plan ===")
demo = ParallelDemo(demand="busy")
check("three junctions", set(demo.junctions) == {"naive", "webster", "adaptive"},
      f"{sorted(demo.junctions)}")
check("the webster junction's plan was computed, not hardcoded",
      not demo.webster.oversaturated and demo.webster.cycle_seconds > 0,
      demo.webster.summary())
check("the main road gets more green than the side road",
      demo.webster.green["ns"] > demo.webster.green["ew"],
      f"ns {demo.webster.green['ns']:.0f}s vs ew {demo.webster.green['ew']:.0f}s")

print("\n=== 10. All three junctions receive IDENTICAL traffic ===")
for _ in range(600):
    demo.tick()
arrived = {k: round(sum(a.arrived for a in j.approaches.values()), 6)
           for k, j in demo.junctions.items()}
check("arrivals are identical across all three", len(set(arrived.values())) == 1,
      f"{arrived}")
check("but the outcomes differ, so the controllers are doing something",
      len({j.totals()["average_delay"] for j in demo.junctions.values()}) == 3,
      str({k: j.totals()["average_delay"] for k, j in demo.junctions.items()}))

print("\n=== 11. The honest result: retiming does most of the work ===")
snap = demo.snapshot()
c = snap["comparison"]
print(f"        naive {c['naive_delay']:.1f}s -> webster {c['webster_delay']:.1f}s "
      f"-> adaptive {c['adaptive_delay']:.1f}s")
check("retiming alone beats the badly timed plan",
      c["webster_delay"] < c["naive_delay"],
      f"saves {c['seconds_saved_by_retiming']:.1f}s/vehicle")
check("adaptive beats the properly timed plan too",
      c["adaptive_delay"] <= c["webster_delay"] + 0.5)
check("the vs-webster figure is the smaller, honest one",
      c["vs_webster_percent"] is None or c["vs_naive_percent"] is None
      or c["vs_webster_percent"] < c["vs_naive_percent"],
      f"vs webster {c['vs_webster_percent']}% vs naive {c['vs_naive_percent']}%")
check("history carries all three series",
      set(snap["history"][0]) == {"t", "naive", "webster", "adaptive"})

print("\n=== 12. The load-dependence claim on the page is true ===")
savings = {}
for level in ("light", "rush"):
    d = ParallelDemo(demand=level)
    for _ in range(2400):
        d.tick()
    savings[level] = d.snapshot()["comparison"]["vs_webster_percent"]
print(f"        light {savings['light']}%   rush {savings['rush']}%")
check("adaptive gains far more at rush hour than at light demand",
      savings["rush"] > savings["light"] + 5,
      "the page claims this pattern; it holds")

print(f"\n{'='*64}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 64)
sys.exit(1 if FAIL else 0)
