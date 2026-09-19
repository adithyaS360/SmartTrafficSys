"""
Standard traffic-engineering calculations: PCU conversion, saturation flow,
and Webster signal timing.

WHY THIS MODULE EXISTS - it fixes a weakness in the comparison:

Until now the adaptive controller was measured against a fixed 30-second plan.
That is close to a strawman. A real engineer does not pick 30 seconds; they
measure the flows and compute a plan, and the standard way to do that has been
Webster's method since 1958. Beating an arbitrary number proves little. Beating
the plan a competent engineer would actually have set is a claim worth making.

So the fixed-time baseline is now computed, not invented.

PCU - why vehicle counts alone are not enough:

An auto-rickshaw does not occupy a junction like a car, and a bus occupies it
like three. Traffic engineering handles this by converting every vehicle into
Passenger Car Units before doing any capacity arithmetic. On an Indian junction
this is not a refinement - it is the difference between a usable number and a
meaningless one, because the mix is dominated by two- and three-wheelers that
Western defaults do not account for.

This is also where the detector's classification accuracy stops being a
footnote. If auto-rickshaws are misread as cars, the PCU total is wrong, the
saturation ratio is wrong, and every timing derived from it is wrong. The
COCO-has-no-auto-rickshaw problem propagates all the way here.

Factors below are IRC 106-1990, the Indian standard.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# PCU
# ---------------------------------------------------------------------------

# IRC 106-1990 passenger-car-unit equivalents for urban roads.
IRC_PCU_FACTORS: Dict[str, float] = {
    "two_wheeler": 0.5,     # motorcycle, scooter
    "three_wheeler": 1.2,   # auto-rickshaw
    "car": 1.0,             # car, jeep, van - the unit by definition
    "bus": 2.2,
    "heavy": 4.0,           # truck, multi-axle
    "bicycle": 0.4,
}

# How the detector's COCO classes map onto those categories.
#
# NOTE THE GAP, and it is the important one: COCO has no auto-rickshaw class.
# YOLO will read an auto as 'car' (1.0 PCU) or 'motorcycle' (0.5) when it should
# be 1.2. On a junction where autos are a fifth of the traffic that is a
# systematic error in the PCU total, and therefore in every timing computed from
# it. Correcting it needs a fine-tuned model, not a mapping table - this dict
# only records where the error enters.
COCO_TO_CATEGORY: Dict[str, str] = {
    "motorcycle": "two_wheeler",
    "bicycle": "bicycle",
    "car": "car",
    "bus": "bus",
    "truck": "heavy",
}

# Composition of a typical Indian urban arterial approach, as a share of
# vehicles (not PCU). Taken from the mix reported in published Indian junction
# surveys; the SHAPE is what matters here, and it is what makes this traffic
# behave unlike the Western defaults most tutorials assume.
INDIAN_URBAN_MIX: Dict[str, float] = {
    "two_wheeler": 0.42,
    "three_wheeler": 0.18,
    "car": 0.28,
    "bus": 0.07,
    "heavy": 0.05,
}


def to_pcu(counts: Dict[str, float],
           factors: Optional[Dict[str, float]] = None) -> float:
    """Convert a count of vehicles by category into PCU."""
    factors = factors or IRC_PCU_FACTORS
    return sum(n * factors.get(cat, 1.0) for cat, n in counts.items())


def mix_pcu_per_vehicle(mix: Optional[Dict[str, float]] = None,
                        factors: Optional[Dict[str, float]] = None) -> float:
    """
    Average PCU contributed by one vehicle drawn from a given mix.

    Useful because the simulator counts vehicles while capacity arithmetic
    needs PCU; this is the conversion factor between the two. For the Indian
    mix above it comes out near 1.0 purely by coincidence - a lot of small
    two-wheelers balancing a few buses - which is worth noticing rather than
    relying on.
    """
    mix = mix or INDIAN_URBAN_MIX
    factors = factors or IRC_PCU_FACTORS
    total_share = sum(mix.values()) or 1.0
    return sum(share * factors.get(cat, 1.0) for cat, share in mix.items()) / total_share


# ---------------------------------------------------------------------------
# Saturation flow
# ---------------------------------------------------------------------------

def saturation_flow(lane_width_m: float = 3.5,
                    lanes: int = 1,
                    base_per_metre: float = 525.0) -> float:
    """
    Saturation flow in PCU/hour for an approach.

    IRC 106 expresses saturation flow for urban roads as a function of
    carriageway width rather than a flat per-lane figure, which suits Indian
    conditions where lane discipline is loose and vehicles occupy the road by
    width rather than by lane. Roughly 525 PCU/hour per metre of width is the
    conventional value for widths in the normal urban range.

    A 3.5m lane therefore gives about 1840 PCU/hour, close to the ~1800 figure
    used for planning - which is a useful check that this is not drifting away
    from standard practice.
    """
    return base_per_metre * lane_width_m * lanes


# ---------------------------------------------------------------------------
# Webster timing
# ---------------------------------------------------------------------------

@dataclass
class PhaseDemand:
    """Demand on one signal phase, in PCU/hour."""

    phase_id: str
    flow_pcu: float             # critical approach flow for this phase
    saturation_pcu: float       # saturation flow of that critical approach

    @property
    def flow_ratio(self) -> float:
        """y = q/s. The share of the cycle this phase fundamentally needs."""
        return self.flow_pcu / self.saturation_pcu if self.saturation_pcu > 0 else 1.0


@dataclass
class WebsterPlan:
    """A computed fixed-time plan."""

    cycle_seconds: float
    lost_time: float
    critical_ratio: float                       # Y
    green: Dict[str, float] = field(default_factory=dict)
    oversaturated: bool = False
    note: str = ""

    def summary(self) -> str:
        if self.oversaturated:
            return f"OVERSATURATED (Y={self.critical_ratio:.2f}) - {self.note}"
        greens = ", ".join(f"{k} {v:.0f}s" for k, v in self.green.items())
        return f"cycle {self.cycle_seconds:.0f}s (Y={self.critical_ratio:.2f}): {greens}"


def webster_plan(phases: Sequence[PhaseDemand],
                 lost_time_per_phase: float = 2.0,
                 all_red_total: float = 0.0,
                 min_green: float = 10.0,
                 max_cycle: float = 120.0) -> WebsterPlan:
    """
    Webster's optimum cycle length and green split.

        C0 = (1.5L + 5) / (1 - Y)
        Gi = (yi / Y) * (C0 - L)

    where Y is the sum of critical flow ratios across phases and L is total
    lost time. Reference: Webster (1958), Road Research Laboratory.

    THE FORMULA HAS A SINGULARITY AND IT IS NOT A CORNER CASE. As Y approaches
    1 the junction is approaching capacity and C0 goes to infinity; past 1 it
    turns negative, which is meaningless. Rather than return a nonsense number
    this reports oversaturation explicitly, because an oversaturated junction
    cannot be fixed by retiming - it needs more capacity or less demand, and a
    plan that implies otherwise is worse than no plan.
    """
    if not phases:
        raise ValueError("Webster timing needs at least one phase.")

    lost_time = lost_time_per_phase * len(phases) + all_red_total
    critical_ratio = sum(p.flow_ratio for p in phases)

    if critical_ratio >= 0.95:
        return WebsterPlan(
            cycle_seconds=max_cycle, lost_time=lost_time,
            critical_ratio=critical_ratio, oversaturated=True,
            green={p.phase_id: max(min_green, (max_cycle - lost_time) * p.flow_ratio
                                   / max(critical_ratio, 1e-6)) for p in phases},
            note=("demand is at or above capacity; no fixed plan clears it. "
                  "Retiming cannot help - this junction needs more lanes or "
                  "less demand."),
        )

    cycle = (1.5 * lost_time + 5.0) / (1.0 - critical_ratio)
    cycle = min(max(cycle, lost_time + min_green * len(phases)), max_cycle)

    effective_green = cycle - lost_time
    green = {
        p.phase_id: max(min_green, effective_green * p.flow_ratio / critical_ratio)
        for p in phases
    }

    # Clamping individual greens to min_green can push the total past the cycle;
    # rescale so the plan is internally consistent rather than merely plausible.
    total = sum(green.values())
    if total > effective_green and total > 0:
        scale = effective_green / total
        green = {k: max(min_green, v * scale) for k, v in green.items()}

    return WebsterPlan(cycle_seconds=cycle, lost_time=lost_time,
                       critical_ratio=critical_ratio, green=green,
                       note="within capacity")


def webster_delay(flow_pcu: float, saturation_pcu: float,
                  cycle: float, green: float) -> float:
    """
    Webster's average delay per vehicle, in seconds, for one approach.

        d = C(1-L)^2 / (2(1-Lx))  +  x^2 / (2q(1-x))  -  0.65 (C/q^2)^(1/3) x^(2+5L)

    with L = g/C (the green ratio) and x = q/(L*s) (degree of saturation).

    The three terms are uniform delay, random-arrival delay, and an empirical
    correction. Returns a large sentinel rather than diverging when x >= 1: at
    saturation the queue never clears and average delay is unbounded, so any
    finite number would be a lie.
    """
    if saturation_pcu <= 0 or cycle <= 0 or green <= 0 or flow_pcu <= 0:
        return 0.0

    green_ratio = green / cycle
    capacity = green_ratio * saturation_pcu
    x = flow_pcu / capacity if capacity > 0 else 99.0
    if x >= 1.0:
        return 999.0

    q_per_sec = flow_pcu / 3600.0
    uniform = cycle * (1 - green_ratio) ** 2 / (2 * (1 - green_ratio * x))
    random_term = x ** 2 / (2 * q_per_sec * (1 - x))
    correction = 0.65 * (cycle / q_per_sec ** 2) ** (1 / 3) * x ** (2 + 5 * green_ratio)
    return max(0.0, uniform + random_term - correction)


def plan_from_approach_flows(phase_approaches: Dict[str, List[str]],
                             approach_flow_vehicles_per_hour: Dict[str, float],
                             approach_lanes: Dict[str, int],
                             mix: Optional[Dict[str, float]] = None,
                             **kwargs) -> WebsterPlan:
    """
    Convenience: go from measured vehicle counts straight to a timing plan.

    Each phase's demand is its CRITICAL approach - the worst one, not the sum.
    Approaches in the same phase run concurrently, so the phase needs whatever
    its busiest member needs; adding them would double-count and inflate the
    cycle.
    """
    pcu_per_vehicle = mix_pcu_per_vehicle(mix)
    phases: List[PhaseDemand] = []

    for phase_id, approaches in phase_approaches.items():
        worst: Optional[PhaseDemand] = None
        for cam in approaches:
            flow_pcu = approach_flow_vehicles_per_hour.get(cam, 0.0) * pcu_per_vehicle
            sat = saturation_flow(lanes=approach_lanes.get(cam, 1))
            candidate = PhaseDemand(phase_id, flow_pcu, sat)
            if worst is None or candidate.flow_ratio > worst.flow_ratio:
                worst = candidate
        if worst is not None:
            phases.append(worst)

    return webster_plan(phases, **kwargs)
