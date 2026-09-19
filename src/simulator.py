"""
Closed-loop traffic simulator.

WHY A SIMULATOR AT ALL, when the point of the project is real footage:

Two reasons, and neither is "because real data is hard".

1. The LSTM needs history that does not exist yet. A model cannot be trained on
   an empty database, and you cannot film a junction for a fortnight before
   writing any code. Synthetic history lets the whole training pipeline be built
   and tested now, then re-run on real data when you have it.

2. You cannot measure a controller against real footage. Footage is a
   recording: the cars in it did what they did, and they will do the same thing
   however your signal behaves. To claim "adaptive control reduced delay" you
   need a world that RESPONDS - where a longer green actually clears more
   vehicles and a shorter one actually leaves them waiting. That is what closed
   loop means, and it is the only way to get a number rather than an assertion.

So: real footage proves the detector sees vehicles correctly. The simulator
proves the controller uses that information well. Both are needed and neither
substitutes for the other, which is worth stating plainly in the report.

THE TRAFFIC MODEL, in standard traffic-engineering terms:

  ARRIVALS are a non-homogeneous Poisson process. Poisson is the standard model
  for vehicles arriving at an isolated junction - arrivals are independent and
  the probability of one in a small interval is proportional to that interval.
  Non-homogeneous means the rate varies with time of day, which is what produces
  rush hours.

  DEPARTURES during green occur at the SATURATION FLOW RATE: the maximum rate a
  queue discharges once moving. The standard figure is about 1900 vehicles per
  hour per lane, roughly one vehicle every 1.9 seconds per lane. Indian urban
  junctions with mixed traffic are usually measured lower; 1800 is a common
  planning figure and the value is configurable here.

  STARTUP LOST TIME is the two-ish seconds at the beginning of green before
  discharge reaches saturation, while drivers react and accelerate. It matters:
  it is why very short greens are inefficient, and why a controller that
  switches phases constantly performs worse than one that does not, even though
  each individual switch looks locally reasonable.

  DELAY is accumulated as vehicle-seconds: every vehicle in the queue accrues
  one vehicle-second per second of waiting. Total delay divided by vehicles
  served gives AVERAGE DELAY PER VEHICLE, which is the standard measure of
  junction performance and the number your report should lead with.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from src.traffic_controller import IntersectionController, Phase
from src.utils.logger import get_logger

log = get_logger(__name__)

SECONDS_PER_HOUR = 3600.0


def _raw_diurnal(hour: float) -> float:
    morning = 1.6 * math.exp(-((hour - 9.0) ** 2) / (2 * 1.1 ** 2))
    evening = 2.0 * math.exp(-((hour - 18.5) ** 2) / (2 * 1.5 ** 2))
    overnight = 0.10 if hour < 5.5 or hour > 23 else 0.0
    return max(0.03, 0.45 + morning + evening - overnight)


# Normalising constant so the curve peaks at exactly 1.0.
_DIURNAL_PEAK = max(_raw_diurnal(h / 60.0) for h in range(24 * 60))


def diurnal_multiplier(hour: float) -> float:
    """
    Demand multiplier over a 24-hour day, in [0, 1], peaking at 1.0.

    Two Gaussians on a low base: a morning peak around 09:00 and a larger
    evening peak around 18:30, with an overnight trough. The evening peak is
    broader because the evening commute is spread over a longer, less
    disciplined window than the morning one.

    NORMALISED TO PEAK AT 1.0 so that an approach's peak_vehicles_per_hour means
    what it says. Before normalising, the curve topped out at about 2.45, so
    configuring "1100 vehicles/hour" silently produced 2700 at the evening peak
    - well past the ~1800/hour a two-lane approach can discharge with half the
    cycle. Every approach was oversaturated, queues grew without bound, and the
    simulator reported 13-minute average delays. The comparison still ranked the
    strategies correctly, but the absolute figures were meaningless.

    The shape matters more than the exact parameters: a model trained on traffic
    that is flat all day learns nothing, because there is no pattern to learn.
    """
    return _raw_diurnal(hour) / _DIURNAL_PEAK


@dataclass
class ApproachSim:
    """One approach: a queue that fills with arrivals and drains on green."""

    camera_id: str
    lanes: int = 2
    peak_vehicles_per_hour: float = 900.0
    saturation_flow_per_lane: float = 1800.0   # veh/hour/lane at full discharge
    startup_lost_time: float = 2.0             # seconds before saturation is reached

    queue: float = 0.0
    arrived: float = 0.0
    served: float = 0.0
    delay_vehicle_seconds: float = 0.0
    peak_queue: float = 0.0
    _green_elapsed: float = 0.0
    _crossings_since_snapshot: float = 0.0
    _recent_arrivals: List[float] = field(default_factory=list)

    def arrival_rate(self, hour: float) -> float:
        """Vehicles per second at this time of day."""
        return self.peak_vehicles_per_hour * diurnal_multiplier(hour) / SECONDS_PER_HOUR

    def step(self, dt: float, hour: float, is_green: bool, rng: random.Random,
             arrivals: Optional[float] = None) -> None:
        """
        Advance this approach by dt seconds.

        Args:
            arrivals: vehicles arriving this step. Normally left as None, so the
                approach draws its own Poisson sample. Pass a value to feed the
                SAME arrivals to two approaches running in parallel - which is
                what makes a live side-by-side comparison of two control
                strategies honest. Without it each junction would draw its own
                random traffic, and any difference in delay could just as
                easily be luck as control quality.
        """
        # --- arrivals: Poisson with the current rate ---
        if arrivals is None:
            expected = self.arrival_rate(hour) * dt
            arrivals = float(_poisson(expected, rng))
        self.queue += arrivals
        self.arrived += arrivals
        self._recent_arrivals.append(arrivals)
        if len(self._recent_arrivals) > int(60 / dt) + 1:
            self._recent_arrivals.pop(0)

        # --- departures: only on green, at saturation flow after startup ---
        if is_green:
            self._green_elapsed += dt
            # Ramp discharge in over the startup lost time rather than switching
            # it on instantly - this is what penalises very short greens.
            ramp = min(1.0, self._green_elapsed / max(self.startup_lost_time, 1e-6))
            capacity = self.saturation_flow_per_lane * self.lanes / SECONDS_PER_HOUR * dt * ramp
            departures = min(self.queue, capacity)
            self.queue -= departures
            self.served += departures
            self._crossings_since_snapshot += departures
        else:
            self._green_elapsed = 0.0

        # --- delay: every queued vehicle accrues dt vehicle-seconds ---
        self.delay_vehicle_seconds += self.queue * dt
        self.peak_queue = max(self.peak_queue, self.queue)

    # ---- observable quantities, i.e. what a camera would report ----

    def observed_flow_per_minute(self, dt: float) -> float:
        if not self._recent_arrivals:
            return 0.0
        return sum(self._recent_arrivals) / (len(self._recent_arrivals) * dt) * 60.0

    def take_crossings(self) -> int:
        whole = int(self._crossings_since_snapshot)
        self._crossings_since_snapshot -= whole
        return whole

    @property
    def average_delay(self) -> float:
        """Average delay per vehicle served, in seconds. The headline metric."""
        return self.delay_vehicle_seconds / self.served if self.served > 0 else 0.0

    def capacity_per_hour(self, green_fraction: float) -> float:
        """Vehicles per hour this approach can discharge given its share of green."""
        return self.saturation_flow_per_lane * self.lanes * green_fraction

    def degree_of_saturation(self, green_fraction: float) -> float:
        """
        Peak demand divided by capacity - 'v/c ratio' in traffic engineering.

        Below about 0.85 a junction behaves normally and delay is modest.
        Approaching 1.0, delay rises steeply. Above 1.0 the junction is
        OVERSATURATED: arrivals exceed what green time can discharge, the
        residual queue grows every cycle, and average delay depends mostly on
        how long you ran the simulation. No control strategy fixes that - it is
        a capacity problem, needing more lanes or less demand.

        Worth checking before quoting any delay figure. An oversaturated
        comparison still ranks strategies correctly but the absolute numbers are
        an artefact of run length, not a property of the junction.
        """
        cap = self.capacity_per_hour(green_fraction)
        return self.peak_vehicles_per_hour / cap if cap > 0 else float("inf")


def _poisson(mean: float, rng: random.Random) -> int:
    """
    Draw from a Poisson distribution.

    Knuth's method for small means, a normal approximation above 30 where the
    multiplicative loop gets slow and the approximation is already good.
    """
    if mean <= 0:
        return 0
    if mean > 30:
        return max(0, int(rng.gauss(mean, math.sqrt(mean)) + 0.5))
    limit = math.exp(-mean)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


@dataclass
class SimSnapshot:
    """
    What the simulator hands the controller.

    Deliberately exposes the same attribute names as TrafficSnapshot from
    data_collector - queue_length, flow_rate, vehicle_count - so the controller
    consumes simulated and real data through exactly the same interface and
    cannot tell them apart. If the controller needed to know which it was
    getting, the comparison would be worthless.
    """

    camera_id: str
    timestamp: datetime
    queue_length: int
    flow_rate: float
    vehicle_count: float
    crossings_delta: int = 0
    pedestrian_count: float = 0.0
    avg_dwell_seconds: float = 0.0
    avg_speed_kmh: Optional[float] = None
    total_crossings: int = 0
    class_breakdown: Optional[dict] = None
    processing_ms: float = 0.0


@dataclass
class SimResult:
    """Outcome of one simulation run."""

    strategy: str
    duration_hours: float
    total_served: float
    total_arrived: float
    average_delay: float          # seconds per vehicle - the headline
    peak_queue: float
    per_approach: Dict[str, dict]
    reason_counts: Dict[str, int]
    green_durations: Dict[str, List[float]]
    snapshots: List[SimSnapshot] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.strategy:<10} avg delay {self.average_delay:6.1f}s  "
                f"served {self.total_served:7.0f}  peak queue {self.peak_queue:5.1f}")


class TrafficSimulator:
    """
    Runs approaches and a controller together in closed loop.

    The loop is: observe queues -> controller decides -> signal state changes ->
    queues discharge differently -> observe again. That feedback is the whole
    point; without it the controller's decisions would not affect anything and
    any measured "improvement" would be an artefact.
    """

    def __init__(self,
                 approaches: Sequence[ApproachSim],
                 controller: IntersectionController,
                 dt: float = 1.0,
                 start_hour: float = 6.0,
                 seed: int = 42):
        """
        Args:
            dt: simulation timestep in seconds. 1s is fine - signal decisions
                happen on a scale of seconds and finer steps only cost time.
            start_hour: hour of day to begin at
            seed: RNG seed. FIXING THIS IS WHAT MAKES THE COMPARISON VALID -
                fixed-time and adaptive must face the identical arrival
                sequence, or you are comparing two different days of traffic
                and the difference means nothing.
        """
        self.approaches = {a.camera_id: a for a in approaches}
        self.controller = controller
        self.dt = dt
        self.start_hour = start_hour
        self.seed = seed

    def _warn_if_oversaturated(self) -> None:
        """
        Check demand against capacity before running, and say so if the junction
        cannot cope. Delay figures from an oversaturated run are a function of
        how long you ran it, not of the controller - quoting them as a result is
        the easiest way to put a wrong number in a report.
        """
        n_phases = max(len(self.controller.phases), 1)
        # Rough share of the cycle each approach gets, net of yellow and all-red.
        phase = self.controller.phases[0]
        lost = (phase.yellow + phase.all_red) * n_phases
        cycle = phase.max_green * n_phases + lost
        green_fraction = (cycle / n_phases - lost / n_phases) / cycle

        for cam, approach in self.approaches.items():
            dos = approach.degree_of_saturation(green_fraction)
            if dos >= 1.0:
                log.warning(
                    "Approach '{}' is OVERSATURATED (v/c = {:.2f}): peak demand "
                    "{:.0f} veh/h exceeds capacity {:.0f} veh/h. Queues will grow "
                    "without bound and delay figures will reflect run length, not "
                    "control quality. Reduce peak_vehicles_per_hour, add a lane, "
                    "or treat this run as a capacity study rather than a comparison.",
                    cam, dos, approach.peak_vehicles_per_hour,
                    approach.capacity_per_hour(green_fraction))
            elif dos >= 0.85:
                log.info("Approach '{}' is near capacity (v/c = {:.2f}) - delay "
                         "will be sensitive to control quality, which is the "
                         "interesting regime.", cam, dos)

    def run(self,
            hours: float = 4.0,
            snapshot_interval: float = 5.0,
            collect_snapshots: bool = True,
            start_time: Optional[datetime] = None) -> SimResult:
        """Simulate `hours` of traffic and return the result."""
        rng = random.Random(self.seed)
        for a in self.approaches.values():
            a.queue = a.arrived = a.served = a.delay_vehicle_seconds = a.peak_queue = 0.0
            a._green_elapsed = a._crossings_since_snapshot = 0.0
            a._recent_arrivals.clear()

        self._warn_if_oversaturated()

        steps = int(hours * SECONDS_PER_HOUR / self.dt)
        wall_start = start_time or datetime.now(timezone.utc) - timedelta(hours=hours)
        snapshots: List[SimSnapshot] = []
        observed: Dict[str, SimSnapshot] = {}
        next_snapshot = 0.0

        for step in range(steps):
            t = step * self.dt
            hour = (self.start_hour + t / SECONDS_PER_HOUR) % 24.0

            # Observe. The controller only ever sees these, never the true state.
            if t >= next_snapshot:
                stamp = wall_start + timedelta(seconds=t)
                observed = {
                    cam: SimSnapshot(
                        camera_id=cam,
                        timestamp=stamp,
                        queue_length=int(round(a.queue)),
                        flow_rate=round(a.observed_flow_per_minute(self.dt), 2),
                        vehicle_count=round(a.queue, 2),
                        crossings_delta=a.take_crossings(),
                        avg_dwell_seconds=round(a.average_delay, 2),
                    )
                    for cam, a in self.approaches.items()
                }
                if collect_snapshots:
                    snapshots.extend(observed.values())
                next_snapshot = t + snapshot_interval

            self.controller.tick(t, observed)

            for cam, approach in self.approaches.items():
                approach.step(self.dt, hour, self.controller.is_green_for(cam), rng)

        served = sum(a.served for a in self.approaches.values())
        delay = sum(a.delay_vehicle_seconds for a in self.approaches.values())

        return SimResult(
            strategy=self.controller.strategy.name,
            duration_hours=hours,
            total_served=round(served, 1),
            total_arrived=round(sum(a.arrived for a in self.approaches.values()), 1),
            average_delay=round(delay / served, 2) if served else 0.0,
            peak_queue=round(max(a.peak_queue for a in self.approaches.values()), 1),
            per_approach={
                cam: {
                    "served": round(a.served, 1),
                    "average_delay": round(a.average_delay, 2),
                    "peak_queue": round(a.peak_queue, 1),
                }
                for cam, a in self.approaches.items()
            },
            reason_counts=self.controller.reason_counts(),
            green_durations=self.controller.green_durations(),
            snapshots=snapshots,
        )


def compare_strategies(approaches_factory,
                       phases: Sequence[Phase],
                       strategies: Dict[str, object],
                       hours: float = 4.0,
                       seed: int = 42,
                       start_hour: float = 6.0) -> Dict[str, SimResult]:
    """
    Run several strategies against the IDENTICAL traffic and compare.

    approaches_factory must return a FRESH list of ApproachSim each call, so no
    state leaks between runs, and every run uses the same seed so every run sees
    the same arrivals. Those two things are what make the comparison a
    measurement rather than a coincidence.
    """
    results: Dict[str, SimResult] = {}
    for label, strategy in strategies.items():
        controller = IntersectionController(
            signal_id="sim_signal", phases=list(phases), strategy=strategy
        )
        sim = TrafficSimulator(approaches_factory(), controller,
                               start_hour=start_hour, seed=seed)
        results[label] = sim.run(hours=hours, collect_snapshots=False)
    return results
