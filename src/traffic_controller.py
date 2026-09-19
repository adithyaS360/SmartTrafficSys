"""
Traffic signal control.

THE SHAPE OF THIS MODULE, and why it is a state machine rather than a function
that returns a green duration:

A signal has safety invariants that must hold no matter what any algorithm
decides. Green must never become red without yellow in between. Two conflicting
approaches must never be green together. A green shorter than a few seconds
traps drivers who have already started moving. These are not preferences to be
traded against throughput - violating one is a collision.

So the invariants live in the state machine, which owns all transitions, and the
"clever" part is confined to a Strategy that may only answer one question:
should the current green be extended, or should it end now? A strategy cannot
skip yellow, cannot run green past max_green, and cannot cut it below min_green,
because it is never given the opportunity. A buggy strategy degrades timing;
it cannot produce an unsafe signal.

That split is also what makes the ML work land cleanly later: an MLStrategy is a
new subclass, and the controller does not change.

THREE STRATEGIES:
    FixedTimeStrategy   the baseline every real deployment is compared against.
                        Fixed durations, ignores traffic entirely.
    AdaptiveStrategy    extends green while vehicles are still queued and
                        arriving. This is the Phase 3 deliverable.
    (MLStrategy)        Phase 3b - predicts the next few minutes and pre-empts.

The baseline is not scaffolding to be deleted. "Adaptive reduces delay by X%" is
only a claim you can make if you ran the fixed-time case through the same
simulator, the same detector and the same code path. Keep it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence

from src.utils.logger import get_logger

log = get_logger(__name__)


class SignalState(str, Enum):
    """
    States of one intersection.

    Note there is no 'RED' state for the intersection as a whole. Red is not
    something the intersection is in - it is what every approach that is not
    currently served is showing. The intersection is either serving a phase
    (GREEN), clearing it (YELLOW), or in the all-red gap between phases.
    """

    GREEN = "green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass
class Phase:
    """
    One green interval and the approaches it serves.

    A phase groups approaches that can safely run together - typically the two
    opposing directions of one road. Phases run in a fixed order; what adapts is
    how long each one gets, not which comes next. Reordering phases dynamically
    is possible but confuses drivers who have learned the junction's rhythm, and
    it is not what this project claims to do.
    """

    id: str
    camera_ids: List[str]
    min_green: float = 10.0
    max_green: float = 60.0
    yellow: float = 3.0
    all_red: float = 2.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.min_green < 5.0:
            raise ValueError(
                f"Phase '{self.id}': min_green of {self.min_green}s is unsafe. "
                f"Drivers need time to perceive the change and clear the stop line; "
                f"5s is the floor, and real junctions use 7-15s."
            )
        if self.max_green < self.min_green:
            raise ValueError(
                f"Phase '{self.id}': max_green ({self.max_green}s) is below "
                f"min_green ({self.min_green}s) - the phase could never run."
            )
        if self.yellow < 3.0:
            raise ValueError(
                f"Phase '{self.id}': yellow of {self.yellow}s is unsafe. "
                f"Yellow must cover driver reaction plus stopping distance; "
                f"3s is the minimum and it scales with approach speed."
            )
        if not self.camera_ids:
            raise ValueError(f"Phase '{self.id}' serves no approaches.")


@dataclass
class Decision:
    """A record of one control decision, for the audit trail and the report."""

    timestamp: datetime
    signal_id: str
    phase_id: str
    state: SignalState
    duration_seconds: float
    reason: str
    queue_at_decision: Optional[int] = None
    flow_at_decision: Optional[float] = None
    predicted_flow: Optional[float] = None


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class ControlStrategy(ABC):
    """
    Decides only whether to extend the running green.

    The contract is deliberately narrow. Returning (True, reason) extends by one
    decision interval; (False, reason) ends the phase. The controller applies
    min_green and max_green regardless of the answer, so a strategy cannot
    produce an unsafe signal however wrong it is.

    The reason string is not a log message - it is stored on every decision, so
    you can count how often each rule fired. "We hit max_green 40% of the time
    on the north approach" is a finding; "it seemed to work" is not.
    """

    name = "base"

    @abstractmethod
    def should_extend(self,
                      phase: Phase,
                      elapsed: float,
                      snapshots: Dict[str, object]) -> tuple:
        """Return (extend: bool, reason: str)."""

    @staticmethod
    def _queue(phase: Phase, snapshots: Dict[str, object]) -> int:
        """Total queued vehicles across the approaches this phase serves."""
        return sum(
            getattr(snapshots[c], "queue_length", 0)
            for c in phase.camera_ids if c in snapshots
        )

    @staticmethod
    def _flow(phase: Phase, snapshots: Dict[str, object]) -> float:
        """Total arrival flow across the approaches this phase serves."""
        return sum(
            getattr(snapshots[c], "flow_rate", 0.0)
            for c in phase.camera_ids if c in snapshots
        )


class FixedTimeStrategy(ControlStrategy):
    """
    The baseline: every phase runs for a fixed duration, traffic ignored.

    This is what almost every junction in the world actually does, and it is the
    thing your project has to beat. Keep it exercised - a baseline you cannot
    run is a baseline you cannot cite.
    """

    name = "fixed"

    def __init__(self, green_duration: Optional[float] = None):
        self.green_duration = green_duration

    def should_extend(self, phase, elapsed, snapshots) -> tuple:
        target = self.green_duration if self.green_duration else phase.max_green
        target = max(phase.min_green, min(target, phase.max_green))
        return (elapsed < target, "fixed_schedule")


class AdaptiveStrategy(ControlStrategy):
    """
    Extend green while this phase still has demand and the next one can wait.

    THE RULE: keep the green if vehicles are still queued here, unless a waiting
    phase has been starved for too long.

    WHY IT HELPS: a fixed 30s green wastes time when the queue cleared at 12s,
    and truncates when 40 vehicles are still waiting. Both are pure delay.
    Matching green to the queue recovers it.

    WHY THE STARVATION GUARD EXISTS: "extend while there is demand" on a busy
    main road means the side road is never served. max_green caps one phase, but
    a phase can also be starved across several cycles. Tracking how long each
    phase has waited and forcing a handover is what keeps the junction fair
    rather than merely efficient. A controller that minimises total delay by
    ignoring one approach entirely has optimised the wrong thing.
    """

    name = "adaptive"

    def __init__(self,
                 queue_threshold: int = 2,
                 max_starvation: float = 120.0,
                 gap_out_flow: float = 2.0):
        """
        Args:
            queue_threshold: keep green while at least this many vehicles queue
            max_starvation: force a handover if another phase has waited longer
                than this, however busy the current one is
            gap_out_flow: if arrival flow drops below this (vehicles/min), the
                platoon has passed - end the phase even if a straggler is queued.
                This is "gap-out" in traffic engineering terms.
        """
        self.queue_threshold = queue_threshold
        self.max_starvation = max_starvation
        self.gap_out_flow = gap_out_flow
        self._waiting_since: Dict[str, float] = {}

    def note_waiting(self, phase_id: str, since: float) -> None:
        self._waiting_since[phase_id] = since

    def should_extend(self, phase, elapsed, snapshots) -> tuple:
        queue = self._queue(phase, snapshots)
        flow = self._flow(phase, snapshots)

        starved = [
            pid for pid, since in self._waiting_since.items()
            if pid != phase.id and since > self.max_starvation
        ]
        if starved:
            return (False, "starvation_guard")

        if queue >= self.queue_threshold:
            if flow < self.gap_out_flow and queue < self.queue_threshold * 2:
                return (False, "gap_out")
            return (True, "queue_present")

        return (False, "queue_cleared")


class MLStrategy(AdaptiveStrategy):
    """
    Adaptive control with a demand forecast folded in.

    HOW A FORECAST CAN HELP AT ALL, which is worth thinking through before
    assuming it does: the adaptive strategy already sees the current queue, so a
    prediction only adds value where there is a LAG between deciding and the
    decision taking effect. Two such cases exist here:

      - A surge is coming to THIS phase. Holding green a little longer now
        clears the head of it rather than stranding it for a full cycle.
      - A surge is coming to ANOTHER phase. Handing over early means that phase
        starts its green with an empty queue instead of a full one.

    HOW IT CANNOT HELP: at an isolated junction where the controller already
    observes the queue directly and greens last about ten seconds, a
    fifteen-minute forecast is mostly irrelevant - by the time the predicted
    traffic arrives, dozens of cycles have passed and the queue observation has
    long since told the controller everything the forecast could. Prediction
    earns its place on coordinated corridors (green waves between junctions),
    on long cycles, and for anticipating a peak to re-time the whole plan.

    Measure it before claiming it. tools/simulate.py includes an ORACLE that is
    given the true future: if a perfect forecast does not beat plain adaptive
    control in your setup, no real model will, and the honest conclusion is that
    prediction belongs to a different part of the problem.
    """

    name = "ml"

    def __init__(self,
                 predictor,
                 surge_ratio: float = 1.25,
                 lull_ratio: float = 0.75,
                 **kwargs):
        """
        Args:
            predictor: callable(camera_id, snapshots) -> predicted flow per
                minute at the forecast horizon, or None if unavailable.
            surge_ratio: predicted/current above which demand counts as rising
            lull_ratio: predicted/current below which demand counts as falling
        """
        super().__init__(**kwargs)
        self.predictor = predictor
        self.surge_ratio = surge_ratio
        self.lull_ratio = lull_ratio

    def _predicted(self, phase: Phase, snapshots) -> Optional[float]:
        values = [self.predictor(cam, snapshots) for cam in phase.camera_ids]
        usable = [v for v in values if v is not None]
        return sum(usable) if usable else None

    def should_extend(self, phase, elapsed, snapshots) -> tuple:
        extend, reason = super().should_extend(phase, elapsed, snapshots)

        predicted = self._predicted(phase, snapshots)
        current = self._flow(phase, snapshots)
        # No forecast, or nothing to compare against: fall back to the adaptive
        # decision unchanged. A missing model must never break the signal.
        if predicted is None or current <= 0.1:
            return (extend, reason)

        ratio = predicted / current
        queue = self._queue(phase, snapshots)

        if not extend and ratio >= self.surge_ratio and queue >= 1:
            # Demand is rising and someone is still waiting - hold a little longer.
            return (True, "predicted_surge")
        if extend and ratio <= self.lull_ratio and queue <= self.queue_threshold * 2:
            # Demand is falling away - release the junction early.
            return (False, "predicted_lull")
        return (extend, reason)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class IntersectionController:
    """
    Cycles phases, enforces the safety invariants, records every decision.

    Drive it by calling tick() with the current time and the latest snapshot per
    camera. It returns a Decision when the state changed, otherwise None.
    """

    def __init__(self,
                 signal_id: str,
                 phases: Sequence[Phase],
                 strategy: ControlStrategy,
                 repository=None,
                 decision_interval: float = 1.0):
        """
        Args:
            signal_id: identifier used in the signal_events table
            phases: the fixed cycle order, at least two
            strategy: decides extensions only
            repository: TrafficRepository. Decisions are persisted when given.
                Optional so the controller can be unit tested and simulated
                without a database.
            decision_interval: how often the strategy is consulted
        """
        if len(phases) < 2:
            raise ValueError(
                "An intersection needs at least two phases. With one phase "
                "nothing ever yields and there is nothing to control."
            )
        overlapping = self._overlapping_approaches(phases)
        if overlapping:
            raise ValueError(
                f"Approach(es) {sorted(overlapping)} appear in more than one phase. "
                f"They would be given green twice per cycle and conflict with "
                f"themselves - each approach belongs to exactly one phase."
            )

        self.signal_id = signal_id
        self.phases = list(phases)
        self.strategy = strategy
        self.repository = repository
        self.decision_interval = decision_interval

        self._index = 0
        self.state = SignalState.GREEN
        self._state_since = 0.0
        self._last_decision_at = 0.0
        self._phase_waiting_since: Dict[str, float] = {p.id: 0.0 for p in self.phases}
        self.decisions: List[Decision] = []
        self._started = False

    @staticmethod
    def _overlapping_approaches(phases: Sequence[Phase]) -> set:
        seen, dupes = set(), set()
        for phase in phases:
            for cam in phase.camera_ids:
                if cam in seen:
                    dupes.add(cam)
                seen.add(cam)
        return dupes

    # ---- introspection ----------------------------------------------

    @property
    def current_phase(self) -> Phase:
        return self.phases[self._index]

    def elapsed(self, now: float) -> float:
        return now - self._state_since

    def signal_for(self, camera_id: str) -> str:
        """What a given approach is showing right now - 'green', 'yellow' or 'red'."""
        if camera_id not in self.current_phase.camera_ids:
            return "red"
        return "green" if self.state is SignalState.GREEN else (
            "yellow" if self.state is SignalState.YELLOW else "red"
        )

    def is_green_for(self, camera_id: str) -> bool:
        return self.signal_for(camera_id) == "green"

    # ---- the state machine ------------------------------------------

    def tick(self, now: float, snapshots: Dict[str, object]) -> Optional[Decision]:
        """
        Advance the controller. Returns a Decision if the state changed.

        Every transition goes through _transition(), which is the only place
        that mutates state. The invariants therefore hold by construction:
        GREEN can only become YELLOW, YELLOW can only become ALL_RED, and
        ALL_RED can only become GREEN on the next phase.
        """
        if not self._started:
            self._state_since = now
            self._last_decision_at = now
            self._started = True
            return self._record(now, SignalState.GREEN, 0.0, "cycle_start", snapshots)

        elapsed = self.elapsed(now)
        phase = self.current_phase

        if self.state is SignalState.YELLOW:
            if elapsed >= phase.yellow:
                return self._transition(now, SignalState.ALL_RED, "yellow_complete", snapshots)
            return None

        if self.state is SignalState.ALL_RED:
            if elapsed >= phase.all_red:
                self._index = (self._index + 1) % len(self.phases)
                self._phase_waiting_since[self.current_phase.id] = 0.0
                return self._transition(now, SignalState.GREEN, "phase_start", snapshots)
            return None

        # --- GREEN ---
        # Invariant 1: never end a green before min_green, whatever the strategy says.
        if elapsed < phase.min_green:
            return None

        # Invariant 2: never run past max_green, whatever the strategy says.
        # This is the starvation cap and it is not negotiable.
        if elapsed >= phase.max_green:
            return self._transition(now, SignalState.YELLOW, "max_green", snapshots)

        # Between the two bounds, and only there, the strategy has a say.
        if now - self._last_decision_at < self.decision_interval:
            return None
        self._last_decision_at = now

        if isinstance(self.strategy, AdaptiveStrategy):
            for pid, since in self._phase_waiting_since.items():
                if pid != phase.id:
                    self.strategy.note_waiting(pid, now - since if since else 0.0)

        extend, reason = self.strategy.should_extend(phase, elapsed, snapshots)
        if not extend:
            return self._transition(now, SignalState.YELLOW, reason, snapshots)
        return None

    def _transition(self, now: float, new_state: SignalState,
                    reason: str, snapshots: Dict[str, object]) -> Decision:
        duration = self.elapsed(now)
        finished = self.current_phase

        if new_state is SignalState.YELLOW:
            # The phase we are leaving starts waiting from now.
            self._phase_waiting_since[finished.id] = now

        self.state = new_state
        self._state_since = now
        self._last_decision_at = now
        return self._record(now, new_state, duration, reason, snapshots, finished)

    def _record(self, now: float, state: SignalState, duration: float,
                reason: str, snapshots: Dict[str, object],
                phase: Optional[Phase] = None) -> Decision:
        phase = phase or self.current_phase
        decision = Decision(
            timestamp=datetime.now(timezone.utc),
            signal_id=self.signal_id,
            phase_id=phase.id,
            state=state,
            duration_seconds=round(duration, 2),
            reason=reason,
            queue_at_decision=ControlStrategy._queue(phase, snapshots),
            flow_at_decision=round(ControlStrategy._flow(phase, snapshots), 2),
        )
        self.decisions.append(decision)

        if self.repository is not None:
            try:
                self.repository.record_signal_event(
                    signal_id=self.signal_id,
                    camera_id=phase.camera_ids[0] if phase.camera_ids else None,
                    phase=state.value,
                    duration_seconds=decision.duration_seconds,
                    reason=reason,
                    queue_at_decision=decision.queue_at_decision,
                    flow_at_decision=decision.flow_at_decision,
                )
            except Exception as exc:  # noqa: BLE001
                # Losing the audit trail must not stop the signal running.
                log.warning("Could not persist signal decision: {}", exc)

        log.debug("[{}] {} {} after {:.1f}s ({})",
                  self.signal_id, phase.id, state.value, duration, reason)
        return decision

    # ---- reporting ---------------------------------------------------

    def green_durations(self) -> Dict[str, List[float]]:
        """Green durations per phase - the raw material for the report's tables."""
        out: Dict[str, List[float]] = {p.id: [] for p in self.phases}
        for d in self.decisions:
            if d.state is SignalState.YELLOW:
                out.setdefault(d.phase_id, []).append(d.duration_seconds)
        return out

    def reason_counts(self) -> Dict[str, int]:
        """How often each rule ended a phase. Shows what the controller is doing."""
        counts: Dict[str, int] = {}
        for d in self.decisions:
            if d.state is SignalState.YELLOW:
                counts[d.reason] = counts.get(d.reason, 0) + 1
        return counts


def build_phases_from_config(config) -> List[Phase]:
    """Construct Phase objects from the traffic_signals section of config.yaml."""
    phases: List[Phase] = []
    for name, sig in config.signals.items():
        timings = sig.get("timings", {})
        cams = sig.get("camera_ids") or ([sig["camera_id"]] if sig.get("camera_id") else [])
        phases.append(Phase(
            id=sig.get("id", name),
            name=sig.get("name", name),
            camera_ids=cams,
            min_green=float(timings.get("green_min", 10)),
            max_green=float(timings.get("green_max", 60)),
            yellow=float(timings.get("yellow", 3)),
            all_red=float(timings.get("all_red", 2)),
        ))
    return phases
