"""
Verification of Phase 3b: dataset construction, leakage, metrics, baselines and
the ML control strategy.

The leakage tests are the ones that matter. A model evaluated on a leaky split
reports a wonderful score and predicts nothing, and the failure is invisible -
there is no error, just an unearned number. These assert the two properties that
make the evaluation trustworthy: the scaler never sees the test set, and the
split is chronological rather than random.

The LSTM itself is not trained here - it takes minutes. Gradient boosting stands
in for "a learned model", since what is being tested is the surrounding
machinery, not any particular architecture.
"""
import sys, types, math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2 = types.ModuleType("cv2"); cv2.__getattr__ = lambda n: (lambda *a, **k: None)
cv2.FONT_HERSHEY_SIMPLEX = 0; cv2.CAP_PROP_BUFFERSIZE = 38; cv2.CAP_PROP_POS_FRAMES = 1
cv2.VideoCapture = object
sys.modules["cv2"] = cv2
ultra = types.ModuleType("ultralytics"); ultra.YOLO = object
sys.modules["ultralytics"] = ultra

from src.models.baselines import (Metrics, MeanPredictor, PersistencePredictor,
                                  SeasonalNaivePredictor, evaluate,
                                  poisson_noise_floor, run_baselines, skill_score)
from src.models.features import FEATURE_COLUMNS, build_dataset
from src.simulator import ApproachSim, TrafficSimulator, diurnal_multiplier
from src.traffic_controller import (AdaptiveStrategy, IntersectionController,
                                    MLStrategy, Phase, SignalState)

PASS, FAIL = [], []
def check(label, cond, detail=""):
    (PASS if cond else FAIL).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))


def synthetic_rows(days=6, base=8.0, amplitude=6.0, seed=1):
    """A clean daily sine wave plus noise - enough structure to be learnable."""
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for minute in range(days * 1440):
        hour = (minute % 1440) / 60.0
        rate = base + amplitude * math.sin((hour - 6) / 24 * 2 * math.pi)
        crossings = max(0.0, rate + rng.normal(0, 1.0))
        rows.append({
            "minute": (start + timedelta(minutes=minute)).isoformat(sep=" "),
            "crossings": round(crossings, 2),
            "queue_length": int(max(0, crossings / 2)),
            "vehicle_count": round(crossings / 3, 2),
            "avg_dwell_seconds": round(5 + crossings / 4, 2),
            "sample_count": 12,
        })
    return rows


print("\n=== 1. Dataset shape and windowing ===")
rows = synthetic_rows(days=6)
ds = build_dataset(rows, lookback=30, horizon=15)
check("windows have (n, lookback, features) shape",
      ds.X_train.shape[1:] == (30, len(FEATURE_COLUMNS)), f"{ds.X_train.shape}")
check("train/val/test all non-empty",
      len(ds.X_train) and len(ds.X_val) and len(ds.X_test),
      f"{len(ds.X_train)}/{len(ds.X_val)}/{len(ds.X_test)}")
check("roughly a 70/15/15 split",
      abs(len(ds.X_train) / (len(ds.X_train)+len(ds.X_val)+len(ds.X_test)) - 0.70) < 0.02)
check("one target per window", len(ds.y_train) == len(ds.X_train))


print("\n=== 2. The split is CHRONOLOGICAL, not random ===")
check("test period comes entirely after training period",
      min(ds.timestamps_test) > datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=3),
      f"earliest test sample {min(ds.timestamps_test)}")
check("test timestamps are contiguous and ordered",
      all(b > a for a, b in zip(ds.timestamps_test, ds.timestamps_test[1:])))


print("\n=== 3. No scaler leakage - fitted on the training split only ===")
# Rebuild with an enormous spike placed ONLY in the final (test) portion.
spiked = synthetic_rows(days=6)
for r in spiked[-500:]:
    r["crossings"] = 5000.0
ds_clean = build_dataset(synthetic_rows(days=6), lookback=30, horizon=15)
ds_spiked = build_dataset(spiked, lookback=30, horizon=15)
check("a spike confined to the test split does not move the scaler",
      abs(ds_clean.target_mean - ds_spiked.target_mean) < 1e-6
      and abs(ds_clean.target_std - ds_spiked.target_std) < 1e-6,
      f"mean {ds_clean.target_mean:.4f} vs {ds_spiked.target_mean:.4f}")
check("training features are scaled to roughly zero mean",
      abs(float(ds.X_train.mean())) < 0.35, f"{float(ds.X_train.mean()):.3f}")


print("\n=== 4. Gaps in history are filled, not silently spanned ===")
gappy = synthetic_rows(days=6)
del gappy[3000:3120]                     # two-hour outage
ds_gap = build_dataset(gappy, lookback=30, horizon=15)
gaps = [(b - a).total_seconds() for a, b in zip(ds_gap.timestamps_test, ds_gap.timestamps_test[1:])]
check("no window spans a missing period", all(g == 60.0 for g in gaps),
      f"max gap {max(gaps) if gaps else 0}s")


print("\n=== 5. Too little history fails loudly with a useful message ===")
try:
    build_dataset(synthetic_rows(days=1), lookback=30, horizon=15)
    check("insufficient history rejected", False)
except ValueError as e:
    check("insufficient history rejected", "simulate.py generate" in str(e), str(e)[:58])


print("\n=== 6. Metrics are computed correctly ===")
m = evaluate(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]))
check("MAE correct", abs(m.mae - 7/3) < 1e-9, f"{m.mae:.4f}")
check("RMSE correct", abs(m.rmse - math.sqrt(17/3)) < 1e-9, f"{m.rmse:.4f}")
check("bias detects systematic over-prediction",
      evaluate(np.array([10.,10.,10.]), np.array([12.,12.,12.])).bias == 2.0)
check("perfect prediction gives R2 = 1",
      abs(evaluate(np.array([1.,2.,3.]), np.array([1.,2.,3.])).r2 - 1.0) < 1e-9)
check("MAPE is None when the series has many zeros",
      evaluate(np.array([0.,0.,0.,5.]), np.array([1.,1.,1.,5.])).mape is None)


print("\n=== 7. Skill score semantics ===")
better = Metrics(mae=2.0, rmse=3.0, mape=None, bias=0.0, r2=0.9)
ref = Metrics(mae=4.0, rmse=5.0, mape=None, bias=0.0, r2=0.5)
check("halving the error is +50% skill", abs(skill_score(better, ref) - 50.0) < 1e-9)
check("matching the baseline is 0% skill", abs(skill_score(ref, ref)) < 1e-9)
check("losing to the baseline is negative skill", skill_score(ref, better) < 0)


print("\n=== 8. Poisson noise floor ===")
rng = np.random.default_rng(0)
lam = 9.0
draws = rng.poisson(lam, 200000).astype(float)
empirical = float(np.mean(np.abs(draws - lam)))
check("formula matches the empirical Poisson deviation",
      abs(poisson_noise_floor(draws) - empirical) / empirical < 0.02,
      f"formula {poisson_noise_floor(draws):.3f} vs empirical {empirical:.3f}")
check("floor is zero for an all-zero series", poisson_noise_floor(np.zeros(10)) == 0.0)


print("\n=== 9. Baselines behave as described ===")
base = run_baselines(ds)
check("all three baselines evaluated", set(base) == {"mean", "persistence", "seasonal_naive"})
check("mean predictor is the worst", base["mean"].mae > base["persistence"].mae
      and base["mean"].mae > base["seasonal_naive"].mae,
      f"mean {base['mean'].mae:.3f}")
check("seasonal naive exploits the daily cycle",
      base["seasonal_naive"].mae < base["mean"].mae * 0.6,
      f"naive {base['seasonal_naive'].mae:.3f} vs mean {base['mean'].mae:.3f}")
sn = SeasonalNaivePredictor().predict(ds)
check("seasonal naive returns one prediction per test sample", len(sn) == len(ds.y_test_raw))


print("\n=== 10. A learned model beats the naive baseline on learnable data ===")
from src.models.lstm_model import GradientBoostingForecaster
gbr = GradientBoostingForecaster(n_estimators=120).fit(ds)
gbr_metrics = evaluate(ds.y_test_raw, gbr.predict(ds))
gbr_skill = skill_score(gbr_metrics, base["seasonal_naive"])
check("gradient boosting beats seasonal naive", gbr_skill > 5, f"skill {gbr_skill:+.1f}%")

# The floor must match the noise process the data ACTUALLY has. synthetic_rows()
# adds Gaussian noise with sigma = 1.0, for which the expected absolute
# deviation is sigma * sqrt(2/pi) ~ 0.798 - NOT the Poisson floor, which assumes
# a different distribution entirely and here would be about three times too high.
# Applying poisson_noise_floor() to non-Poisson data is precisely the misuse its
# docstring warns about, and doing it by accident is how the first version of
# this test failed.
gaussian_floor = 1.0 * math.sqrt(2 / math.pi)
check("model approaches, but does not beat, the true noise floor",
      gaussian_floor * 0.95 < gbr_metrics.mae < gaussian_floor * 2.0,
      f"MAE {gbr_metrics.mae:.3f} vs Gaussian floor {gaussian_floor:.3f} "
      f"(Poisson floor would be {poisson_noise_floor(ds.y_test_raw):.3f} - wrong model)")
check("beating the true floor would indicate leakage, and does not happen",
      gbr_metrics.mae > gaussian_floor * 0.9)


print("\n=== 11. MLStrategy degrades safely when the model is unavailable ===")
phases = lambda: [Phase(id="ns", camera_ids=["north", "south"], min_green=10, max_green=50),
                  Phase(id="ew", camera_ids=["east", "west"], min_green=10, max_green=50)]
snaps = {c: types.SimpleNamespace(queue_length=5, flow_rate=10.0)
         for c in ("north", "south", "east", "west")}

dead = MLStrategy(predictor=lambda cam, s: None, queue_threshold=2)
plain = AdaptiveStrategy(queue_threshold=2)
p = phases()[0]
check("a model returning None reproduces the adaptive decision exactly",
      dead.should_extend(p, 15.0, snaps) == plain.should_extend(p, 15.0, snaps),
      f"{dead.should_extend(p, 15.0, snaps)}")

def boom(cam, s): raise RuntimeError("model crashed")
crashing = MLStrategy(predictor=boom, queue_threshold=2)
try:
    crashing.should_extend(p, 15.0, snaps)
    check("a crashing model propagates rather than silently mispredicting", False)
except RuntimeError:
    check("a crashing model propagates rather than silently mispredicting", True,
          "caller decides; it does not fail open into bad timing")


print("\n=== 12. MLStrategy cannot break the safety invariants ===")
# A predictor that always screams "surge" tries to hold green forever.
always_surge = MLStrategy(predictor=lambda cam, s: 10_000.0, queue_threshold=2)
ctrl = IntersectionController("s", phases(), always_surge, decision_interval=0.5)
for t in range(4000):
    ctrl.tick(t * 0.5, snaps)
greens = [d for d in ctrl.decisions if d.state is SignalState.YELLOW]
check("max_green still caps an over-eager model",
      max(d.duration_seconds for d in greens) <= 50.0 + 0.6,
      f"longest {max(d.duration_seconds for d in greens)}s")
served = {}
for d in ctrl.decisions:
    if d.state is SignalState.GREEN:
        served[d.phase_id] = served.get(d.phase_id, 0) + 1
check("both phases still served despite the model", len(served) == 2, f"{served}")
seq = [d.state for d in ctrl.decisions]
check("no GREEN -> ALL_RED transition sneaks through",
      not any(a is SignalState.GREEN and b is SignalState.ALL_RED
              for a, b in zip(seq, seq[1:])))


print("\n=== 13. A PERFECT forecast does not beat adaptive at an isolated junction ===")
PEAK = {"north": 1250, "south": 1150, "east": 420, "west": 360}
appr = lambda: [ApproachSim(c, lanes=2 if c in ("north", "south") else 1,
                            peak_vehicles_per_hour=PEAK[c]) for c in PEAK]

class Oracle:
    """Reads the true future arrival rate. No real model can do better."""
    def __init__(self): self.t = 0.0
    def __call__(self, cam, snaps):
        hour = (6.0 + (self.t + 900) / 3600.0) % 24.0
        return PEAK[cam] * diurnal_multiplier(hour) / 60.0

def run(strategy, oracle=None, hours=4.0):
    c = IntersectionController("s", phases(), strategy)
    if oracle is not None:
        real = c.tick
        def tick(t, s):
            oracle.t = t
            return real(t, s)
        c.tick = tick
    return TrafficSimulator(appr(), c, seed=42, start_hour=6.0).run(
        hours=hours, collect_snapshots=False)

o = Oracle()
adaptive_result = run(AdaptiveStrategy(queue_threshold=2))
oracle_result = run(MLStrategy(predictor=o, queue_threshold=2), oracle=o)
gain = (adaptive_result.average_delay - oracle_result.average_delay) / adaptive_result.average_delay * 100
print(f"        adaptive {adaptive_result.average_delay:.2f}s   "
      f"oracle {oracle_result.average_delay:.2f}s   ({gain:+.2f}%)")
check("a perfect forecast adds essentially nothing here (|gain| < 5%)",
      abs(gain) < 5.0,
      "documented negative result: prediction needs coordination or long cycles to pay off")
check("the oracle still produces a safe, working signal",
      oracle_result.total_served > adaptive_result.total_served * 0.97,
      f"served {oracle_result.total_served:.0f} vs {adaptive_result.total_served:.0f}")

print(f"\n{'='*66}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 66)
sys.exit(1 if FAIL else 0)
