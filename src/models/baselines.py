"""
Baseline predictors and evaluation metrics.

WHY BASELINES COME BEFORE THE LSTM, and why this is the most important file in
the phase:

An LSTM will always produce a number. Reported alone, "MAE 2.4 vehicles/minute"
sounds like a result. It is not a result until you know what MAE a trivial
method achieves on the same data. Traffic is strongly periodic - roughly the
same thing happens at 09:00 every weekday - so "however many vehicles crossed at
this minute yesterday" is already a decent forecast. If the LSTM cannot beat
that, it has learned nothing that the calendar did not already know, however
good its loss curve looked.

This is the single most common failure in student ML projects: an impressive
score with no baseline, where the trivial method would have won. Including the
baselines is what turns "we built an LSTM" into "we established that an LSTM is
or is not worth the complexity here", which is a stronger claim either way.

THREE BASELINES, in increasing order of difficulty to beat:

    MeanPredictor      always predicts the training mean. The floor. A model
                       that loses to this is broken, not merely weak.
    PersistencePredictor  predicts the most recent observed value. Strong on
                       smooth series, weak across turning points - which is
                       exactly where a traffic prediction would be useful.
    SeasonalNaivePredictor  predicts the value at the same minute yesterday.
                       THIS IS THE ONE THAT MATTERS. Beating it is the real bar.

SKILL SCORE is how the comparison is reported: the fractional improvement over
the seasonal-naive baseline. Positive means the model adds something; zero means
it matched the calendar; negative means it is worse than doing nothing clever.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Error metrics in the units of the target: vehicles per minute."""

    mae: float          # mean absolute error - the headline, in vehicles/min
    rmse: float         # penalises large misses more; report both
    mape: Optional[float]   # percentage error; None when the series hits zero
    bias: float         # mean signed error: positive = systematically over-predicting
    r2: float           # fraction of variance explained

    def __str__(self) -> str:
        mape = f"{self.mape:5.1f}%" if self.mape is not None else "   n/a"
        return (f"MAE {self.mae:6.3f}  RMSE {self.rmse:6.3f}  MAPE {mape}  "
                f"bias {self.bias:+6.3f}  R2 {self.r2:6.3f}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute all metrics for one set of predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true

    # MAPE is undefined where the truth is zero, and traffic hits zero overnight.
    # Computing it only over non-zero entries and returning None when there are
    # too few is more honest than the usual epsilon fudge, which inflates the
    # denominator and quietly flatters the model.
    nonzero = np.abs(y_true) > 1e-6
    mape = (float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100)
            if nonzero.sum() > len(y_true) * 0.5 else None)

    variance = float(np.var(y_true))
    r2 = 1.0 - float(np.mean(error ** 2)) / variance if variance > 1e-12 else 0.0

    return Metrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error ** 2))),
        mape=mape,
        bias=float(np.mean(error)),
        r2=r2,
    )


def poisson_noise_floor(y_true: np.ndarray) -> float:
    """
    The best MAE any model could achieve, if arrivals are a Poisson process.

    WHY THIS IS WORTH COMPUTING, and why almost nobody does:

    Vehicle arrivals are random. Even a model that knew the true arrival RATE
    exactly - a perfect oracle - would still mispredict the COUNT, because the
    count fluctuates around the rate by chance. For X ~ Poisson(lambda), the
    expected absolute deviation from the mean approaches sqrt(2*lambda/pi).

    That figure is a hard floor. It tells you how much of your remaining error
    is signal you have failed to extract, and how much is irreducible noise you
    can never extract. Without it you cannot tell a model that is nearly optimal
    from one with plenty of headroom, and you can burn weeks tuning an
    architecture against randomness.

    CAVEAT WORTH STATING IN THE REPORT: this floor assumes Poisson arrivals,
    which is exactly what the simulator generates - so on synthetic data the
    comparison is somewhat self-fulfilling. Real traffic is burstier than
    Poisson (vehicles arrive in platoons released by upstream signals), so the
    real floor is higher and harder to characterise. Treat this as a sanity
    check on synthetic data, not as proof of optimality on real data.
    """
    lam = float(np.mean(y_true))
    if lam <= 0:
        return 0.0
    return float(np.sqrt(2.0 * lam / np.pi))


def skill_score(model: Metrics, reference: Metrics) -> float:
    """
    Fractional improvement in MAE over a reference model, as a percentage.

    +20 means the model's error is 20% lower than the baseline's.
      0 means it matched the baseline and added nothing.
    -15 means it is worse than the trivial method, and should not be deployed
        however good its training curve looked.
    """
    if reference.mae <= 1e-12:
        return 0.0
    return (reference.mae - model.mae) / reference.mae * 100.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class Predictor:
    """Minimal interface shared by baselines and the LSTM."""

    name = "predictor"

    def fit(self, dataset) -> "Predictor":
        return self

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        raise NotImplementedError


class MeanPredictor(Predictor):
    """Always predicts the training mean. The absolute floor."""

    name = "mean"

    def __init__(self):
        self._mean = 0.0

    def fit(self, dataset) -> "MeanPredictor":
        self._mean = float(np.mean(dataset.inverse_target(dataset.y_train)))
        return self

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        n = len(dataset.y_test if split == "test" else dataset.y_val)
        return np.full(n, self._mean, dtype=np.float32)


class PersistencePredictor(Predictor):
    """
    Predicts that flow stays at its most recent observed value.

    Deceptively strong: over a 15-minute horizon traffic usually has not changed
    much, so this is hard to beat on average. It fails precisely at the turning
    points - the start and end of a peak - which is where a forecast would
    actually change a signal decision. Watch its RMSE against its MAE: a large
    gap means it is missing the transitions badly while doing fine in between.
    """

    name = "persistence"

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        X = dataset.X_test if split == "test" else dataset.X_val
        target_index = 0  # crossings is the first feature column
        last_scaled = X[:, -1, target_index]
        return dataset.inverse_target(last_scaled)


class SeasonalNaivePredictor(Predictor):
    """
    Predicts the value observed at the same minute one day earlier.

    THE BAR THE LSTM MUST CLEAR. Traffic is dominated by daily rhythm, so this
    captures most of the structure for free, with no training and no
    dependencies. If a neural network cannot beat it, the honest conclusion is
    that the extra complexity is not earning its place - which is a perfectly
    good finding to report, and a more credible one than an unchallenged score.
    """

    name = "seasonal_naive"

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        if split != "test":
            raise ValueError(
                "Seasonal-naive predictions are precomputed for the test split only."
            )
        return dataset.seasonal_naive_test


def run_baselines(dataset) -> Dict[str, Metrics]:
    """Fit and evaluate every baseline on the test split."""
    results: Dict[str, Metrics] = {}
    for predictor in (MeanPredictor(), PersistencePredictor(), SeasonalNaivePredictor()):
        predictor.fit(dataset)
        results[predictor.name] = evaluate(dataset.y_test_raw, predictor.predict(dataset))
    return results
