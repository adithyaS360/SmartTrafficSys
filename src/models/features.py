"""
Turning stored traffic history into a supervised learning problem.

THE PROBLEM STATEMENT, stated precisely, because a vague one produces a model
that scores well and is useless:

    Given the last LOOKBACK minutes of observations at an approach,
    predict the vehicle flow HORIZON minutes from now.

HORIZON is the design decision that matters. Predicting one minute ahead is easy
and pointless - the signal controller already knows the queue in front of it.
Predicting sixty minutes ahead is hard and equally pointless, because the
controller cannot act on it. The useful horizon is the one where a decision
changes: long enough to start clearing a queue before a platoon arrives, short
enough that the prediction is still informative. 10-15 minutes.

TWO RULES THAT DECIDE WHETHER THE RESULT IS REAL:

1. NO LEAKAGE. Every feature for a sample at time t must be computable at time
   t. That sounds obvious and is violated constantly - most often by scaling
   with statistics computed over the whole dataset, which quietly leaks the
   test set's distribution into training. Here the scaler is fitted on the
   TRAINING SPLIT ONLY, in build_dataset, and applied to the others.

2. SPLIT BY TIME, NOT RANDOMLY. A random split puts 10:31 in training and 10:32
   in test. Adjacent minutes of traffic are nearly identical, so the model can
   score brilliantly by memorising neighbours while having learned nothing about
   the future. Time series must be split chronologically: train on the earliest
   days, validate on the middle, test on the most recent. That is also the only
   split that matches how the model would actually be used.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# Columns taken from the per-minute roll-up, in a fixed order. The order is part
# of the model's contract - change it and saved models silently mispredict.
FEATURE_COLUMNS = ("crossings", "queue_length", "vehicle_count", "avg_dwell_seconds")
TARGET_COLUMN = "crossings"


@dataclass
class Dataset:
    """Windowed sequences ready for training, already split and scaled."""

    X_train: np.ndarray          # (n, lookback, n_features)
    y_train: np.ndarray          # (n,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Unscaled targets, needed to report errors in vehicles rather than in
    # standard deviations - "MAE 0.31" means nothing to a reader.
    y_test_raw: np.ndarray
    y_val_raw: np.ndarray
    target_mean: float
    target_std: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    lookback: int
    horizon: int
    timestamps_test: List[datetime]
    # The value at (t - 1440 min) for each test sample: the seasonal-naive
    # prediction, carried alongside so the baseline is evaluated on exactly the
    # same samples as the model.
    seasonal_naive_test: np.ndarray

    def inverse_target(self, scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to vehicles per minute."""
        return scaled * self.target_std + self.target_mean

    def describe(self) -> str:
        return (f"train {self.X_train.shape}  val {self.X_val.shape}  "
                f"test {self.X_test.shape}  lookback={self.lookback}m "
                f"horizon={self.horizon}m")


def _to_matrix(rows: Sequence[dict]) -> Tuple[np.ndarray, List[datetime]]:
    """Convert per-minute dicts to a float matrix plus the timestamp index."""
    values = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows], dtype=np.float32)
    stamps = [datetime.fromisoformat(str(r["minute"])) for r in rows]
    return values, stamps


def _fill_missing_minutes(rows: Sequence[dict]) -> List[dict]:
    """
    Insert zero rows for minutes with no data.

    WHY: per_minute() only returns minutes that had observations. If the
    collector was down from 03:00 to 04:00, those minutes are simply absent, and
    a windowing function that assumes consecutive rows will silently build a
    sequence spanning the gap - treating 02:59 and 04:00 as adjacent. The model
    then learns from sequences that never happened.

    Filling with zeros is honest for traffic (no observation at 3am on a quiet
    road usually does mean no traffic) but it is an assumption, and one to
    revisit on real data where a gap more often means the camera failed.
    """
    if not rows:
        return []
    filled: List[dict] = []
    previous: Optional[datetime] = None
    for row in rows:
        stamp = datetime.fromisoformat(str(row["minute"]))
        if previous is not None:
            gap = int((stamp - previous).total_seconds() // 60) - 1
            if gap > 0:
                log.debug("Filling {} missing minute(s) before {}", gap, stamp)
                for k in range(1, gap + 1):
                    from datetime import timedelta
                    filled.append({"minute": previous + timedelta(minutes=k),
                                   **{c: 0 for c in FEATURE_COLUMNS}})
        filled.append(row)
        previous = stamp
    return filled


def build_dataset(rows: Sequence[dict],
                  lookback: int = 30,
                  horizon: int = 15,
                  train_frac: float = 0.7,
                  val_frac: float = 0.15,
                  seasonal_period: int = 1440) -> Dataset:
    """
    Build windowed, scaled, chronologically split sequences.

    Args:
        rows: output of TrafficRepository.per_minute(), oldest first
        lookback: minutes of history per sample
        horizon: minutes ahead to predict
        train_frac / val_frac: chronological split fractions; the remainder is test
        seasonal_period: minutes in one seasonal cycle - 1440 for daily

    Raises:
        ValueError if there is not enough history to build a single sample.
    """
    rows = _fill_missing_minutes(list(rows))
    needed = lookback + horizon + seasonal_period
    if len(rows) < needed:
        raise ValueError(
            f"Only {len(rows)} minutes of history; need at least {needed} "
            f"(lookback {lookback} + horizon {horizon} + one seasonal period "
            f"{seasonal_period}). Generate more data: "
            f"python tools/simulate.py generate --days 14"
        )

    matrix, stamps = _to_matrix(rows)
    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)

    # Window indices. Start at seasonal_period so every sample has a
    # same-time-yesterday value available for the naive baseline.
    starts = list(range(seasonal_period, len(matrix) - lookback - horizon))
    if not starts:
        raise ValueError("Not enough history after reserving one seasonal period.")

    n = len(starts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # CHRONOLOGICAL split. See the module docstring - a random split here would
    # inflate every score and invalidate the whole comparison.
    train_starts = starts[:n_train]
    val_starts = starts[n_train:n_train + n_val]
    test_starts = starts[n_train + n_val:]

    # Scale using TRAINING ROWS ONLY. Fitting on everything leaks the test
    # distribution into the model and is the most common silent cheat in
    # time-series work.
    train_end_row = train_starts[-1] + lookback if train_starts else len(matrix)
    train_rows = matrix[:train_end_row]
    feature_means = train_rows.mean(axis=0)
    feature_stds = train_rows.std(axis=0)
    feature_stds[feature_stds < 1e-6] = 1.0        # constant column -> no scaling

    scaled = (matrix - feature_means) / feature_stds
    target_mean = float(feature_means[target_index])
    target_std = float(feature_stds[target_index])

    def windows(start_list: List[int]):
        X = np.stack([scaled[s:s + lookback] for s in start_list]) if start_list else \
            np.empty((0, lookback, len(FEATURE_COLUMNS)), dtype=np.float32)
        idx = [s + lookback + horizon - 1 for s in start_list]
        y = np.array([scaled[i, target_index] for i in idx], dtype=np.float32)
        y_raw = np.array([matrix[i, target_index] for i in idx], dtype=np.float32)
        naive = np.array([matrix[max(i - seasonal_period, 0), target_index]
                          for i in idx], dtype=np.float32)
        return X, y, y_raw, naive, [stamps[i] for i in idx]

    X_tr, y_tr, _, _, _ = windows(train_starts)
    X_va, y_va, y_va_raw, _, _ = windows(val_starts)
    X_te, y_te, y_te_raw, naive_te, stamps_te = windows(test_starts)

    log.info("Dataset: train {} / val {} / test {} samples, {} features",
             len(X_tr), len(X_va), len(X_te), len(FEATURE_COLUMNS))

    return Dataset(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te, y_test_raw=y_te_raw, y_val_raw=y_va_raw,
        target_mean=target_mean, target_std=target_std,
        feature_means=feature_means, feature_stds=feature_stds,
        lookback=lookback, horizon=horizon,
        timestamps_test=stamps_te, seasonal_naive_test=naive_te,
    )
