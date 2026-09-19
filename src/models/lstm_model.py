"""
LSTM traffic-flow forecaster, plus a dependency-light alternative.

WHY AN LSTM, AND WHETHER IT IS ACTUALLY THE RIGHT CHOICE - worth being straight
about, because the honest answer strengthens the report rather than weakening it:

An LSTM is designed for sequences with long-range dependencies, and traffic has
them: the morning peak's size depends on the whole preceding night, congestion
propagates with a lag. That is the case for it.

The case against is that this particular problem - a strongly periodic
univariate-ish series with a few thousand training rows - is territory where
gradient boosting on lag features usually matches or beats a recurrent network,
trains in two seconds rather than two minutes, and has one hyperparameter worth
tuning instead of six. Deep learning tends to win on this task when there are
many correlated sensors, months of data, and irregular events to learn.

So both are implemented here and the training script runs them side by side
against the baselines. Reporting "we tried an LSTM and gradient boosting against
a seasonal-naive baseline, and here is what actually won" is a considerably
stronger result than "we built an LSTM", whichever one comes out ahead.

ARCHITECTURE NOTES:
Two stacked LSTM layers with dropout, then a dense head. Small on purpose - the
training set is a few thousand windows, and a large network would memorise it.
Early stopping on validation loss with weight restoration does the real
regularisation work; without it the model overfits within about twenty epochs
and the test score quietly degrades while training loss keeps falling.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.baselines import Predictor
from src.utils.logger import get_logger

log = get_logger(__name__)

_TF_HELP = (
    "TensorFlow is not installed.\n"
    "  pip install -r requirements-ml.txt\n"
    "If pip cannot find a build, TensorFlow does not support your Python version "
    "yet (check `python --version`; it typically lags new releases by months). "
    "Either create a virtualenv on a supported Python, or use --model gbr, which "
    "needs only scikit-learn and is often just as accurate on this problem."
)


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_TF_HELP) from exc


class LSTMForecaster(Predictor):
    """Stacked LSTM regressor over windowed traffic features."""

    name = "lstm"

    def __init__(self,
                 units: int = 64,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 batch_size: int = 64,
                 patience: int = 10,
                 seed: int = 42):
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.model = None
        self.history = None

    def _build(self, lookback: int, n_features: int):
        tf = _import_tf()
        tf.random.set_seed(self.seed)
        from tensorflow.keras import layers, models, optimizers

        model = models.Sequential([
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(self.units, return_sequences=True),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units // 2),
            layers.Dropout(self.dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            # Huber rather than MSE: traffic has occasional spikes, and squared
            # error lets a handful of them dominate the gradient, dragging the
            # model towards over-predicting the quiet majority of minutes.
            loss="huber",
            metrics=["mae"],
        )
        return model

    def fit(self, dataset, verbose: int = 0) -> "LSTMForecaster":
        tf = _import_tf()
        from tensorflow.keras import callbacks

        self.model = self._build(dataset.lookback, dataset.X_train.shape[-1])

        stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience,
            # Without this the weights left at the end are the OVERFITTED ones
            # from the final epoch, not the best ones seen. It is a one-line
            # difference that routinely costs several percent of accuracy.
            restore_best_weights=True,
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, self.patience // 3), min_lr=1e-5
        )

        self.history = self.model.fit(
            dataset.X_train, dataset.y_train,
            validation_data=(dataset.X_val, dataset.y_val),
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[stopping, reduce_lr], verbose=verbose, shuffle=True,
        )
        epochs_run = len(self.history.history["loss"])
        log.info("LSTM trained for {} epoch(s) (best val_loss {:.4f})",
                 epochs_run, min(self.history.history["val_loss"]))
        return self

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X = dataset.X_test if split == "test" else dataset.X_val
        scaled = self.model.predict(X, verbose=0).flatten()
        return dataset.inverse_target(scaled)

    def save(self, path: str, dataset) -> None:
        """
        Persist the network AND the scaling constants.

        The scaler is not an afterthought: a saved model that receives unscaled
        input produces confident nonsense, with no error. Model and scaler are
        one artefact and must travel together.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path_obj.with_suffix(".keras"))
        np.savez(
            path_obj.with_suffix(".scaler.npz"),
            feature_means=dataset.feature_means, feature_stds=dataset.feature_stds,
            target_mean=dataset.target_mean, target_std=dataset.target_std,
            lookback=dataset.lookback, horizon=dataset.horizon,
        )
        log.info("Saved model to {} and scaler alongside it", path_obj.with_suffix(".keras"))


class GradientBoostingForecaster(Predictor):
    """
    Gradient boosting on flattened lag features - the pragmatic alternative.

    The windowed sequence is flattened into one long feature vector, which
    discards the explicit notion of time order. That sounds like a serious
    handicap and often is not: with a fixed-length window the model can still
    learn "the value 3 steps ago matters most", it just has to discover the
    ordering rather than being handed it.

    Needs only scikit-learn, trains in seconds, and gives the LSTM something
    real to beat.
    """

    name = "gbr"

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.05, seed: int = 42):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, random_state=seed)
        self.model = None

    def fit(self, dataset, verbose: int = 0) -> "GradientBoostingForecaster":
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:
            raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

        self.model = HistGradientBoostingRegressor(
            max_iter=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            random_state=self.params["random_state"],
            early_stopping=True, validation_fraction=0.15,
        )
        n = len(dataset.X_train)
        self.model.fit(dataset.X_train.reshape(n, -1), dataset.y_train)
        return self

    def predict(self, dataset, split: str = "test") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        X = dataset.X_test if split == "test" else dataset.X_val
        scaled = self.model.predict(X.reshape(len(X), -1))
        return dataset.inverse_target(scaled)
