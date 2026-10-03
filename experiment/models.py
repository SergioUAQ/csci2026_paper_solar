"""Base-learner constructors shared by run_experiment.py and tune_tpe.py.

`hp` keys: units, n_layers, dropout, lr (batch is used by the caller's fit).
DEFAULT_HP reproduces the original fixed configuration exactly:
  MLP : Dense(32) -> Dropout(0.1) -> Dense(16) -> Dense(1)   (n_layers=2)
  LSTM: LSTM(32) -> Dropout(0.1) -> Dense(1)                (n_layers=1)
  GRU : GRU(32)  -> Dropout(0.1) -> Dense(1)                (n_layers=1)
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam

DEFAULT_HP = {
    "MLP": dict(units=32, n_layers=2, dropout=0.1, lr=1e-3, batch=32),
    "LSTM": dict(units=32, n_layers=1, dropout=0.1, lr=1e-3, batch=32),
    "GRU": dict(units=32, n_layers=1, dropout=0.1, lr=1e-3, batch=32),
}


def build_model(arch, window, n_features, hp):
    units, n_layers, dropout = int(hp["units"]), int(hp["n_layers"]), float(hp["dropout"])
    if arch == "MLP":
        layers = [Flatten(input_shape=(window, n_features)),
                  Dense(units, activation="relu"),
                  Dropout(dropout)]
        if n_layers >= 2:
            layers.append(Dense(units // 2, activation="relu"))
        layers.append(Dense(1))
    elif arch in ("LSTM", "GRU"):
        cell = LSTM if arch == "LSTM" else GRU
        layers = []
        for i in range(n_layers):
            kw = dict(input_shape=(window, n_features)) if i == 0 else {}
            layers.append(cell(units, return_sequences=(i < n_layers - 1), **kw))
        layers += [Dropout(dropout), Dense(1)]
    else:
        raise ValueError(arch)
    m = Sequential(layers)
    m.compile(optimizer=Adam(float(hp["lr"])), loss="mse")
    return m
