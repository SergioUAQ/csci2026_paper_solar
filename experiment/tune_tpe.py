"""Per-architecture hyperparameter tuning with the Tree-structured Parzen
Estimator (TPE) plus median early-stopping pruning (Optuna).

Why TPE: it is a sequential model-based optimizer, so every evaluation
updates its density model and the next proposal is drawn from the promising
region immediately -- no population has to be filled before learning starts.
On a small (5-D) search space this gives a steep early descent, so a short
budget (N_TRIALS) captures almost all of the attainable gain. Median pruning
additionally aborts trials whose per-epoch validation loss is already worse
than the median of earlier trials at the same epoch, so bad configurations
cost only a few epochs instead of a full training run.

Protocol (no test leakage):
  * Tuning uses a dedicated seed (TUNE_SEED) disjoint from the 30 evaluation
    seeds of run_experiment.py, and only the train/validation splits.
  * Objective = validation MAE in W/m^2 (inverse-scaled).
  * The best configuration per architecture is written to best_hparams.json
    and then reused, frozen, by `run_experiment.py --tuned` for all seeds and
    all six configurations (the residual-stage LSTM/GRU reuse the LSTM/GRU
    configuration).

Usage:
    python tune_tpe.py                  # tune MLP, LSTM, GRU
    python tune_tpe.py --models MLP     # tune a subset (merges into the json)
"""
import os
import sys
import json
import time
import random
import argparse
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import optuna
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
ARR_DIR = os.path.join(HERE, "arrays_mx")
OUT_PATH = os.path.join(HERE, "best_hparams.json")
TRIALS_PATH = os.path.join(HERE, "tpe_trials.csv")

N_TRIALS = 30          # short budget: TPE's gain is concentrated in early trials
TUNE_SEED = 12345      # disjoint from evaluation seeds 0..29
EPOCHS = 80
PATIENCE = 10

X_train = np.load(os.path.join(ARR_DIR, "X_train.npy"))
y_train = np.load(os.path.join(ARR_DIR, "y_train.npy"))
X_val = np.load(os.path.join(ARR_DIR, "X_val.npy"))
y_val = np.load(os.path.join(ARR_DIR, "y_val.npy"))
scaler_y = joblib.load(os.path.join(ARR_DIR, "scaler_y.pkl"))

sys.path.insert(0, HERE)
from models import build_model  # noqa: E402


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inv(y_scaled):
    return scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def suggest(trial):
    """5-D search space shared by the three architectures."""
    return dict(
        units=trial.suggest_categorical("units", [16, 32, 64, 128]),
        n_layers=trial.suggest_int("n_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.4),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch=trial.suggest_categorical("batch", [16, 32, 64, 128]),
    )


class PruningCallback(Callback):
    """Report per-epoch val_loss to Optuna and abort hopeless trials."""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(float(logs["val_loss"]), step=epoch)
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned()


def make_objective(arch):
    def objective(trial):
        hp = suggest(trial)
        tf.keras.backend.clear_session()
        set_seed(TUNE_SEED + trial.number)
        model = build_model(arch, X_train.shape[1], X_train.shape[2], hp)
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS,
                  batch_size=hp["batch"], callbacks=[es, PruningCallback(trial)], verbose=0)
        pred = model.predict(X_val, verbose=0).ravel()
        return mean_absolute_error(inv(y_val), inv(pred))
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["MLP", "LSTM", "GRU"])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    best = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            best = json.load(f)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_trials = []
    for arch in args.models:
        sampler = optuna.samplers.TPESampler(seed=TUNE_SEED, multivariate=True,
                                             n_startup_trials=8)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name=f"tpe_{arch}")
        t0 = time.time()
        study.optimize(make_objective(arch), n_trials=args.trials)
        elapsed = time.time() - t0

        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        best[arch] = dict(**study.best_params, val_mae=study.best_value,
                          n_trials=args.trials, n_pruned=n_pruned, tune_seconds=elapsed)
        print(f"{arch}: best val MAE={study.best_value:.2f} W/m2  params={study.best_params}  "
              f"pruned={n_pruned}/{args.trials}  time={elapsed:.0f}s", flush=True)

        df = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
        df.insert(0, "model", arch)
        all_trials.append(df)

        with open(OUT_PATH, "w") as f:
            json.dump(best, f, indent=2)

    import pandas as pd
    out = pd.concat(all_trials, ignore_index=True)
    out["duration"] = out["duration"].dt.total_seconds()
    header = not os.path.exists(TRIALS_PATH)
    out.to_csv(TRIALS_PATH, mode="a", header=header, index=False)
    print(f"Saved {OUT_PATH} and {TRIALS_PATH}")


if __name__ == "__main__":
    main()
