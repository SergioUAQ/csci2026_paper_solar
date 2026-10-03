"""Main experiment: compare single models, average ensemble, static stacking,
and a boosting-base + stacking-meta ensemble, across N_SEEDS independent runs.

Base learners: MLP, LSTM, GRU. Default run: small, fixed architecture
(models.DEFAULT_HP). `--tuned` run: TPE-tuned configuration (see below).

Boosting-base + stacking-meta (novel configuration):
  1) MLP trained on the raw target y.
  2) LSTM trained on the residual y - MLP_pred (same input window).
  3) GRU trained on the residual y - (MLP_pred + LSTM_pred).
  4) The three stage outputs [MLP_pred, LSTM_pred, GRU_pred] (not their
     cumulative sum) are combined by a Ridge meta-learner fit on the
     validation split, exactly as in the static-stacking configuration.
With `--tuned`, every base learner uses the per-architecture configuration
found by tune_tpe.py (best_hparams.json) instead of the fixed one, and
results go to results_mx_tpe.csv; everything else is unchanged.

Stage-1 (MLP) is shared with the plain single-model MLP baseline: training
it once per seed and reusing its predictions is equivalent to training a
separate identical model, since the architecture, target, and data are
identical; this halves the compute without changing the comparison.
"""
import os
import sys
import time
import json
import random
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ARR_DIR = os.path.join(HERE, "arrays_mx")
sys.path.insert(0, HERE)
from models import build_model, DEFAULT_HP  # noqa: E402

TUNED = "--tuned" in sys.argv
if TUNED:
    with open(os.path.join(HERE, "best_hparams.json")) as f:
        HP = {k: v for k, v in json.load(f).items()}
    RESULTS_PATH = os.path.join(HERE, "results_mx_tpe.csv")
else:
    HP = DEFAULT_HP
    RESULTS_PATH = os.path.join(HERE, "results_mx.csv")

# `--shard i/n`: run only seeds with seed % n == i, writing to a part file
# (merge the parts afterwards with `--merge`), so seeds can run in parallel.
SHARD = next((a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
              for a in sys.argv if a.startswith("--shard")), None)
FINAL_PATH = RESULTS_PATH
if SHARD:
    SHARD_I, SHARD_N = map(int, SHARD.split("/"))
    RESULTS_PATH = RESULTS_PATH.replace(".csv", f".part{SHARD_I}.csv")

# Test-set predictions (W/m^2) of every configuration are saved per seed so
# that metrics on subsets (e.g. daytime hours) can be computed afterwards.
PRED_DIR = os.path.join(HERE, "preds_tpe" if TUNED else "preds_fixed")
os.makedirs(PRED_DIR, exist_ok=True)
ONLY_SEEDS = None
if "--seeds" in sys.argv:
    ONLY_SEEDS = {int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")}

N_SEEDS = 30
EPOCHS = 80
PATIENCE = 10

X_train = np.load(os.path.join(ARR_DIR, "X_train.npy"))
y_train = np.load(os.path.join(ARR_DIR, "y_train.npy"))
X_val = np.load(os.path.join(ARR_DIR, "X_val.npy"))
y_val = np.load(os.path.join(ARR_DIR, "y_val.npy"))
X_test = np.load(os.path.join(ARR_DIR, "X_test.npy"))
y_test = np.load(os.path.join(ARR_DIR, "y_test.npy"))
scaler_y = joblib.load(os.path.join(ARR_DIR, "scaler_y.pkl"))
WINDOW, N_FEATURES = X_train.shape[1], X_train.shape[2]


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inv(y_scaled):
    return scaler_y.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def metrics(y_true_scaled, y_pred_scaled):
    yt, yp = inv(y_true_scaled), inv(y_pred_scaled)
    return dict(
        mae=mean_absolute_error(yt, yp),
        rmse=np.sqrt(mean_squared_error(yt, yp)),
        r2=r2_score(yt, yp),
    )


def fit(arch, X, y, Xv, yv):
    hp = HP[arch]
    model = build_model(arch, WINDOW, N_FEATURES, hp)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    model.fit(X, y, validation_data=(Xv, yv), epochs=EPOCHS, batch_size=int(hp["batch"]),
              callbacks=[es], verbose=0)
    return model


def flat(pred):
    return np.asarray(pred).ravel()


def run_seed(seed):
    rows = []
    set_seed(seed)

    # --- independent base learners ---
    mlp = fit("MLP", X_train, y_train, X_val, y_val)
    set_seed(seed + 10_000)
    lstm = fit("LSTM", X_train, y_train, X_val, y_val)
    set_seed(seed + 20_000)
    gru = fit("GRU", X_train, y_train, X_val, y_val)

    mlp_tr, mlp_va, mlp_te = flat(mlp.predict(X_train, verbose=0)), flat(mlp.predict(X_val, verbose=0)), flat(mlp.predict(X_test, verbose=0))
    lstm_va, lstm_te = flat(lstm.predict(X_val, verbose=0)), flat(lstm.predict(X_test, verbose=0))
    gru_va, gru_te = flat(gru.predict(X_val, verbose=0)), flat(gru.predict(X_test, verbose=0))

    for name, pred_te in [("MLP", mlp_te), ("LSTM", lstm_te), ("GRU", gru_te)]:
        rows.append(dict(seed=seed, config=name, **metrics(y_test, pred_te)))

    # --- simple average ensemble ---
    avg_te = (mlp_te + lstm_te + gru_te) / 3.0
    rows.append(dict(seed=seed, config="Average", **metrics(y_test, avg_te)))

    # --- static stacking (Ridge meta on validation predictions) ---
    meta_static = Ridge(alpha=1.0, random_state=seed)
    meta_static.fit(np.column_stack([mlp_va, lstm_va, gru_va]), y_val)
    stack_te = meta_static.predict(np.column_stack([mlp_te, lstm_te, gru_te]))
    rows.append(dict(seed=seed, config="StaticStacking", **metrics(y_test, stack_te)))

    # --- boosting-base + stacking-meta ---
    resid1_train = y_train - mlp_tr
    set_seed(seed + 30_000)
    lstm_b = fit("LSTM", X_train, resid1_train, X_val, y_val - mlp_va)
    lstm_b_tr = flat(lstm_b.predict(X_train, verbose=0))
    lstm_b_va = flat(lstm_b.predict(X_val, verbose=0))
    lstm_b_te = flat(lstm_b.predict(X_test, verbose=0))

    combined2_tr = mlp_tr + lstm_b_tr
    combined2_va = mlp_va + lstm_b_va
    resid2_train = y_train - combined2_tr
    set_seed(seed + 40_000)
    gru_b = fit("GRU", X_train, resid2_train, X_val, y_val - combined2_va)
    gru_b_va = flat(gru_b.predict(X_val, verbose=0))
    gru_b_te = flat(gru_b.predict(X_test, verbose=0))

    meta_boost = Ridge(alpha=1.0, random_state=seed)
    meta_boost.fit(np.column_stack([mlp_va, lstm_b_va, gru_b_va]), y_val)
    boost_te = meta_boost.predict(np.column_stack([mlp_te, lstm_b_te, gru_b_te]))
    rows.append(dict(seed=seed, config="BoostingStacking", **metrics(y_test, boost_te)))

    np.savez_compressed(os.path.join(PRED_DIR, f"seed_{seed:02d}.npz"),
                        MLP=inv(mlp_te), LSTM=inv(lstm_te), GRU=inv(gru_te), Average=inv(avg_te),
                        StaticStacking=inv(stack_te), BoostingStacking=inv(boost_te))

    return rows


def merge():
    import glob
    parts = sorted(glob.glob(FINAL_PATH.replace(".csv", ".part*.csv")))
    frames = [pd.read_csv(p) for p in parts]
    if os.path.exists(FINAL_PATH):
        frames.insert(0, pd.read_csv(FINAL_PATH))
    df = pd.concat(frames).drop_duplicates(["seed", "config"]).sort_values(["seed"], kind="stable")
    df.to_csv(FINAL_PATH, index=False)
    for p in parts:
        os.remove(p)
    print(f"Merged {len(parts)} part files -> {FINAL_PATH} ({df['seed'].nunique()} seeds)")


def main():
    if "--merge" in sys.argv:
        return merge()
    completed = set()
    if os.path.exists(RESULTS_PATH):
        prev = pd.read_csv(RESULTS_PATH)
        completed = set(prev["seed"].unique())
        print(f"Resuming: {len(completed)} seeds already completed.")

    for seed in range(N_SEEDS):
        if ONLY_SEEDS is not None and seed not in ONLY_SEEDS:
            continue
        if seed in completed or (SHARD and seed % SHARD_N != SHARD_I):
            continue
        t0 = time.time()
        rows = run_seed(seed)
        df = pd.DataFrame(rows)
        header = not os.path.exists(RESULTS_PATH)
        df.to_csv(RESULTS_PATH, mode="a", header=header, index=False)
        print(f"seed {seed:02d} done in {time.time()-t0:.1f}s", flush=True)

    print("ALL DONE")


if __name__ == "__main__":
    main()
