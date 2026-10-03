"""Daytime-only evaluation and comparison with reference forecasts.

Inputs: preds_fixed/ and preds_tpe/ (per-seed test predictions written by
run_experiment.py), test_meta.csv and baselines.csv (baselines.py), and the
per-seed CSVs (used only as a consistency check).

Outputs:
  * results_day_fixed.csv / results_day_tpe.csv -- per seed and configuration:
    MAE/RMSE/R2 on daytime hours (solar zenith < 85 deg) and MAE at night.
  * mae_masked / rmse_masked: all-hours metrics after setting outputs to zero
    at night (z >= 85 deg), as smart persistence implicitly does.
  * summary_day.csv -- mean/std per configuration and protocol, plus the
    reference forecasts, with forecast skill vs. smart persistence
    (1 - RMSE / RMSE_smart), all hours and daytime.
  * Friedman + Nemenyi on daytime MAE within each protocol.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ["MLP", "LSTM", "GRU", "Average", "StaticStacking", "BoostingStacking"]

meta = pd.read_csv(os.path.join(HERE, "test_meta.csv"))
y, day = meta["y"].to_numpy(), meta["daytime"].to_numpy(bool)
base = pd.read_csv(os.path.join(HERE, "baselines.csv")).set_index("config")
rmse_sp, rmse_sp_day = base.loc["SmartPersistence", "rmse"], base.loc["SmartPersistence", "rmse_day"]


def m(yt, yp):
    return (mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp)), r2_score(yt, yp))


summ = []
for tag, csv in [("fixed", "results_mx.csv"), ("tpe", "results_mx_tpe.csv")]:
    ref = pd.read_csv(os.path.join(HERE, csv)).set_index(["seed", "config"])
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, f"preds_{tag}", "seed_*.npz"))):
        seed = int(os.path.basename(f)[5:7])
        P = np.load(f)
        for c in CONFIGS:
            p = P[c]
            mae, rmse, r2 = m(y, p)
            # consistency: all-hours MAE from saved predictions == CSV
            assert abs(mae - ref.loc[(seed, c), "mae"]) < 1e-3, (tag, seed, c, mae,
                                                                ref.loc[(seed, c), "mae"])
            dmae, drmse, dr2 = m(y[day], p[day])
            # night mask: set outputs to zero when the sun is down (z >= 85 deg),
            # as smart persistence implicitly does
            pm = np.where(day, p, 0.0)
            mmae, mrmse, _ = m(y, pm)
            rows.append(dict(seed=seed, config=c, mae=mae, rmse=rmse, r2=r2, mae_day=dmae,
                             rmse_day=drmse, r2_day=dr2,
                             mae_night=mean_absolute_error(y[~day], p[~day]),
                             mae_masked=mmae, rmse_masked=mrmse))
    df = pd.DataFrame(rows)
    assert df["seed"].nunique() == 30, (tag, df["seed"].nunique())
    df.to_csv(os.path.join(HERE, f"results_day_{tag}.csv"), index=False)

    g = df.groupby("config")
    for c in CONFIGS:
        d = g.get_group(c)
        summ.append(dict(protocol=tag, config=c,
                         mae=d.mae.mean(), mae_std=d.mae.std(), rmse=d.rmse.mean(),
                         mae_day=d.mae_day.mean(), mae_day_std=d.mae_day.std(),
                         rmse_day=d.rmse_day.mean(), r2_day=d.r2_day.mean(),
                         mae_night=d.mae_night.mean(),
                         mae_masked=d.mae_masked.mean(), rmse_masked=d.rmse_masked.mean(),
                         skill=1 - d.rmse.mean() / rmse_sp, skill_day=1 - d.rmse_day.mean() / rmse_sp_day,
                         frac_seeds_beat_sp_rmse_day=(d.rmse_day < rmse_sp_day).mean()))

    wide = df.pivot(index="seed", columns="config", values="mae_day")[CONFIGS]
    chi2, p = stats.friedmanchisquare(*[wide[c] for c in CONFIGS])
    ranks = wide.rank(axis=1).mean().sort_values()
    nem = sp.posthoc_nemenyi_friedman(wide.values)
    nem.index = nem.columns = CONFIGS
    nem.to_csv(os.path.join(HERE, f"nemenyi_day_{tag}.csv"))
    print(f"\n=== {tag}: daytime MAE  Friedman chi2={chi2:.2f} p={p:.2e}")
    print("mean ranks:", ranks.round(2).to_dict())
    print(nem.round(4).to_string())

for name in ["Persistence", "Persistence24h", "SmartPersistence"]:
    b = base.loc[name]
    summ.append(dict(protocol="reference", config=name, mae=b.mae, mae_std=0.0, rmse=b.rmse,
                     mae_day=b.mae_day, mae_day_std=0.0, rmse_day=b.rmse_day, r2_day=b.r2_day,
                     mae_night=np.nan, mae_masked=np.nan, rmse_masked=np.nan,
                     skill=1 - b.rmse / rmse_sp,
                     skill_day=1 - b.rmse_day / rmse_sp_day, frac_seeds_beat_sp_rmse_day=np.nan))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(HERE, "summary_day.csv"), index=False)
print()
print(S.round(3).to_string(index=False))
