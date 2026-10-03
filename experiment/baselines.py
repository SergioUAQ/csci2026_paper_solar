"""Reference forecasts and daytime mask for the test split.

Rebuilds the exact windows of prepare_data.py (same files, resampling, gap
rule, chronological split) to recover the timestamp of every forecast target,
and checks that the rebuilt targets match arrays_mx/y_test.npy. Then:

  * Solar geometry: zenith angle at the middle of each target hour (Spencer's
    Fourier-series solar-position equations, as used by NOAA; station clock is UTC-6, as stated in the RUOA
    file header), and Haurwitz clear-sky GHI
        GHI_cs = 1098 cos(z) exp(-0.057 / cos(z)),  cos(z) > 0.
  * Daytime mask: target hours with z < 85 deg.
  * Reference forecasts (no training, deterministic):
      - persistence           y_hat(t+1) = y(t)
      - 24-h persistence      y_hat(t+1) = y(t-23)   (same hour, previous day)
      - smart persistence     y_hat(t+1) = k * GHI_cs(t+1), where k is the
        clear-sky index y/GHI_cs of the most recent hour in the window with
        GHI_cs >= 50 W/m^2 (k clipped to [0, 1.2]; 0 if none).

Writes test_meta.csv (timestamp, target, daytime flag, clear-sky GHI, the
three reference forecasts) and baselines.csv (MAE/RMSE/R2, all hours and
daytime only).
"""
import glob
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_mx")
ARR_DIR = os.path.join(HERE, "arrays_mx")
LAT, LON, TZ_OFFSET_H = 20.703, -100.4473, -6.0
WINDOW = 24
ZENITH_DAY_DEG = 85.0
COLS = ["Temp_Avg", "RH_Avg", "WSpeed_Avg", "WSpeed_Max", "WDir_Avg",
        "WDir_SD", "Rain_Tot", "Press_Avg", "Rad_Avg"]
FEATURES = ["Rad_Avg", "Temp_Avg", "RH_Avg", "WSpeed_Avg", "hour_sin", "hour_cos"]


def load_hourly():
    frames = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "RUOA_jqro_*.csv"))):
        if os.path.getsize(fp) == 0:
            continue
        frames.append(pd.read_csv(fp, skiprows=8, header=None, names=["TIMESTAMP"] + COLS,
                                  na_values=["null", "NAN", "NaN"]))
    raw = pd.concat(frames, ignore_index=True)
    raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"])
    raw = raw.drop_duplicates(subset="TIMESTAMP").set_index("TIMESTAMP").sort_index()
    hourly = raw[["Rad_Avg", "Temp_Avg", "RH_Avg", "WSpeed_Avg"]].resample("h").mean()
    hourly["Rad_Avg"] = hourly["Rad_Avg"].clip(lower=0)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
    return hourly.dropna(subset=FEATURES)


def test_windows(hourly):
    """Same split and gap rule as prepare_data.py; returns, per kept window,
    the timestamps of its 24 inputs and of its target, and raw irradiance."""
    n = len(hourly)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    test_df = hourly.iloc[n_train + n_val:]
    idx, rad = test_df.index, test_df["Rad_Avg"].to_numpy()
    rows = []
    for i in range(len(test_df) - WINDOW):
        span = idx[i:i + WINDOW + 1]
        if ((span[1:] - span[:-1]) == pd.Timedelta(hours=1)).all():
            rows.append((span, rad[i:i + WINDOW + 1]))
    return rows


def cos_zenith(ts_local):
    """Solar position (Spencer 1971 series); ts_local are hour-start local times (UTC-6);
    evaluated at mid-hour."""
    t = pd.DatetimeIndex(ts_local) + pd.Timedelta(minutes=30)
    doy = t.dayofyear.to_numpy()
    hour = (t.hour + t.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * LON - 60 * TZ_OFFSET_H
    ha = np.deg2rad(tst / 4 - 180)
    lat = np.deg2rad(LAT)
    return np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)


def ghi_clearsky(cz):
    cz = np.asarray(cz)
    out = np.zeros_like(cz)
    m = cz > 0
    out[m] = 1098 * cz[m] * np.exp(-0.057 / cz[m])
    return out


def metrics(y, p):
    return dict(mae=mean_absolute_error(y, p), rmse=np.sqrt(mean_squared_error(y, p)),
                r2=r2_score(y, p))


def main():
    hourly = load_hourly()
    rows = test_windows(hourly)
    y_true = np.array([r[1][-1] for r in rows])

    # consistency with the arrays the models were trained/evaluated on
    scaler_y = joblib.load(os.path.join(ARR_DIR, "scaler_y.pkl"))
    y_saved = scaler_y.inverse_transform(
        np.load(os.path.join(ARR_DIR, "y_test.npy")).reshape(-1, 1)).ravel()
    assert len(y_saved) == len(y_true), (len(y_saved), len(y_true))
    assert np.allclose(y_saved, y_true, atol=1e-2), np.abs(y_saved - y_true).max()

    t_target = pd.DatetimeIndex([r[0][-1] for r in rows])
    cz_all = [cos_zenith(r[0]) for r in rows]
    cs_all = [ghi_clearsky(c) for c in cz_all]
    cz_target = np.array([c[-1] for c in cz_all])
    day = cz_target > np.cos(np.deg2rad(ZENITH_DAY_DEG))

    persist = np.array([r[1][-2] for r in rows])            # y(t)
    persist24 = np.array([r[1][0] for r in rows])           # y(t-23) = same hour yesterday
    smart = []
    for (span, rad), cs in zip(rows, cs_all):
        k = 0.0
        for j in range(WINDOW - 1, -1, -1):                 # most recent usable hour
            if cs[j] >= 50:
                k = float(np.clip(rad[j] / cs[j], 0, 1.2))
                break
        smart.append(k * cs[-1])
    smart = np.array(smart)

    meta = pd.DataFrame(dict(timestamp=t_target, y=y_true, daytime=day,
                             cos_zenith=cz_target, ghi_cs=[c[-1] for c in cs_all],
                             persistence=persist, persistence_24h=persist24,
                             smart_persistence=smart))
    meta.to_csv(os.path.join(HERE, "test_meta.csv"), index=False)

    out = []
    for name, p in [("Persistence", persist), ("Persistence24h", persist24),
                    ("SmartPersistence", smart)]:
        a, d = metrics(y_true, p), metrics(y_true[day], p[day])
        out.append(dict(config=name, mae=a["mae"], rmse=a["rmse"], r2=a["r2"],
                        mae_day=d["mae"], rmse_day=d["rmse"], r2_day=d["r2"]))
    pd.DataFrame(out).to_csv(os.path.join(HERE, "baselines.csv"), index=False)

    print(f"test windows: {len(y_true)}  daytime: {day.sum()}  "
          f"period: {t_target.min()} .. {t_target.max()}")
    print("months:", t_target.to_period("M").value_counts().sort_index().to_dict())
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
