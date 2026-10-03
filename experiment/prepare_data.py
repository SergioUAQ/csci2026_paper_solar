"""Prepare real, ground-measured solar irradiance data from RUOA (Red
Universitaria de Observatorios Atmosfericos, UNAM), Atmospheric Observatory
Juriquilla, Queretaro, Mexico. NOT simulated: Rad_Avg is a real pyranometer
measurement (W/m^2). Target: hourly global irradiance one step ahead.
Features: irradiance, temperature, humidity, wind speed, hour-of-day.

Handles real-world gaps explicitly: minute data resampled to hourly means,
then only contiguous (no-gap) 24h+1 windows are kept -- a window is never
allowed to silently span a missing period (e.g. the entire empty October
file, or the partial end-of-March outage).
"""
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_mx")
OUT_DIR = os.path.join(HERE, "arrays_mx")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW = 24  # hourly steps -> 24h look-back
COLS = ["Temp_Avg", "RH_Avg", "WSpeed_Avg", "WSpeed_Max", "WDir_Avg",
        "WDir_SD", "Rain_Tot", "Press_Avg", "Rad_Avg"]
FEATURES = ["Rad_Avg", "Temp_Avg", "RH_Avg", "WSpeed_Avg", "hour_sin", "hour_cos"]
TARGET = "Rad_Avg"

files = sorted(glob.glob(os.path.join(DATA_DIR, "RUOA_jqro_*.csv")))
frames = []
for fp in files:
    if os.path.getsize(fp) == 0:
        print(f"skip empty file: {os.path.basename(fp)}")
        continue
    df = pd.read_csv(fp, skiprows=8, header=None, names=["TIMESTAMP"] + COLS,
                      na_values=["null", "NAN", "NaN"])
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"])
raw = raw.drop_duplicates(subset="TIMESTAMP").set_index("TIMESTAMP").sort_index()
print(f"Raw minute rows: {len(raw)}  range: {raw.index.min()} .. {raw.index.max()}")

# Resample to hourly means (real-world minute data -> hourly forecasting task)
hourly = raw[["Rad_Avg", "Temp_Avg", "RH_Avg", "WSpeed_Avg"]].resample("h").mean()
hourly["Rad_Avg"] = hourly["Rad_Avg"].clip(lower=0)  # nighttime sensor offset -> 0
hourly["hour_sin"] = np.sin(2 * np.pi * hourly.index.hour / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly.index.hour / 24)
n_before = len(hourly)
hourly = hourly.dropna(subset=FEATURES)
print(f"Hourly rows: {n_before} -> {len(hourly)} after dropping incomplete hours "
      f"({n_before - len(hourly)} gap-hours removed, incl. the empty October file "
      f"and other real sensor outages)")

n = len(hourly)
n_train = int(n * 0.70)
n_val = int(n * 0.15)
train_df = hourly.iloc[:n_train]
val_df = hourly.iloc[n_train:n_train + n_val]
test_df = hourly.iloc[n_train + n_val:]
print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

scaler_x = MinMaxScaler().fit(train_df[FEATURES])
scaler_y = MinMaxScaler().fit(train_df[[TARGET]])


def make_windows(sub_df):
    """Only emit a window if all L+1 consecutive hourly timestamps are
    exactly 1h apart -- real gaps (missing month, sensor outages) break
    the window instead of being silently bridged."""
    idx = sub_df.index
    x_scaled = scaler_x.transform(sub_df[FEATURES])
    y_scaled = scaler_y.transform(sub_df[[TARGET]]).ravel()
    X, y = [], []
    dropped = 0
    for i in range(len(sub_df) - WINDOW):
        span = idx[i:i + WINDOW + 1]
        if ((span[1:] - span[:-1]) == pd.Timedelta(hours=1)).all():
            X.append(x_scaled[i:i + WINDOW])
            y.append(y_scaled[i + WINDOW])
        else:
            dropped += 1
    print(f"  windows kept: {len(X)}  dropped (span a gap): {dropped}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


print("train:"); X_train, y_train = make_windows(train_df)
print("val:");   X_val, y_val = make_windows(val_df)
print("test:");  X_test, y_test = make_windows(test_df)

for name, arr in [("X_train", X_train), ("y_train", y_train),
                   ("X_val", X_val), ("y_val", y_val),
                   ("X_test", X_test), ("y_test", y_test)]:
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), arr)
    print(name, arr.shape)

joblib.dump(scaler_y, os.path.join(OUT_DIR, "scaler_y.pkl"))
print("Saved arrays to", OUT_DIR)
