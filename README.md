# Deep Ensembles Against Smart Persistence for Solar Irradiance Forecasting

Code and results for the paper

> **A Multi-Seed Statistical Evaluation of Deep Ensembles Against Smart Persistence for Solar Irradiance Forecasting**
> S. A. Pérez-Rodríguez, J. M. Álvarez-Alvarado, D.-M. Córdova-Esparza, J.-A. Romero-González, J. Rodríguez-Reséndiz
> Universidad Autónoma de Querétaro — submitted to CSCI 2026 (under review).

The study compares six one-hour-ahead forecasting configurations — three base learners (MLP, LSTM, GRU) and three ensembles (simple averaging, ridge stacking, and residual-correction stacking) — over 30 independent seeds, under two hyperparameter protocols:

* **fixed**: one shared default configuration for all base learners;
* **tuned**: per-architecture hyperparameters selected by a Tree-structured Parzen Estimator (TPE) with median pruning (Optuna), 30 trials per architecture.

Results are validated with Friedman tests, all-pairs Nemenyi post-hoc analysis, Holm-corrected Wilcoxon tests, and Vargha–Delaney effect sizes, and compared with persistence, 24-h persistence, and clear-sky (smart) persistence reference forecasts, over all hours and daytime hours only.

## Repository layout

```
experiment/
  prepare_data.py          hourly aggregation, gap-aware windowing, 70/15/15 split
  models.py                MLP / LSTM / GRU constructors (fixed defaults in DEFAULT_HP)
  tune_tpe.py              TPE + median pruning per architecture -> best_hparams.json
  run_experiment.py        30 seeds x 6 configurations (--tuned, --shard i/n, --merge)
  baselines.py             reference forecasts, solar geometry, daytime mask
  analyze_results.py       Friedman, Nemenyi, Vargha-Delaney (--tuned: + fixed-vs-tuned)
  compare_fixed_tuned.py   all-pairs tables, joint 12-variant analysis, figures (-> figures/)
  daytime_analysis.py      daytime metrics and forecast skill vs. smart persistence
  *.csv, *.json            results reported in the paper
  preds_fixed/, preds_tpe/ per-seed test-set predictions (W/m^2)
```

## Data

The ground-station records are openly available from the **Red Universitaria de Observatorios Atmosféricos (RUOA), UNAM**, station *Juriquilla (jqro)*, Querétaro, México (20.703° N, 100.4473° W, 1945 m a.s.l.): <https://www.ruoa.unam.mx/>.

Download the one-minute files for September 2025 – August 2026 and place them in `experiment/data_mx/` named `RUOA_jqro_YYYY_MM.csv` (October 2025 is an empty file in the original record; keep it empty or omit it). Timestamps are local time (UTC−6). Raw data are not redistributed in this repository.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd experiment

python prepare_data.py                     # -> arrays_mx/
python tune_tpe.py                         # -> best_hparams.json, tpe_trials.csv

# 30 seeds per protocol; CPU, fixed thread count for bit-for-bit reproducibility
export CUDA_VISIBLE_DEVICES="" TF_NUM_INTRAOP_THREADS=5 TF_NUM_INTEROP_THREADS=1
python run_experiment.py                   # fixed protocol  -> results_mx.csv, preds_fixed/
python run_experiment.py --tuned           # tuned protocol  -> results_mx_tpe.csv, preds_tpe/
# optional parallel run: launch `--shard i/4` for i=0..3, then `--merge` (add --tuned as needed)

python baselines.py                        # -> baselines.csv, test_meta.csv
python analyze_results.py
python analyze_results.py --tuned
python compare_fixed_tuned.py              # -> tables + figures/
python daytime_analysis.py                 # -> results_day_*.csv, summary_day.csv
```

Notes on reproducibility: results were produced with TensorFlow 2.15 on CPU with the thread settings above; a different thread count or a GPU changes floating-point reduction order and therefore the trained weights. Hyperparameter tuning used a dedicated seed (12345), disjoint from the evaluation seeds 0–29, and only the training and validation splits.

## License

Code is released under the MIT License (see `LICENSE`). RUOA data are subject to the terms of their provider.

## Contact

Sergio A. Pérez-Rodríguez — sergio.perez@uaq.mx
