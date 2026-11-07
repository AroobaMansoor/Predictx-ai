# PredictX AI — Intelligent Predictive Maintenance

**PredictX AI: An Explainable Machine Learning Framework for Intelligent Predictive Maintenance and Industrial Equipment Failure Prediction**

An end-to-end, research-oriented ML system that estimates the probability that an industrial machine fails under given
operating conditions, assigns a configurable Low / Medium / High risk category, explains each prediction with SHAP and
produces advisory, rule-based maintenance recommendations. It includes a six-page Streamlit dashboard, batch CSV
scoring, a CLI, a Google Colab notebook and an automated test suite.

> **Dataset caveat.** The AI4I 2020 dataset is a **synthetic benchmark** whose labels were generated from documented
> rules. Results here describe performance on that benchmark, not on a real production line.
> The system does **not** predict time-to-failure / remaining useful life.

---

## Results (actual run)

<!-- RESULTS_START -->
Run: 2026-11-06 17:54:49, seed 42, 13.5 min on 2 CPU cores. **Deployed model: Random Forest**, chosen by highest validation PR-AUC. It uses engineered features, no extra calibration (it had the lowest validation Brier score) and a decision threshold of 0.696 (F1-optimal on validation).

**Held-out test set** (2,000 machines, 68 failures). Each model uses its own validation-tuned threshold.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | FNR | Latency (1 row) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.479 | 0.515 | 0.496 | 0.461 | 0.920 | 0.0197 | 0.485 | 10.2 ms |
| Random Forest ✅ deployed | 1.000 | 0.750 | 0.857 | 0.903 | 0.972 | 0.0000 | 0.250 | 28.1 ms |
| XGBoost | 0.900 | 0.794 | 0.844 | 0.887 | 0.973 | 0.0031 | 0.206 | 7.7 ms |
| LightGBM | 0.921 | 0.853 | 0.885 | 0.898 | 0.973 | 0.0026 | 0.147 | 7.5 ms |
| CatBoost | 0.877 | 0.838 | 0.857 | 0.873 | 0.970 | 0.0041 | 0.162 | 6.0 ms |
| Neural Network (MLP) | 0.723 | 0.691 | 0.707 | 0.762 | 0.977 | 0.0093 | 0.309 | 8.4 ms |

Deployed-model confusion matrix (test): TN=1932, FP=0, FN=17, TP=51.

**Key findings**
- On test, LightGBM scored a slightly higher F1 (0.885) and recall (0.853) than the deployed Random Forest. The model was still kept, because selection was made on validation, not test. Using test results to change the choice would be test-set tuning.
- Engineered features raised the mean validation PR-AUC across all six models from 0.694 to 0.776 (Experiment 2).
- Class weights, SMOTE and undersampling did not improve on the tuned Random Forest's validation PR-AUC. They raise recall at the 0.5 threshold but cost precision. Tuning the threshold on validation balanced that trade-off better (Experiment 3).
- Top SHAP drivers: Temp. difference [K] (24.8%), Wear x Torque [min·Nm] (23.1%), Rotational speed [rpm] (21.2%), Power [W] (12.8%).
- Test recall by failure mode: Tool Wear Failure 1/10, Heat Dissipation Failure 24/29, Power Failure 13/13, Overstrain Failure 15/16, No mode label 0/2. Tool-wear failures are mostly missed, because in this dataset they occur at a random wear time between 200 and 240 min.
<!-- RESULTS_END -->

Full tables for every experiment: [`outputs/reports/results_summary.md`](outputs/reports/results_summary.md).

---

## Quick start

```bash
cd predictx-ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # exact tested versions: requirements-lock.txt

python train.py            # full pipeline + all experiments (~15–25 min on CPU)
python evaluate.py         # re-score the saved model on the saved test split (reproducibility check)
streamlit run app.py       # dashboard at http://localhost:8501
python -m pytest -q        # 40 tests
```

The package already contains the trained artifacts from the reported run, so `streamlit run app.py` works immediately.
Retraining with seed 42 reproduces the same split; small metric differences are possible with other library versions.
Saved `.joblib` models should be loaded with the versions in `requirements-lock.txt`. If yours differ, retrain.

**CLI prediction**

```bash
python predict.py --type M --air-temp 300 --process-temp 310 --rpm 1500 --torque 40 --tool-wear 120
python predict.py --csv data/processed/test.csv --out outputs/predictions/test_scored.csv
```

**Google Colab:** open `notebooks/PredictX_Training.ipynb`, upload `predictx-ai.zip` when prompted, then *Run all*.
The notebook installs dependencies, runs EDA, trains and tunes every model, shows all graphs and SHAP plots, runs the
tests and downloads the trained artifacts as a ZIP. A GPU is not needed.

Roman Urdu guide: [`docs/QUICKSTART_ROMAN_URDU.md`](docs/QUICKSTART_ROMAN_URDU.md) · Report outline:
[`docs/PROJECT_REPORT_OUTLINE.md`](docs/PROJECT_REPORT_OUTLINE.md)

---

## Dataset setup

Official source: UCI Machine Learning Repository, dataset 601 —
<https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset> (CC BY 4.0).

`data/raw/ai4i2020.csv` is included. If it is missing, `src/data_loader.py` downloads the official ZIP from
`config.yaml → dataset.uci_zip_url`. 10,000 rows × 14 columns:

| Column | Use |
|---|---|
| `UDI`, `Product ID` | identifiers, **never used as features** |
| `Type` (L/M/H) | input (categorical) |
| `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | inputs |
| `Machine failure` | binary target |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | failure-mode labels, used only by the optional failure-mode module, **never as inputs** |

---

## Methodology

**Leakage-safe protocol.**
- Stratified 60 / 20 / 20 train / validation / test split with seed 42.
- All fitting, cross-validation and hyperparameter search use the training split only.
- Model selection, the calibration choice and the decision threshold use the validation split only.
- The test split is scored once, at the end.

**Pipeline** (`imblearn.Pipeline`): `FeatureEngineer → ColumnTransformer → resampler → model`.
- `FeatureEngineer` is stateless.
- `ColumnTransformer` does median imputation and one-hot encodes `type`. CatBoost uses its native categorical handling instead. Logistic Regression and the MLP also get standard scaling.
- SMOTE or undersampling (when used) sit inside the pipeline, so they only touch training folds. Validation and test data are never resampled.

**Engineered features** (`src/feature_engineering.py`):

| Feature | Formula | Unit |
|---|---|---|
| `temp_diff_k` | process temp − air temp | K |
| `power_w` | torque × rpm × 2π / 60 | W |
| `wear_torque_min_nm` | tool wear × torque | min·Nm |
| `temp_diff_x_torque` | temp_diff × torque | K·Nm |

**Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, plus an optional MLP neural network.
- Each model is tuned with `RandomizedSearchCV`: 20 candidates, 5-fold stratified CV, scoring = average precision (PR-AUC).
- The search spaces include class weighting.

**Experiments** (all in `train.py`):
1. Six-model comparison.
2. Baseline vs engineered features.
3. Imbalance strategies (none / class weights / SMOTE / random undersampling).
4. Tuning impact (default vs tuned).
5. Probability calibration (none / sigmoid / isotonic, compared by validation Brier score and ECE).
6. SHAP interpretability.
7. False-negative / false-positive error analysis.
8. Drop-one-feature ablation.
9. Optional multi-label failure-mode classifiers.

**Risk categories** (`config.yaml → risk`):
- Low: p < `low_max` (0.10).
- High: p ≥ `high_min`. The default `auto` sets this to the validation-tuned decision threshold, so *High risk ⇔ predicted Failure*.
- Medium: everything in between.

**Explainability.**
- Explainers: TreeSHAP (RF / XGBoost / LightGBM), CatBoost native SHAP, LinearSHAP (LR), KernelSHAP (MLP).
- The one-hot `type_*` contributions are summed back into a single "Product type" contribution.
- SHAP values are model attributions, not proven physical causes.

**Recommendations** (`src/recommendations.py`):
- Rules combine the risk category with operating checks against the failure-mechanism limits documented with the dataset: tool wear 200–240 min, temp gap < 8.6 K with speed < 1380 rpm, power outside 3500–9000 W, and wear × torque above 11000 / 12000 / 13000 min·Nm for L / M / H.
- They also use the top SHAP risk drivers.
- Output is advisory only.

---

## Project structure

```
predictx-ai/
├── README.md  requirements.txt  requirements-lock.txt  config.yaml
├── app.py                 Streamlit dashboard (6 pages)
├── train.py               full training + experiments
├── evaluate.py            re-evaluation + reproducibility check
├── predict.py             CLI single / batch inference
├── data/raw/ai4i2020.csv  official dataset
├── data/processed/        train / val / test splits written by train.py
├── notebooks/PredictX_Training.ipynb
├── src/
│   ├── data_loader.py        load, download, validate
│   ├── eda.py                EDA statistics, figures, observations
│   ├── preprocessing.py      split, ColumnTransformer
│   ├── feature_engineering.py
│   ├── train_models.py       model registry, search spaces, pipelines
│   ├── evaluation.py         metrics, thresholds, calibration, plots
│   ├── explainability.py     SHAP wrapper and plots
│   ├── inference.py          validation + single / batch prediction
│   ├── recommendations.py    advisory maintenance rules
│   ├── failure_modes.py      optional multi-label module
│   └── utils.py              config, logging, seeds, exceptions
├── models/                predictx_best_model.joblib, all_models/, failure_mode_models.joblib, model_card.json
├── outputs/figures/       EDA, evaluation and SHAP figures
├── outputs/reports/       CSV / JSON metrics, results_summary.md
├── outputs/predictions/   batch prediction outputs
├── outputs/logs/          training logs
├── tests/                 pytest suite
└── docs/                  report outline, Roman Urdu quick start
```

## Batch CSV format

The CSV must have the six input columns. Either the UCI headers (`Type`, `Air temperature [K]`, …) or snake_case names
(`type`, `air_temp_k`, `process_temp_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`) work, and extra columns
are kept. Every row is validated. An invalid row gets a `validation_error` message instead of a prediction, and an input
outside the training range gets a `validation_warning`.

## Limitations

- **Synthetic labels:** failures follow known rules, so engineered features that mirror those rules boost benchmark
  scores more than they would on noisy real-world data.
- **Small failure count:** the test split has only 68 failures, so a single misclassification moves recall by about 1.5
  points.
- **Not covered:** time dynamics, sensor drift, maintenance logs and remaining useful life.
- **Random failures (RNF)** are unpredictable by design.
- **Deployment needs more:** real plant data, monitoring, periodic recalibration and human review.

## References

- S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," AI4I 2020.
- AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository, doi:10.24432/C5HS5C.
- S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
