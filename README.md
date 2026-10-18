# crop-health-stress-ml

Crop stress detection from drone multispectral vegetation indices. This is a Masai School × IIT Roorkee
(iHUB DivyaSampark) capstone: *AI-Based Crop Health Monitoring Using Drone Multispectral Data*.

All numbers below are copied from `reports/metrics.json`, written by `python -m src.train`.
`notebooks/capstone.ipynb` re-runs the same steps and its last cell asserts the numbers match. Random seed 42.

## Problem

Each row is one spatial observation in a field (a cell on a 20×20 grid) with processed vegetation indices from a
drone flight. The task is to predict `crop_health_label` (Healthy / Stressed), map predicted stress across the
field, and turn the map into a drone inspection plan.

## Data

`data/raw/crop_multispectral.csv` (the course Google Sheet, *Synthetic Multispectral Crops Data*).

| | |
|---|---|
| Rows × columns | 1,200 × 16 (13 spectral/canopy features, `grid_x`, `grid_y`, label) |
| Class balance | Healthy 780 (65%), Stressed 420 (35%) |
| Missing values / duplicate rows / IQR outliers | 0 / 0 / 0 |
| Grid | 20 × 20; 379 cells occupied, 21 empty; 1–11 observations per cell (mean 3.17) |
| Oddities | 52 `ndvi_max` and 4 `evi` values > 1; NIR reflectance uncorrelated with NDVI (r = 0.003) |

**Leakage finding.** The label is a deterministic rule: *Stressed* ⇔ `0.5·ndvi_mean + 0.3·canopy_density +
0.2·moisture_index` is in the bottom 35% of the field. This reproduces 1,200 / 1,200 labels. The near-perfect
scores below show that the model recovered this rule. They are not a realistic estimate of field performance.
See *Limitations*.

## Method

1. Stratified 80/20 train/test split first (960 / 240 rows, 35% stressed in both). The test set is used once.
2. EDA and a leakage audit on the training set.
3. Each model is an sklearn `Pipeline`: feature selection → median imputation → standard scaling → classifier,
   so preprocessing is fit on training folds only. `grid_x/grid_y` are not model inputs.
4. Baselines (majority-class dummy, logistic regression), then logistic regression, random forest, XGBoost,
   LightGBM and RBF-SVM, compared with stratified 5-fold CV.
5. `RandomizedSearchCV` (40 candidates) on the top two models, refit on log-loss because ROC-AUC saturates at 1.000.
6. Error analysis on out-of-fold training predictions led to one change (dropping near-duplicate features),
   which was re-validated with the same CV.
7. Decision threshold = the one that maximises F2 (recall weighted ×2) on out-of-fold training predictions.
8. Final fit on all training rows, then one evaluation on the test set.
9. SHAP (linear explainer) and permutation importance (ROC-AUC drop, 30 repeats) on the test set.
10. Out-of-fold probabilities for all 1,200 rows are averaged per grid cell for the heatmap, with severity tiers
    and Moran's I for spatial clustering.
11. MLflow logs every run (params, metrics, artifacts) to `mlflow.db` + `mlruns/`.

## Results

### Cross-validation (training set, stratified 5-fold, threshold 0.5)

| Model | F1 (mean ± std) | ROC-AUC (mean ± std) |
|---|---|---|
| Dummy (majority class) | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic regression (baseline) | 0.9685 ± 0.0127 | 0.9987 ± 0.0009 |
| RBF SVM | 0.9510 ± 0.0117 | 0.9959 ± 0.0016 |
| XGBoost | 0.9361 ± 0.0176 | 0.9941 ± 0.0033 |
| LightGBM | 0.9250 ± 0.0135 | 0.9934 ± 0.0027 |
| Random forest | 0.8920 ± 0.0288 | 0.9819 ± 0.0064 |
| **Tuned logistic regression** (L1, C = 13.92, balanced) | 0.9912 ± 0.0071 | 1.0000 ± 0.0000 |
| Tuned RBF SVM (C = 48.70, γ = 0.0191, balanced) | 0.9777 ± 0.0104 | 0.9990 ± 0.0009 |
| **Tuned LR after error-analysis change** (8 features) | **0.9941 ± 0.0055** | **1.0000 ± 0.0000** |

### Held-out test set (240 rows, threshold 0.47)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|
| **Final: tuned LR, 8 features** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 156 / 0 / 0 / 84 |
| Baseline LR (untuned, 13 features, thr 0.5) | 0.9708 | 0.9753 | 0.9405 | 0.9576 | 0.9981 | 154 / 2 / 5 / 79 |
| Dummy (majority) | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 156 / 0 / 84 / 0 |

Gain over the baseline LR on test: F1 +0.0424, recall +0.0595 (5 missed stressed rows → 0), ROC-AUC +0.0019.
Tuning gain in CV (same model): F1 +0.0227, ROC-AUC +0.0013.

**Top features (mean |SHAP|, log-odds):** ndvi_mean 42.76, canopy_density 23.80, moisture_index 19.62.
Permutation importance gives the same order (ROC-AUC drop 0.351 / 0.114 / 0.069). All other features are
below 0.31 |SHAP| and 0.0 permutation importance.

**Error analysis.** The tuned LR made 6 out-of-fold errors (5 FP, 1 FN). All six lie within 0.0027 of the
decision boundary on the composite score, versus a median distance of 0.095. They are borderline plots, not
noise. Dropping the five NDVI-family near-duplicates (r 0.963–0.981 with `ndvi_mean`) cut errors to 4 (4 FP, 0 FN):
CV F1 went from 0.9912 to 0.9941 and log-loss from 0.0150 to 0.0122.

**Spatial.** 379 cells mapped. Severity: Low 145, Moderate 127, High 67, Critical 40. The predicted and observed
cell stress rates correlate at r = 0.9942. Moran's I = −0.0075 (p = 0.543), so stress is not spatially clustered.

**Robustness.** Spatial-block CV (5×5-cell blocks) gives F1 0.9927 ± 0.0080. Without the three rule features,
LR drops to F1 0.7590 ± 0.0506 and ROC-AUC 0.8992 ± 0.0208.

Figures are in `reports/figures/`: `stress_heatmap.png`, `severity_map.png`, `shap_summary.png`,
`permutation_importance.png`, `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `threshold_selection.png`,
`error_analysis.png`, `cv_comparison.png`, `correlation_heatmap.png`, `feature_distributions.png`, `class_balance.png`.

The course write-up is in [`reports/capstone_report.md`](reports/capstone_report.md).

## Repository layout

```
data/raw/                 dataset (CSV)
notebooks/capstone.ipynb  full analysis, executed top to bottom with outputs
src/                      config, data, features, models, train (pipeline), predict, plots
app/main.py               FastAPI service (/health, /predict)
models/                   saved pipeline (joblib) + metadata (features, threshold, tiers)
reports/                  metrics.json, CSVs, figures, course report, mlflow_runs.csv
tests/                    pytest: API and pipeline tests
mlflow.db, mlruns/        MLflow tracking store
Dockerfile                API container
```

## How to run

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train                    # trains everything, writes models/ and reports/
jupyter nbconvert --to notebook --execute --inplace notebooks/capstone.ipynb
pytest -q                              # 8 tests
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at http://127.0.0.1:5000

uvicorn app.main:app --port 8000       # API locally
docker build -t crop-stress-api . && docker run -p 8000:8000 crop-stress-api
```

The committed `mlflow.db` holds the runs from the reported training session (experiments `crop-stress` for the
script and `crop-stress-notebook` for the notebook). Its artifact paths are absolute paths from that machine, so
delete `mlflow.db` and `mlruns/` and re-run `python -m src.train` to get a fully browsable store locally.
`reports/mlflow_runs.csv` is a plain export of the logged metrics.

### Sample request and response (recorded from the Docker container)

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @request.json
```

Request (first Healthy and first Stressed row of the test set):

```json
{
  "observations": [
    {"ndvi_mean": 0.8576, "ndvi_std": 0.0353, "ndvi_min": 0.7551, "ndvi_max": 1.0073, "gndvi": 0.9127,
     "savi": 0.8606, "evi": 0.9596, "red_edge_1": 0.1836, "red_edge_2": 0.3019, "nir_reflectance": 0.8964,
     "soil_brightness": 0.1071, "canopy_density": 0.8693, "moisture_index": 0.9391, "grid_x": 16, "grid_y": 14},
    {"ndvi_mean": 0.3734, "ndvi_std": 0.0901, "ndvi_min": 0.3154, "ndvi_max": 0.4606, "gndvi": 0.397,
     "savi": 0.2721, "evi": 0.3582, "red_edge_1": 0.112, "red_edge_2": 0.1404, "nir_reflectance": 0.6379,
     "soil_brightness": 0.1453, "canopy_density": 0.6933, "moisture_index": 0.1085, "grid_x": 12, "grid_y": 0}
  ]
}
```

Response:

```json
{
  "model": "logreg",
  "predictions": [
    {"grid_x": 16, "grid_y": 14, "stress_probability": 0.0, "predicted_label": "Healthy",
     "threshold": 0.47, "severity": "Low"},
    {"grid_x": 12, "grid_y": 0, "stress_probability": 1.0, "predicted_label": "Stressed",
     "threshold": 0.47, "severity": "Critical"}
  ]
}
```

Probabilities are rounded to 4 decimals. Rows far from the boundary come out at 0.0 and 1.0 because the classes
are linearly separable.

## Limitations

* **The label is a formula of three inputs**, so test F1 = 1.0 means the model recovered the rule. It is not
  evidence of field accuracy. The ablation (F1 0.759 without those three columns) is a more honest sense of difficulty.
* The data is synthetic: the bands are physically inconsistent (NIR independent of NDVI, NDVI > 1) and stress
  has no spatial structure (Moran's I ≈ 0), unlike real fields.
* There is one field and one date, with no crop type, growth stage, weather, soil or temporal information.
* There are few observations per grid cell (mean 3.17), so cell-level estimates are noisy and 21 cells are empty.

With real data: scouted ground-truth labels (ideally stress type), calibrated reflectance bands, multi-date
flights, thermal imagery, spatial/field-level validation, probability calibration and drift monitoring.
