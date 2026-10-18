# AI-Based Crop Health Monitoring Using Drone Multispectral Data

Capstone report: Masai School × IIT Roorkee (iHUB DivyaSampark)

Code, notebook and figures: <https://github.com/tosumitdhaka/crop-health-stress-ml>. Every number in this
report was copied from `reports/metrics.json`, which `python -m src.train` writes (seed 42).
`notebooks/capstone.ipynb` reproduces the same numbers.

---

## 1. Data understanding (Task 1)

The dataset has 1,200 observations and 16 columns: 13 vegetation-index features, grid coordinates
(`grid_x`, `grid_y` on a 20×20 field) and the label. It is 65% Healthy (780) and 35% Stressed (420). There are
no missing values, no duplicate rows, and no IQR outliers.

| Index | What it captures | Why it signals stress |
|---|---|---|
| NDVI (mean, std, min, max) | (NIR − Red)/(NIR + Red): green biomass and chlorophyll | Stressed leaves absorb less red and reflect less NIR, so NDVI drops. *std* shows patchiness and *min* shows the weakest spot in the plot. |
| GNDVI | Green-band NDVI | More sensitive to chlorophyll and nitrogen in dense canopy, where NDVI saturates. |
| SAVI | NDVI with a soil-brightness correction | More reliable when the canopy is sparse and soil shows through. |
| EVI | NDVI corrected for atmosphere and background, using blue | Keeps sensitivity in dense canopy. |
| Red-edge 1 / 2 | Reflectance at about 700–740 nm | The red edge shifts as chlorophyll falls, often before symptoms are visible. This makes it an early-warning band. |
| NIR reflectance | Leaf internal structure | Falls with wilting and senescence. |
| Soil brightness | Exposed soil | High values mean thin canopy and less reliable indices. |
| Canopy density | Ground cover | Low cover suggests poor stand, gaps or die-back. |
| Moisture index | Canopy water content | Low values indicate water stress, which often comes before NDVI loss. |

**What the EDA showed**
* NDVI mean/min/max, GNDVI, SAVI and EVI are near-duplicates (pairwise r ≥ 0.939). They carry the same signal.
* NIR is uncorrelated with NDVI (r = 0.003), and 52 `ndvi_max` and 4 `evi` values exceed 1. Neither is
  physically possible in real imagery, so the dataset is synthetic.
* **Label leakage:** *Stressed* ⇔ `0.5·ndvi_mean + 0.3·canopy_density + 0.2·moisture_index` is in the bottom 35%
  of the field. This rule matches all 1,200 labels. I recovered the weights from the training set
  (0.554 / 0.275 / 0.170) and then confirmed the rounded rule on all rows.

## 2. Machine-learning model (Task 2)

**Setup.** Stratified 80/20 split (960 train / 240 test). Each model is an sklearn Pipeline (select features →
median impute → scale → classifier), so preprocessing is fit inside each CV training fold. Grid coordinates are
not features. Comparison uses stratified 5-fold CV.

| Model (CV, threshold 0.5) | F1 mean ± std | ROC-AUC mean ± std |
|---|---|---|
| Dummy (majority) | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 |
| Logistic regression | 0.9685 ± 0.0127 | 0.9987 ± 0.0009 |
| RBF SVM | 0.9510 ± 0.0117 | 0.9959 ± 0.0016 |
| XGBoost | 0.9361 ± 0.0176 | 0.9941 ± 0.0033 |
| LightGBM | 0.9250 ± 0.0135 | 0.9934 ± 0.0027 |
| Random forest | 0.8920 ± 0.0288 | 0.9819 ± 0.0064 |

**Tuning.** I ran `RandomizedSearchCV` (40 candidates) on logistic regression and SVM. It refits on log-loss
because ROC-AUC saturates at 1.000.
* Tuned LR (L1, C = 13.92, class_weight = balanced): F1 0.9912 ± 0.0071, ROC-AUC 1.0000 ± 0.0000, log-loss 0.0150.
  That is +0.0227 F1 and +0.0013 AUC over untuned LR.
* Tuned SVM: F1 0.9777 ± 0.0104, ROC-AUC 0.9990 ± 0.0009, log-loss 0.0507.

**Error analysis → change.** The tuned LR made 6 out-of-fold errors (5 FP, 1 FN). All six lie within 0.0027 of
the class boundary on the composite score (the median row is 0.095 away), so they are borderline cases. Five of
the features are noisy near-copies of NDVI mean (r 0.963–0.981), which adds noise right at the boundary. I
dropped them (rule: |r| > 0.9, measured on training data) and re-ran the same search. Out-of-fold errors went
from 6 to 4 (4 FP, 0 FN), CV F1 from 0.9912 to 0.9941, and log-loss from 0.0150 to 0.0122. I adopted the change.

**Threshold.** I chose 0.47 by maximising F2 on out-of-fold training predictions (recall weighted twice as much,
because a missed stressed patch costs more than an extra flight). At 0.47 the OOF results are recall 1.0000,
precision 0.9882, and 0 false negatives.

**Held-out test (240 rows, used once)**

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Final tuned LR (8 features, thr 0.47) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Baseline LR (untuned, 0.5) | 0.9708 | 0.9753 | 0.9405 | 0.9576 | 0.9981 |
| Dummy | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

The final model's confusion matrix is TN 156, FP 0, FN 0, TP 84. The baseline missed 5 stressed rows and raised
2 false alarms. Gain over baseline: F1 +0.0424, recall +0.0595, ROC-AUC +0.0019.

**Explainability.** Mean |SHAP| ranks ndvi_mean (42.76), canopy_density (23.80) and moisture_index (19.62) at the
top. Permutation importance (ROC-AUC drop) gives the same order: 0.351, 0.114, 0.069. Agronomically:
photosynthetic vigour, canopy cover and canopy water. Red-edge, NIR and soil brightness contribute about 0 here,
because the synthetic label doesn't use them. In real crops, red-edge is usually an important early-stress band.

## 3. Spatial analysis (Task 3)

Every row was scored out-of-fold (by a model that never saw it), and the probabilities were averaged per grid
cell (`reports/figures/stress_heatmap.png`). There are 379 occupied cells and 21 empty ones. The predicted and
observed cell stress rates correlate at r = 0.9942. **Moran's I = −0.0075 (p = 0.543)**: stressed cells are
scattered, not clustered.

## 4. Drone and agronomy interpretation (Task 4)

The cell tier is set by the mean predicted probability (`reports/figures/severity_map.png`):

| Tier | Cells | Strategy |
|---|---|---|
| Critical (≥ 0.75) | 40 | Re-fly within 24–48 h at low altitude (about 15–30 m) with multispectral, RGB and thermal. Ground-truth with a scout (soil moisture, tissue sample). Candidate for spot treatment or variable-rate irrigation/fertiliser. |
| High (0.50–0.75) | 67 | Targeted waypoint re-flight within the week, with oblique RGB for pests/disease. |
| Moderate (0.25–0.50) | 127 | Keep in routine surveys at higher frequency. Promote to High if the trend rises on two flights in a row. |
| Low (< 0.25) | 145 | Routine whole-field survey at about 80–120 m on the normal schedule. |

Because stress isn't clustered, the efficient plan is a waypoint route through the 107 Critical + High cells,
not area sweeps.

## 5. Reflection (Task 5)

**Limitations.** The label is a deterministic formula of three inputs, so the perfect test score shows the
model learned the rule. It does not measure field accuracy. Without those three columns, LR reaches F1
0.7590 ± 0.0506 and ROC-AUC 0.8992 ± 0.0208. The bands are physically inconsistent, there is one field on one
date, and there are few observations per cell (mean 3.17).

**Improvements with real data.** Scouted ground-truth labels (by stress type); radiometrically calibrated raw
bands; multi-date flights to track change; thermal imagery for water stress; weather, soil and irrigation
layers; spatial-block or field-level validation (spatial-block CV here gave F1 0.9927 ± 0.0080, the same as
random CV, only because the stress has no spatial structure); probability calibration; and drift monitoring.

---

## Appendix: explaining each step in an interview

* **Why split first?** The test set stands in for future fields. If it influences any choice (features,
  threshold, hyper-parameters), the final score is optimistic. I touched it once, at the end.
* **Why a Pipeline?** Scaling and imputation learn statistics (means, medians). Fitting them on the whole
  dataset before CV leaks validation information into training. Inside a Pipeline they are refit on each
  training fold.
* **Why stratified CV and mean ± std?** It keeps the 35% stressed share in every fold. The std shows how stable a
  score is, so a gap smaller than the std isn't meaningful.
* **Why a dummy baseline?** It shows that 65% accuracy is free. Any useful model must beat it on F1 and ROC-AUC.
* **Why logistic regression won?** The true boundary is a weighted sum of three indices (a straight line). Linear
  models represent that exactly. Trees approximate an oblique line with steps.
* **Why refit the search on log-loss?** Many candidates tie at ROC-AUC 1.000. Log-loss still separates them by
  how confident and correct their probabilities are.
* **Why threshold 0.47 and not 0.5?** Missing stress is costlier than an extra flight. F2 encodes "recall
  matters twice as much". I picked the threshold on out-of-fold training predictions so the test set stays clean.
* **What did the error analysis find?** All misclassified rows sit right on the boundary. Removing noisy
  duplicate NDVI features sharpened the boundary, taking errors from 6 to 4 in CV.
* **How do SHAP and permutation importance differ?** SHAP explains each prediction (how much each feature pushed
  this plot toward stressed). Permutation importance measures how much test performance drops when a feature
  is shuffled. Agreement between the two strengthens the conclusion.
* **Why out-of-fold predictions for the map?** Scoring training rows with a model trained on them would look
  perfect by construction. Out-of-fold scoring gives every row an honest prediction.
* **What is Moran's I?** It measures whether neighbouring cells have similar values. About 0 means a random
  pattern, which tells the drone team to target individual cells rather than zones.
* **Why is a perfect score not a success here?** Because I showed the label is a formula of the inputs. The
  honest claim is that the pipeline is sound and the model recovers the labelling rule. Real-world accuracy
  needs real, independently labelled data.
