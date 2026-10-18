"""End-to-end training pipeline.

Run:  python -m src.train
The notebook (notebooks/capstone.ipynb) calls the same functions step by step.

Order of operations (the test set is touched exactly once, in `evaluate_on_test`):
  1. stratified train/test split
  2. EDA + leakage audit on the training set
  3. baseline + model comparison with stratified 5-fold CV
  4. hyper-parameter search on the two best models
  5. error analysis on out-of-fold predictions -> one change, re-validated by CV
  6. decision threshold chosen on out-of-fold predictions
  7. fit on the full training set, evaluate once on the test set
  8. explainability (SHAP + permutation importance)
  9. spatial aggregation into grid cells
"""
import json
import time
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from . import plots
from .config import (
    CV_FOLDS,
    FEATURES,
    GRID_COLS,
    META_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_URI,
    MODEL_DIR,
    MODEL_PATH,
    REPORTS_DIR,
    ROOT,
    SEED,
    TARGET_COL,
)
from .data import load_data, make_target, split
from .features import drop_correlated
from .models import SEARCH_SPACES, base_estimators, make_pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCORING = {"f1": "f1", "roc_auc": "roc_auc", "precision": "precision", "recall": "recall",
           "accuracy": "accuracy", "average_precision": "average_precision", "neg_log_loss": "neg_log_loss"}
# Hyper-parameter search refits on log-loss: ROC-AUC saturates at 1.000 on this
# dataset, so it cannot tell configurations apart, while log-loss (a proper
# scoring rule) still rewards confident, well-calibrated probabilities.
REFIT = "neg_log_loss"
N_ITER = 40
SEVERITY_TIERS = [  # (name, lower bound on mean P(stressed) in the cell)
    ("Low", 0.0), ("Moderate", 0.25), ("High", 0.5), ("Critical", 0.75),
]


def cv_splitter():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def setup_mlflow(experiment: str = MLFLOW_EXPERIMENT):
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns" / experiment).as_uri())
    mlflow.set_experiment(experiment)


def _r(x, n=4):
    return float(round(float(x), n))


# --------------------------------------------------------------------------- 1-2
def eda(df: pd.DataFrame, train_df: pd.DataFrame):
    """EDA numbers + figures. Correlations and the leakage audit use the training set only."""
    y_all = make_target(df)
    y = make_target(train_df)
    num = df[FEATURES]
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    outliers = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
    corr = train_df[FEATURES + GRID_COLS].corr()
    corr_with_y = train_df[FEATURES + GRID_COLS].corrwith(y).sort_values()
    near_dupes = [
        (a, b, _r(corr.loc[a, b], 3))
        for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
        if abs(corr.loc[a, b]) > 0.9
    ]
    cells = df.groupby(GRID_COLS).size()

    # Leakage audit: is the label a simple deterministic function of a few inputs?
    # Fit a linear model on the 3 most label-correlated non-duplicate signals
    # (training set only) and check whether a single cut on that composite
    # reproduces every training label.
    audit_cols = ["ndvi_mean", "canopy_density", "moisture_index"]
    lin = LinearRegression().fit(train_df[audit_cols], y)
    w = lin.coef_ / lin.coef_.sum()
    composite = train_df[audit_cols] @ w
    # best single cut on the composite
    order = np.sort(composite.values)
    cuts = (order[1:] + order[:-1]) / 2
    accs = [((composite < c).astype(int) == y).mean() for c in cuts]
    best_cut = cuts[int(np.argmax(accs))]
    # The weights recovered are ~0.5 / 0.3 / 0.2; test the rounded rule on ALL rows
    rule = 0.5 * df["ndvi_mean"] + 0.3 * df["canopy_density"] + 0.2 * df["moisture_index"]
    rule_cut = float(rule.quantile(y_all.mean()))
    rule_acc = float(((rule <= rule_cut).astype(int) == y_all).mean())

    summary = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_features_used": len(FEATURES),
        "class_counts": {k: int(v) for k, v in df[TARGET_COL].value_counts().items()},
        "stressed_share": _r(y_all.mean()),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "iqr_outliers_per_feature": {k: int(v) for k, v in outliers.items()},
        "out_of_physical_range": {
            "ndvi_max>1": int((df["ndvi_max"] > 1).sum()),
            "evi>1": int((df["evi"] > 1).sum()),
        },
        "grid": {
            "x_range": [int(df.grid_x.min()), int(df.grid_x.max())],
            "y_range": [int(df.grid_y.min()), int(df.grid_y.max())],
            "occupied_cells": int(len(cells)),
            "total_cells": int((df.grid_x.max() + 1) * (df.grid_y.max() + 1)),
            "obs_per_cell_mean": _r(cells.mean(), 2),
            "obs_per_cell_min": int(cells.min()),
            "obs_per_cell_max": int(cells.max()),
        },
        "corr_with_stressed_train": {k: _r(v, 3) for k, v in corr_with_y.items()},
        "near_duplicate_pairs_abs_r_gt_0.9": near_dupes,
        "nir_vs_ndvi_r": _r(corr.loc["nir_reflectance", "ndvi_mean"], 3),
        "leakage_audit": {
            "composite_features": audit_cols,
            "recovered_weights_train": {c: _r(v, 3) for c, v in zip(audit_cols, w)},
            "train_accuracy_single_cut": _r(max(accs)),
            "train_best_cut": _r(best_cut),
            "rule": "stressed <=> 0.5*ndvi_mean + 0.3*canopy_density + 0.2*moisture_index "
                    "is in the bottom 35% of the field",
            "rule_cut_value": _r(rule_cut),
            "rule_accuracy_all_rows": _r(rule_acc),
        },
    }
    plots.class_balance(y_all)
    plots.correlation_heatmap(corr)
    plots.feature_distributions(train_df, y, FEATURES)
    return summary, corr


# --------------------------------------------------------------------------- 3
def compare_models(train_df, y, features=None, log=True):
    rows = []
    for name, est in base_estimators().items():
        pipe = make_pipeline(est, features)
        t0 = time.time()
        res = cross_validate(pipe, train_df, y, cv=cv_splitter(), scoring=SCORING, n_jobs=1)
        row = {"model": name, "fit_seconds": _r(time.time() - t0, 1)}
        for m in SCORING:
            row[f"{m}_mean"] = _r(res[f"test_{m}"].mean())
            row[f"{m}_std"] = _r(res[f"test_{m}"].std())
        rows.append(row)
        if log:
            with mlflow.start_run(run_name=f"cv_{name}"):
                mlflow.set_tags({"stage": "model_comparison", "model": name})
                mlflow.log_params({"cv_folds": CV_FOLDS, "seed": SEED, "n_features": len(features or FEATURES),
                                   **{k: str(v) for k, v in est.get_params().items()}})
                mlflow.log_metrics({k: v for k, v in row.items() if k not in ("model",)})
    cv_df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    plots.cv_comparison(cv_df)
    cv_df.to_csv(REPORTS_DIR / "cv_comparison.csv", index=False)
    return cv_df


# --------------------------------------------------------------------------- 4
def tune(train_df, y, name, features=None, run_label="tune"):
    search = RandomizedSearchCV(
        make_pipeline(base_estimators()[name], features),
        SEARCH_SPACES[name], n_iter=N_ITER, scoring=SCORING, refit=REFIT,
        cv=cv_splitter(), random_state=SEED, n_jobs=-1,
    )
    search.fit(train_df, y)
    i = search.best_index_
    r = search.cv_results_
    out = {"model": name, "best_params": {k: (v.item() if hasattr(v, "item") else v)
                                          for k, v in search.best_params_.items()}}
    for m in SCORING:
        out[f"{m}_mean"] = _r(r[f"mean_test_{m}"][i])
        out[f"{m}_std"] = _r(r[f"std_test_{m}"][i])
    with mlflow.start_run(run_name=f"{run_label}_{name}"):
        mlflow.set_tags({"stage": run_label, "model": name})
        mlflow.log_params({"n_iter": N_ITER, "seed": SEED, "n_features": len(features or FEATURES),
                           **{k: str(v) for k, v in out["best_params"].items()}})
        mlflow.log_metrics({k: v for k, v in out.items() if k.endswith(("_mean", "_std"))})
        path = REPORTS_DIR / f"search_{run_label}_{name}.csv"
        pd.DataFrame(r).drop(columns=["params"]).to_csv(path, index=False)
        mlflow.log_artifact(str(path))
    return out, search.best_estimator_


# --------------------------------------------------------------------------- 5
def oof_proba(pipe, X, y):
    return cross_val_predict(clone(pipe), X, y, cv=cv_splitter(), method="predict_proba")[:, 1]


def error_analysis(train_df, y, pipe, rule_cut):
    proba = oof_proba(pipe, train_df, y)
    pred = (proba >= 0.5).astype(int)
    err = pred != y.values
    composite = (0.5 * train_df["ndvi_mean"] + 0.3 * train_df["canopy_density"]
                 + 0.2 * train_df["moisture_index"]).values
    dist = np.abs(composite - rule_cut)
    errs = train_df.loc[err, FEATURES].assign(
        y_true=y.values[err], oof_proba=proba[err].round(3), composite=composite[err].round(4),
        dist_to_boundary=dist[err].round(4),
    )
    plots.error_analysis(composite, proba, y.values, err, rule_cut)
    summary = {
        "n_errors": int(err.sum()),
        "false_positives": int(((pred == 1) & (y.values == 0)).sum()),
        "false_negatives": int(((pred == 0) & (y.values == 1)).sum()),
        "max_dist_to_boundary_errors": _r(dist[err].max()) if err.any() else None,
        "median_dist_to_boundary_all": _r(np.median(dist)),
        "share_of_rows_within_error_band": _r((dist <= dist[err].max()).mean()) if err.any() else None,
    }
    return summary, errs, proba


# --------------------------------------------------------------------------- 6
def choose_threshold(y, proba, beta=2.0):
    """Max F-beta (beta=2 weighs recall twice as much as precision) on OOF
    predictions. Ties go to the lowest threshold, i.e. the more cautious one."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        p = (proba >= t).astype(int)
        rows.append({"threshold": t, "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p), "f1": f1_score(y, p),
                     "f2": fbeta_score(y, p, beta=beta), "fn": int(((p == 0) & (y == 1)).sum()),
                     "fp": int(((p == 1) & (y == 0)).sum())})
    df = pd.DataFrame(rows)
    best = float(df.loc[df["f2"].idxmax(), "threshold"])
    plots.threshold_curve(df, best)
    return best, df


# --------------------------------------------------------------------------- 7
def metrics_at(y, proba, thr):
    p = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {"threshold": thr, "accuracy": _r(accuracy_score(y, p)), "precision": _r(precision_score(y, p, zero_division=0)),
            "recall": _r(recall_score(y, p)), "f1": _r(f1_score(y, p)), "roc_auc": _r(roc_auc_score(y, proba)),
            "average_precision": _r(average_precision_score(y, proba)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_on_test(final_pipe, baseline_pipe, dummy_pipe, train_df, y_train, test_df, y_test, thr):
    final_pipe.fit(train_df, y_train)
    baseline_pipe.fit(train_df, y_train)
    dummy_pipe.fit(train_df, y_train)
    proba = final_pipe.predict_proba(test_df)[:, 1]
    base_proba = baseline_pipe.predict_proba(test_df)[:, 1]
    dummy_proba = dummy_pipe.predict_proba(test_df)[:, 1]
    res = {
        "final_at_chosen_threshold": metrics_at(y_test, proba, thr),
        "final_at_0.5": metrics_at(y_test, proba, 0.5),
        "baseline_logreg_at_0.5": metrics_at(y_test, base_proba, 0.5),
        "dummy_majority_at_0.5": metrics_at(y_test, dummy_proba, 0.5),
    }
    plots.confusion(y_test, (proba >= thr).astype(int), f"Test confusion matrix (threshold {thr:.2f})")
    plots.roc_pr(y_test, proba, base_proba)
    return res, proba


# --------------------------------------------------------------------------- 8
def explain(final_pipe, train_df, test_df, y_test):
    import matplotlib.pyplot as plt
    import shap

    pre = final_pipe[:-1]
    clf = final_pipe[-1]
    names = list(pre.get_feature_names_out())
    Xtr = pd.DataFrame(pre.transform(train_df), columns=names)
    Xte = pd.DataFrame(pre.transform(test_df), columns=names)
    explainer = shap.Explainer(clf, Xtr)  # Linear for LR, Tree for tree models
    sv = explainer(Xte)
    vals = sv.values if sv.values.ndim == 2 else sv.values[..., 1]
    shap_rank = pd.Series(np.abs(vals).mean(axis=0), index=names).sort_values(ascending=False)
    plt.figure()
    shap.summary_plot(vals, Xte, show=False, plot_size=(7, 4.5))
    plt.title("SHAP summary (test set, log-odds of stress)")
    plots._save(plt.gcf(), "shap_summary.png")
    plt.close("all")

    pi = permutation_importance(final_pipe, test_df, y_test, scoring="roc_auc",
                                n_repeats=30, random_state=SEED, n_jobs=-1)
    # Permutation importance is computed on raw input columns; keep the model's features.
    cols = list(test_df.columns)
    imp = pd.DataFrame({"feature": cols, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std})
    imp = imp[imp.feature.isin(names)].sort_values("importance_mean", ascending=False).reset_index(drop=True)
    plots.permutation_importance_plot(imp)
    return shap_rank, imp


# --------------------------------------------------------------------------- 9
def morans_i(cell_df, value_col, n_perm=999):
    """Moran's I with rook adjacency on occupied cells, permutation p-value."""
    idx = {(r.grid_x, r.grid_y): i for i, r in enumerate(cell_df.itertuples())}
    n = len(idx)
    W = np.zeros((n, n))
    for (x, yv), i in idx.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((x + dx, yv + dy))
            if j is not None:
                W[i, j] = 1
    z = cell_df[value_col].values - cell_df[value_col].values.mean()

    def stat(v):
        return (n / W.sum()) * (v @ W @ v) / (v @ v)

    obs = stat(z)
    rng = np.random.default_rng(SEED)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(perms >= obs) + 1) / (n_perm + 1)
    return _r(obs), _r(p), _r(-1 / (n - 1))


def spatial(df, final_pipe):
    """Out-of-fold P(stressed) for every row (each row scored by a model that never
    saw it), then aggregated per grid cell."""
    y = make_target(df)
    proba = oof_proba(final_pipe, df, y)
    obs = df[GRID_COLS].assign(proba=proba, stressed=y.values)
    cells = obs.groupby(GRID_COLS).agg(mean_proba=("proba", "mean"), max_proba=("proba", "max"),
                                       observed_rate=("stressed", "mean"), n_obs=("proba", "size")).reset_index()
    bounds = [t[1] for t in SEVERITY_TIERS]
    cells["severity"] = [SEVERITY_TIERS[int(np.searchsorted(bounds, v, side="right") - 1)][0]
                         for v in cells["mean_proba"]]
    n_x, n_y = int(df.grid_x.max() + 1), int(df.grid_y.max() + 1)
    plots.grid_heatmaps(cells, n_x, n_y)
    plots.severity_map(cells, n_x, n_y, SEVERITY_TIERS)
    cells.to_csv(REPORTS_DIR / "grid_cell_stress.csv", index=False)
    mi, p, expected = morans_i(cells, "mean_proba")
    tier_counts = cells["severity"].value_counts().reindex([t[0] for t in SEVERITY_TIERS]).fillna(0).astype(int)
    # Row-wise structure: does stress depend on location at all?
    x_rows = obs.groupby("grid_x")["stressed"].mean()
    y_rows = obs.groupby("grid_y")["stressed"].mean()
    summary = {
        "oof_roc_auc_all_rows": _r(roc_auc_score(y, proba)),
        "cells_occupied": int(len(cells)),
        "cells_empty": int(n_x * n_y - len(cells)),
        "corr_cell_pred_vs_observed": _r(np.corrcoef(cells.mean_proba, cells.observed_rate)[0, 1]),
        "severity_counts": {k: int(v) for k, v in tier_counts.items()},
        "severity_share_of_cells": {k: _r(v / len(cells), 3) for k, v in tier_counts.items()},
        "morans_i": mi, "morans_i_expected_random": expected, "morans_i_p_value": p,
        "stressed_rate_by_grid_x_range": [_r(x_rows.min(), 3), _r(x_rows.max(), 3)],
        "stressed_rate_by_grid_y_range": [_r(y_rows.min(), 3), _r(y_rows.max(), 3)],
        "top10_cells": cells.sort_values(["mean_proba", "n_obs"], ascending=False).head(10)
                            .round(3).to_dict(orient="records"),
    }
    return summary, cells


# --------------------------------------------------------------------------- robustness
def robustness(train_df, y, final_pipe):
    """(a) spatial-block CV: hold out 5x5-cell blocks so neighbours can't leak;
    (b) ablation: drop the three label-defining features."""
    blocks = (train_df["grid_x"] // 5).astype(int) * 10 + (train_df["grid_y"] // 5).astype(int)
    res_block = cross_validate(clone(final_pipe), train_df, y, cv=GroupKFold(n_splits=5),
                               groups=blocks, scoring=["f1", "roc_auc"])
    defining = {"ndvi_mean", "canopy_density", "moisture_index"}
    ablate = [f for f in FEATURES if f not in defining]
    res_abl = cross_validate(make_pipeline(base_estimators()["logreg"], ablate), train_df, y,
                             cv=cv_splitter(), scoring=["f1", "roc_auc"])
    return {
        "spatial_block_cv": {"f1_mean": _r(res_block["test_f1"].mean()), "f1_std": _r(res_block["test_f1"].std()),
                             "roc_auc_mean": _r(res_block["test_roc_auc"].mean()),
                             "roc_auc_std": _r(res_block["test_roc_auc"].std())},
        "ablation_without_ndvi_mean_canopy_moisture_logreg": {
            "features": ablate,
            "f1_mean": _r(res_abl["test_f1"].mean()), "f1_std": _r(res_abl["test_f1"].std()),
            "roc_auc_mean": _r(res_abl["test_roc_auc"].mean()), "roc_auc_std": _r(res_abl["test_roc_auc"].std()),
        },
    }


# --------------------------------------------------------------------------- main
def run_all():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow()
    results = {"seed": SEED}

    df = load_data()
    train_df, test_df = split(df)
    y_train, y_test = make_target(train_df), make_target(test_df)
    results["split"] = {"train_rows": len(train_df), "test_rows": len(test_df),
                        "train_stressed_share": _r(y_train.mean()), "test_stressed_share": _r(y_test.mean())}

    results["eda"], _ = eda(df, train_df)

    cv_df = compare_models(train_df, y_train)
    results["cv_comparison"] = cv_df.to_dict(orient="records")

    candidates = [m for m in cv_df["model"] if m != "dummy_majority"][:2]
    tuned = {}
    for name in candidates:
        tuned[name] = tune(train_df, y_train, name)
    results["tuned"] = {k: v[0] for k, v in tuned.items()}
    best_name = max(tuned, key=lambda k: tuned[k][0]["neg_log_loss_mean"])
    best_pipe = tuned[best_name][1]
    results["best_tuned_model"] = best_name

    # Error analysis on OOF predictions of the tuned model.
    rule_cut = results["eda"]["leakage_audit"]["rule_cut_value"]
    ea_before, errs, _ = error_analysis(train_df, y_train, best_pipe, rule_cut)
    errs.to_csv(REPORTS_DIR / "error_analysis_misclassified.csv")

    # Change motivated by the error analysis + EDA: drop near-duplicate NDVI-family
    # features (|r| > 0.9 with an earlier feature, measured on the training set),
    # then re-run the same search.
    kept, dropped = drop_correlated(train_df, FEATURES, 0.9)
    after_cv, after_pipe = tune(train_df, y_train, best_name, kept, run_label="tune_dedup")
    ea_after, _, _ = error_analysis(train_df, y_train, after_pipe, rule_cut)
    adopt = (after_cv["neg_log_loss_mean"] > results["tuned"][best_name]["neg_log_loss_mean"]
             and after_cv["f1_mean"] >= results["tuned"][best_name]["f1_mean"])
    # re-draw the error figure for the model we keep
    final_pipe = after_pipe if adopt else best_pipe
    ea_final, _, oof_final = error_analysis(train_df, y_train, final_pipe, rule_cut)
    results["error_analysis"] = {
        "before": {**ea_before, "cv": {k: results["tuned"][best_name][k] for k in
                                       ("f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
                                        "precision_mean", "recall_mean", "neg_log_loss_mean")}},
        "change": "drop near-duplicate features (|r|>0.9): " + ", ".join(dropped),
        "dropped": dropped, "kept_features": kept,
        "after": {**ea_after, "cv": {k: after_cv[k] for k in
                                     ("f1_mean", "f1_std", "roc_auc_mean", "roc_auc_std",
                                      "precision_mean", "recall_mean", "neg_log_loss_mean")},
                  "best_params": after_cv["best_params"]},
        "adopted": bool(adopt),
    }
    final_features = kept if adopt else list(FEATURES)

    thr, thr_df = choose_threshold(y_train.values, oof_final)
    thr_df.to_csv(REPORTS_DIR / "threshold_scan_oof.csv", index=False)
    results["threshold"] = {
        "chosen": thr, "rule": "max F2 on out-of-fold training predictions (recall weighted 2x)",
        "oof_at_chosen": thr_df.loc[thr_df.threshold == thr].round(4).to_dict(orient="records")[0],
        "oof_at_0.5": thr_df.loc[thr_df.threshold == 0.5].round(4).to_dict(orient="records")[0],
    }

    baseline_pipe = make_pipeline(base_estimators()["logreg"])
    dummy_pipe = make_pipeline(base_estimators()["dummy_majority"])
    test_res, test_proba = evaluate_on_test(final_pipe, baseline_pipe, dummy_pipe,
                                            train_df, y_train, test_df, y_test, thr)
    results["test"] = test_res
    f = test_res["final_at_chosen_threshold"]
    b = test_res["baseline_logreg_at_0.5"]
    d = test_res["dummy_majority_at_0.5"]
    results["gain"] = {
        "test_f1_vs_baseline_logreg": _r(f["f1"] - b["f1"]),
        "test_roc_auc_vs_baseline_logreg": _r(f["roc_auc"] - b["roc_auc"]),
        "test_recall_vs_baseline_logreg": _r(f["recall"] - b["recall"]),
        "test_f1_vs_dummy": _r(f["f1"] - d["f1"]),
        "test_roc_auc_vs_dummy": _r(f["roc_auc"] - d["roc_auc"]),
        "cv_f1_tuned_vs_untuned_same_model": _r(results["tuned"][best_name]["f1_mean"]
                                                 - cv_df.set_index("model").loc[best_name, "f1_mean"]),
        "cv_roc_auc_tuned_vs_untuned_same_model": _r(results["tuned"][best_name]["roc_auc_mean"]
                                                      - cv_df.set_index("model").loc[best_name, "roc_auc_mean"]),
    }
    misses = test_df.loc[(test_proba >= thr).astype(int) != y_test.values, FEATURES + GRID_COLS]
    results["test_misclassified"] = misses.assign(
        y_true=y_test[misses.index].values, proba=test_proba[(test_proba >= thr).astype(int) != y_test.values].round(3)
    ).round(4).to_dict(orient="records")

    shap_rank, perm = explain(final_pipe, train_df, test_df, y_test)
    results["shap_mean_abs"] = {k: _r(v) for k, v in shap_rank.items()}
    results["permutation_importance_test_roc_auc"] = perm.round(5).to_dict(orient="records")

    results["spatial"], _ = spatial(df, clone(final_pipe))
    results["robustness"] = robustness(train_df, y_train, final_pipe)

    joblib.dump(final_pipe, MODEL_PATH)
    meta = {"model": best_name, "features": final_features, "threshold": thr,
            "params": {k: str(v) for k, v in final_pipe[-1].get_params().items()},
            "severity_tiers": SEVERITY_TIERS, "seed": SEED,
            "test_metrics": test_res["final_at_chosen_threshold"]}
    META_PATH.write_text(json.dumps(meta, indent=2))

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.set_tags({"stage": "final", "model": best_name})
        mlflow.log_params({"threshold": thr, "n_features": len(final_features), "seed": SEED,
                           **{k: str(v) for k, v in final_pipe[-1].get_params().items()}})
        mlflow.log_metrics({f"test_{k}": v for k, v in f.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in b.items() if isinstance(v, (int, float))})
        for p in sorted((REPORTS_DIR / "figures").glob("*.png")):
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(META_PATH), artifact_path="model")

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    keep = [c for c in runs.columns if c.startswith(("tags.mlflow.runName", "tags.stage", "metrics.", "start_time"))]
    runs[keep].to_csv(REPORTS_DIR / "mlflow_runs.csv", index=False)
    return results


if __name__ == "__main__":
    out = run_all()
    print(json.dumps({k: out[k] for k in ("best_tuned_model", "threshold", "test", "gain")}, indent=2, default=str))
