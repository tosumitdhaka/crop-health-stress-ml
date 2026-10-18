"""Plotting helpers. All figures are written to reports/figures/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from .config import FIG_DIR  # noqa: E402

HEALTHY = "#2a78d6"
STRESSED = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": "#0b0b0b",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    return path


def class_balance(y: pd.Series):
    counts = y.map({0: "Healthy", 1: "Stressed"}).value_counts().reindex(["Healthy", "Stressed"])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(counts.index, counts.values, color=[HEALTHY, STRESSED], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v} ({v / counts.sum():.1%})",
                ha="center", va="bottom", color="#0b0b0b")
    ax.set_ylabel("Observations")
    ax.set_title("Class balance")
    ax.grid(axis="x", visible=False)
    return _save(fig, "class_balance.png")


def correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation (training set)")
    return _save(fig, "correlation_heatmap.png")


def feature_distributions(df: pd.DataFrame, y: pd.Series, features):
    n = len(features)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6 * rows))
    for ax, f in zip(axes.ravel(), features):
        ax.hist(df.loc[y == 0, f], bins=30, alpha=0.6, color=HEALTHY, label="Healthy")
        ax.hist(df.loc[y == 1, f], bins=30, alpha=0.6, color=STRESSED, label="Stressed")
        ax.set_title(f, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False)
    fig.suptitle("Feature distributions by class (training set)", y=1.0)
    fig.tight_layout()
    return _save(fig, "feature_distributions.png")


def cv_comparison(cv_df: pd.DataFrame):
    d = cv_df.sort_values("roc_auc_mean")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, m, title in [(axes[0], "f1", "F1 (threshold 0.5)"), (axes[1], "roc_auc", "ROC-AUC")]:
        ax.barh(d["model"], d[f"{m}_mean"], xerr=d[f"{m}_std"], color=HEALTHY, height=0.55,
                error_kw={"ecolor": INK, "capsize": 3})
        for i, (v, s) in enumerate(zip(d[f"{m}_mean"], d[f"{m}_std"])):
            ax.text(min(v + s + 0.01, 1.02), i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Stratified 5-fold CV on the training set (mean ± std)")
    fig.tight_layout()
    return _save(fig, "cv_comparison.png")


def confusion(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Healthy", "Stressed"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.grid(False)
    ax.set_title(title)
    return _save(fig, "confusion_matrix.png")


def roc_pr(y_true, proba, baseline_proba=None):
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot(fpr, tpr, color=HEALTHY, lw=2, label="Tuned model")
    if baseline_proba is not None:
        bf, bt, _ = roc_curve(y_true, baseline_proba)
        ax.plot(bf, bt, color=STRESSED, lw=2, label="Baseline logistic regression")
    ax.plot([0, 1], [0, 1], ls="--", color=INK, lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (held-out test)")
    ax.legend(frameon=False, loc="lower right")
    p1 = _save(fig, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot(rec, prec, color=HEALTHY, lw=2, label="Tuned model")
    if baseline_proba is not None:
        bp, br, _ = precision_recall_curve(y_true, baseline_proba)
        ax.plot(br, bp, color=STRESSED, lw=2, label="Baseline logistic regression")
    ax.axhline(np.mean(y_true), ls="--", color=INK, lw=1, label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.03)
    ax.set_title("Precision-recall curve (held-out test)")
    ax.legend(frameon=False, loc="lower left")
    p2 = _save(fig, "pr_curve.png")
    return p1, p2


def threshold_curve(thr_df: pd.DataFrame, chosen: float):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(thr_df["threshold"], thr_df["recall"], color=STRESSED, lw=2, label="Recall")
    ax.plot(thr_df["threshold"], thr_df["precision"], color=HEALTHY, lw=2, label="Precision")
    ax.plot(thr_df["threshold"], thr_df["f2"], color="#1baf7a", lw=2, label="F2")
    ax.axvline(chosen, color=INK, ls="--", lw=1)
    ax.text(chosen, 0.02, f" chosen = {chosen:.2f}", color=INK)
    ax.set_xlabel("Decision threshold on P(stressed)")
    ax.set_ylim(0, 1.03)
    ax.set_title("Threshold selection on out-of-fold training predictions")
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, "threshold_selection.png")


def error_analysis(score, proba, y, err_mask, cut):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.scatter(score[y == 0], proba[y == 0], s=10, color=HEALTHY, alpha=0.5, label="Healthy")
    ax.scatter(score[y == 1], proba[y == 1], s=10, color=STRESSED, alpha=0.5, label="Stressed")
    ax.scatter(score[err_mask], proba[err_mask], s=70, facecolors="none", edgecolors="#0b0b0b",
               lw=1.4, label="Misclassified (OOF)")
    ax.axvline(cut, color=INK, ls="--", lw=1)
    ax.set_xlabel("Composite = 0.5·NDVI_mean + 0.3·canopy + 0.2·moisture")
    ax.set_ylabel("Out-of-fold P(stressed)")
    ax.set_title("Errors sit on the class boundary")
    ax.legend(frameon=False, loc="upper right")
    return _save(fig, "error_analysis.png")


def permutation_importance_plot(imp_df: pd.DataFrame):
    d = imp_df.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=HEALTHY, height=0.6,
            error_kw={"ecolor": INK, "capsize": 2})
    ax.set_xlabel("Drop in ROC-AUC when shuffled (test set)")
    ax.set_title("Permutation importance")
    ax.grid(axis="y", visible=False)
    return _save(fig, "permutation_importance.png")


def grid_heatmaps(cell_df: pd.DataFrame, n_x: int, n_y: int):
    """Two panels: mean predicted P(stressed) and observed stressed share, per cell."""
    pred = np.full((n_y, n_x), np.nan)
    obs = np.full((n_y, n_x), np.nan)
    for r in cell_df.itertuples():
        pred[r.grid_y, r.grid_x] = r.mean_proba
        obs[r.grid_y, r.grid_x] = r.observed_rate
    cmap = plt.get_cmap("Oranges").copy()
    cmap.set_bad("#d9d9d6")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, arr, title in [(axes[0], pred, "Predicted stress probability (mean per cell)"),
                           (axes[1], obs, "Observed stressed share (ground truth)")]:
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=0, vmax=1)
        ax.set_xlabel("grid_x")
        ax.set_ylabel("grid_y")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        ax.set_xticks(range(0, n_x, 2))
        ax.set_yticks(range(0, n_y, 2))
    fig.colorbar(im, ax=axes, shrink=0.8, label="0 = healthy · 1 = stressed (grey = no data)")
    return _save(fig, "stress_heatmap.png")


def severity_map(cell_df: pd.DataFrame, n_x: int, n_y: int, tiers):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    names = [t[0] for t in tiers]
    colors = ["#cfe3f7", "#f6c08f", "#eb6834", "#a8320a"][: len(tiers)]
    arr = np.full((n_y, n_x), np.nan)
    for r in cell_df.itertuples():
        arr[r.grid_y, r.grid_x] = names.index(r.severity)
    cmap = ListedColormap(colors)
    cmap.set_bad("#d9d9d6")
    norm = BoundaryNorm(np.arange(-0.5, len(names) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(arr, origin="lower", cmap=cmap, norm=norm)
    for r in cell_df.itertuples():
        if r.severity == names[-1]:
            ax.text(r.grid_x, r.grid_y, "!", ha="center", va="center", color="white", fontsize=8, weight="bold")
    cb = fig.colorbar(im, ax=ax, ticks=range(len(names)), shrink=0.8)
    cb.ax.set_yticklabels(names)
    ax.set_xlabel("grid_x")
    ax.set_ylabel("grid_y")
    ax.set_title("Drone inspection priority per grid cell")
    ax.grid(False)
    return _save(fig, "severity_map.png")
