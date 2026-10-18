"""Model zoo. Every model is a full Pipeline: select -> impute -> scale -> clf.

Imputation and scaling sit inside the Pipeline, so in cross-validation they are
fit on the training folds only.
"""
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import SEED
from .features import SelectFeatures


def make_pipeline(clf, features=None) -> Pipeline:
    return Pipeline([
        ("features", SelectFeatures(features=features)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def base_estimators() -> dict:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=SEED),
    }


# Search spaces for RandomizedSearchCV (keys prefixed with the pipeline step).
SEARCH_SPACES = {
    "logreg": {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
        "clf__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": [None, 4, 6, 8, 12],
        "clf__min_samples_leaf": randint(1, 20),
        "clf__max_features": ["sqrt", 0.5, 0.8],
        "clf__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "clf__n_estimators": randint(100, 600),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": randint(2, 7),
        "clf__min_child_weight": randint(1, 10),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(1e-2, 10),
    },
    "lightgbm": {
        "clf__n_estimators": randint(100, 600),
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__num_leaves": randint(4, 48),
        "clf__min_child_samples": randint(5, 60),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__subsample_freq": [1],
        "clf__colsample_bytree": uniform(0.5, 0.5),
        "clf__reg_lambda": loguniform(1e-2, 10),
    },
    "svm_rbf": {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__gamma": loguniform(1e-4, 1),
        "clf__class_weight": [None, "balanced"],
    },
}
