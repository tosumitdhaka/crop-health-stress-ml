import numpy as np

from src.config import FEATURES, SEED
from src.data import load_data, make_target, split
from src.features import SelectFeatures, drop_correlated
from src.predict import load_model


def test_split_is_stratified_and_reproducible():
    df = load_data()
    a_tr, a_te = split(df)
    b_tr, b_te = split(df)
    assert list(a_te.index) == list(b_te.index)
    assert len(a_te) == 240 and len(a_tr) == 960
    assert abs(make_target(a_tr).mean() - make_target(a_te).mean()) < 0.01
    assert SEED == 42


def test_select_features_ignores_extra_columns():
    df = load_data().head(3)
    out = SelectFeatures(["ndvi_mean", "moisture_index"]).fit_transform(df)
    assert list(out.columns) == ["ndvi_mean", "moisture_index"]


def test_drop_correlated_keeps_first_of_duplicates():
    df = load_data()
    kept, dropped = drop_correlated(df, FEATURES, 0.9)
    assert "ndvi_mean" in kept
    assert all(v[0] in kept for v in dropped.values())


def test_saved_pipeline_scores_probabilities():
    pipe, meta = load_model()
    df = load_data().head(20)
    p = pipe.predict_proba(df)[:, 1]
    assert p.shape == (20,)
    assert np.all((p >= 0) & (p <= 1))
    assert set(meta["features"]) <= set(FEATURES)
