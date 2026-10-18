"""Inference helpers shared by the API and batch scoring."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import META_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return pipe, meta


def severity(prob: float, tiers) -> str:
    name = tiers[0][0]
    for tier_name, lower in tiers:
        if prob >= lower:
            name = tier_name
    return name


def predict_records(records: list[dict]) -> list[dict]:
    pipe, meta = load_model()
    X = pd.DataFrame(records)
    proba = pipe.predict_proba(X)[:, 1]
    thr = meta["threshold"]
    return [
        {
            "stress_probability": round(float(p), 4),
            "predicted_label": "Stressed" if p >= thr else "Healthy",
            "threshold": thr,
            "severity": severity(float(p), meta["severity_tiers"]),
        }
        for p in proba
    ]


if __name__ == "__main__":
    import sys

    from .data import load_data

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    df = load_data().head(n)
    for row, out in zip(df.to_dict(orient="records"), predict_records(df.to_dict(orient="records"))):
        print(row["crop_health_label"], out)
