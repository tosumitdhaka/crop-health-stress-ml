from fastapi.testclient import TestClient

from app.main import app
from src.data import load_data

client = TestClient(app)

FIELDS = ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "gndvi", "savi", "evi", "red_edge_1",
          "red_edge_2", "nir_reflectance", "soil_brightness", "canopy_density", "moisture_index",
          "grid_x", "grid_y"]


def _rows(labels):
    df = load_data()
    out = []
    for lab in labels:
        out.append(df[df.crop_health_label == lab].iloc[0][FIELDS].to_dict())
    return out


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] < 1


def test_predict_known_rows():
    rows = _rows(["Healthy", "Stressed"])
    r = client.post("/predict", json={"observations": rows})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["predicted_label"] == "Healthy"
    assert preds[1]["predicted_label"] == "Stressed"
    for p in preds:
        assert 0.0 <= p["stress_probability"] <= 1.0
        assert p["severity"] in {"Low", "Moderate", "High", "Critical"}


def test_predict_rejects_missing_field():
    row = _rows(["Healthy"])[0]
    row.pop("ndvi_mean")
    r = client.post("/predict", json={"observations": [row]})
    assert r.status_code == 422


def test_predict_rejects_empty_list():
    r = client.post("/predict", json={"observations": []})
    assert r.status_code == 422
