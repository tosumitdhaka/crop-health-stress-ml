"""FastAPI service: POST /predict scores one or more field observations."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_model, predict_records

app = FastAPI(title="Crop stress classifier", version="1.0.0")


class Observation(BaseModel):
    """One spatial observation of processed drone vegetation indices.

    All 13 indices are accepted so the request matches the dataset schema; the
    model uses the subset listed by GET /health. grid_x / grid_y are optional and
    are echoed back so results can be mapped.
    """

    ndvi_mean: float = Field(..., ge=-1, le=1.5)
    ndvi_std: float = Field(..., ge=0)
    ndvi_min: float = Field(..., ge=-1, le=1.5)
    ndvi_max: float = Field(..., ge=-1, le=1.5)
    gndvi: float
    savi: float
    evi: float
    red_edge_1: float
    red_edge_2: float
    nir_reflectance: float
    soil_brightness: float
    canopy_density: float = Field(..., ge=0, le=1)
    moisture_index: float = Field(..., ge=0, le=1)
    grid_x: int | None = None
    grid_y: int | None = None


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    grid_x: int | None
    grid_y: int | None
    stress_probability: float
    predicted_label: str
    threshold: float
    severity: str


class PredictResponse(BaseModel):
    model: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "model": meta["model"], "features": meta["features"], "threshold": meta["threshold"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    _, meta = load_model()
    records = [o.model_dump() for o in req.observations]
    outs = predict_records(records)
    preds = [Prediction(grid_x=r["grid_x"], grid_y=r["grid_y"], **o) for r, o in zip(records, outs)]
    return PredictResponse(model=meta["model"], predictions=preds)
