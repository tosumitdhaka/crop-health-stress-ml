"""Project-wide constants: paths, seed, feature lists."""
from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "crop_multispectral.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "stress_pipeline.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "crop-stress"

TARGET_COL = "crop_health_label"
POSITIVE_LABEL = "Stressed"
GRID_COLS = ["grid_x", "grid_y"]

# Spectral / canopy features fed to the model. grid_x / grid_y are deliberately
# excluded: they are field coordinates, not plant signals, and would not
# transfer to another field. They are used only for spatial aggregation.
FEATURES = [
    "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max", "gndvi", "savi", "evi",
    "red_edge_1", "red_edge_2", "nir_reflectance", "soil_brightness",
    "canopy_density", "moisture_index",
]

TEST_SIZE = 0.2
CV_FOLDS = 5
