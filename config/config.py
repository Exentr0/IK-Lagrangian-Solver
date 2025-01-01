from pathlib import Path


class Config:
    PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
    OBJECT_ROTATION_STEPS = 1000
    FPS = 240
    SIM_TIME_STEP = 1.0 / 240.0
