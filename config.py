import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
    TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
    VAL_SIZE = float(os.getenv("VAL_SIZE", 0.1))
    
class DataConfig:
    """Data configuration"""
    RAW_DATA_PATH = os.path.join(os.getenv("DATA_PATH", "./data/"), "raw/")
    PROCESSED_DATA_PATH = os.path.join(os.getenv("DATA_PATH", "./data/"), "processed/")
    CROPS = ['wheat', 'rice', 'maize', 'soybean']
    FEATURES = ['temperature', 'rainfall', 'co2', 'humidity']
    
class ModelConfig:
    """Model configuration"""
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/")
    RF_N_ESTIMATORS = 200
    XGB_N_ESTIMATORS = 150
    XGB_MAX_DEPTH = 10
    RF_MAX_DEPTH = 20
