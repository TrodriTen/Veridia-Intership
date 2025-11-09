import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
MODELS_DIR = APP_DIR / "models"
DATA_DIR = BASE_DIR / "data"

VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"

TRAIN_DATA_PATH = DATA_DIR / "train_clean_balanced.csv"
TEST_DATA_PATH = DATA_DIR / "test_clean.csv"

RANDOM_STATE = 42
NGRAM_RANGE = (1, 2)
LOWERCASE = False

API_TITLE = "Resume Classifier API"
API_DESCRIPTION = "API for classifying resumes into professional categories using Machine Learning"
API_VERSION = "1.0.0"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

ALLOWED_ORIGINS = ["*"] 

MIN_TEXT_LENGTH = 10
ACCEPTED_FILE_TYPES = ["text/plain", "application/pdf"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

FRONTEND_DIR = BASE_DIR / "frontend"