# backend/tasks.py

import os
import joblib
from datetime import datetime
from backend.file_utils import get_file_extension
from backend.config import LOG_FOLDER, MODEL_FOLDER

# IMPORT YOUR EXISTING ML CODE HERE
from utils.model_handler import ModelHandler2 as ml1
# backend/tasks.py

import os
import joblib
from datetime import datetime
from backend.file_utils import get_file_extension
from backend.config import LOG_FOLDER, MODEL_FOLDER

# IMPORT YOUR EXISTING ML CODE HERE
import ml_system.ml_code1 as ml1
import ml_system.ml_code2 as ml2

def train_model(filepath, model_type):
    ext = get_file_extension(filepath)

    # Match file extension to ML training logic
    if ext == '.csv':
        model = ml1.train_from_csv(filepath)
    elif ext in ['.txt', '.docx', '.pdf']:
        model = ml2.train_from_text(filepath)
    else:
        raise ValueError("Unsupported file type")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODEL_FOLDER, f"{model_type}_{timestamp}.pkl")
    joblib.dump(model, model_path)

    # Save training log
    log_path = os.path.join(LOG_FOLDER, f"{model_type}_{timestamp}.log")
    with open(log_path, 'w') as log:
        log.write(f"Trained {model_type} model on {filepath} at {timestamp}\n")

    return {
        "model_path": model_path,
        "log_path": log_path,
        "status": "success"
    }


def train_model(filepath, model_type):
    ext = get_file_extension(filepath)

    # Match file extension to ML training logic
    if ext == '.csv':
        model = ml1.train_from_csv(filepath)
    elif ext in ['.txt', '.docx', '.pdf']:
        model = ml2.train_from_text(filepath)
    else:
        raise ValueError("Unsupported file type")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(MODEL_FOLDER, f"{model_type}_{timestamp}.pkl")
    joblib.dump(model, model_path)

    # Save training log
    log_path = os.path.join(LOG_FOLDER, f"{model_type}_{timestamp}.log")
    with open(log_path, 'w') as log:
        log.write(f"Trained {model_type} model on {filepath} at {timestamp}\n")

    return {
        "model_path": model_path,
        "log_path": log_path,
        "status": "success"
    }
