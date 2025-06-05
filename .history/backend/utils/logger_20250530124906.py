import json
import os
from datetime import datetime

def save_training_log(file_id, original_filename, model_type, accuracy, dataset_shape, logs_folder='logs'):
    log_data = {
        "file_id": file_id,
        "original_filename": original_filename,
        "model_type": model_type,
        "accuracy": accuracy,
        "num_rows": dataset_shape[0],
        "num_columns": dataset_shape[1],
        "timestamp": datetime.now().isoformat()
    }
    log_path = os.path.join(logs_folder, f"{file_id}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    return log_path
