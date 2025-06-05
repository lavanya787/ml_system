import pandas as pd
import joblib
import os
import time
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uuid

def train_model(data, file_type, original_filename):
    log_path = f"backend/logs/{uuid.uuid4()}_log.txt"

    if file_type == 'structured':
        if data.shape[1] < 2:
            return {"error": "Structured data must have at least two columns."}

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        model_filename = f"{uuid.uuid4()}_model.pkl"
        joblib.dump(model, f"backend/models/{model_filename}")

        with open(log_path, "w") as log_file:
            log_file.write(f"Model trained: {model_filename}\nAccuracy: {model.score(X_test, y_test)}")

        return {"message": "Training complete", "model": model_filename, "log": log_path}

    elif file_type == 'text':
        clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        results = clf(data[:512])  # limit input for demo purposes

        with open(log_path, "w") as log_file:
            log_file.write(f"Sentiment: {results}")

        return {"message": "NLP model ran inference on uploaded text", "result": results, "log": log_path}

    return {"error": "Unsupported file type"}
