import joblib
import os
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import spacy
from transformers import pipeline
import pytz
import dateutil.parser

# Load NLP models
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def is_date_like(s):
    try:
        # Try parsing with dateutil.parser
        dateutil.parser.parse(s)
        return True
    except Exception:
        return False

def train_model(data, file_type, original_filename):
    log_path = f"backend/logs/{uuid.uuid4()}_log.txt"
    failed_date_parsing_log = []

    if file_type == 'structured':
        if data.shape[1] < 2:
            return {"error": "Structured data must have at least two columns."}

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Preprocess features
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"[INFO] Checking if '{col}' is a date...")

                # Save original values for reference
                X[f"{col}_original"] = X[col]

                # Sample some values (up to 10) to check if column is date-like
                sample_values = X[col].dropna().astype(str).sample(min(10, len(X[col])))

                date_like_count = sum(is_date_like(v) for v in sample_values)

                if date_like_count >= len(sample_values) / 2:
                    print(f"[INFO] '{col}' identified as datetime with possible timezones.")

                    parsed = pd.to_datetime(X[col], errors='coerce', dayfirst=True, utc=False)
                    timezones = []
                    parsed_utc = []

                    for val in parsed:
                        if pd.isnull(val):
                            parsed_utc.append(pd.NaT)
                            timezones.append("Unknown")
                        elif val.tzinfo is not None:
                            parsed_utc.append(val.astimezone(pytz.UTC))
                            timezones.append(str(val.tzinfo))
                        else:
                            local_tz = pytz.timezone("Asia/Kolkata")
                            localized = local_tz.localize(val)
                            parsed_utc.append(localized.astimezone(pytz.UTC))
                            timezones.append(str(local_tz))

                    parsed_utc = pd.Series(parsed_utc)
                    X[f"{col}_year"] = parsed_utc.dt.year.fillna(0).astype(int)
                    X[f"{col}_month"] = parsed_utc.dt.month.fillna(0).astype(int)
                    X[f"{col}_day"] = parsed_utc.dt.day.fillna(0).astype(int)
                    X[f"{col}_tz"] = timezones
                    X.drop(columns=[col], inplace=True)

                    failed_indices = parsed[parsed.isna()].index.tolist()
                    if failed_indices:
                        failed_date_parsing_log.append(
                            f"Column '{col}': Could not parse {len(failed_indices)} rows. Indices: {failed_indices[:5]}"
                        )
                else:
                    print(f"[INFO] '{col}' not detected as datetime. Encoding as categorical.")
                    X[col] = X[col].fillna("Unknown")
                    X[col] = LabelEncoder().fit_transform(X[col])

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        model_filename = f"{uuid.uuid4()}_model.pkl"
        model_path = f"backend/models/{model_filename}"
        joblib.dump(model, model_path)

        with open(log_path, "w") as log_file:
            log_file.write(f"Model trained: {model_filename}\n")
            log_file.write(f"Accuracy: {model.score(X_test, y_test)}\n")
            for entry in failed_date_parsing_log:
                log_file.write(f"[WARN] {entry}\n")

        return {"message": "Training complete", "model": model_filename, "log": log_path}

    elif file_type == 'text':
        clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        results = clf(data[:512])  # Truncate long input

        with open(log_path, "w") as log_file:
            log_file.write(f"Sentiment: {results}\n")

        return {"message": "NLP model ran inference on uploaded text", "result": results, "log": log_path}

    return {"error": "Unsupported file type"}
