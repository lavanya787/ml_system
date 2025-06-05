import os
import uuid
import joblib
import pytz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import spacy
from transformers import pipeline

# Load NLP Models
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
text_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def train_model(data, file_type, original_filename):
    log_path = os.path.join("backend", "logs", f"{uuid.uuid4()}_log.txt")
    failed_date_parsing_log = []

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.join("backend", "models"), exist_ok=True)

    if file_type == 'structured':
        if data.shape[1] < 2:
            return {"error": "Structured data must have at least two columns."}

        y = data.iloc[:, -1]
        X = data.iloc[:, :-1]

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"[INFO] Checking if '{col}' is a date...")
                X[f"{col}_original"] = X[col]
                parsed = pd.to_datetime(X[col], errors='coerce', dayfirst=True, utc=False)

                if parsed.notnull().sum() > len(X) * 0.5:
                    print(f"[INFO] '{col}' identified as datetime.")
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
        model_path = os.path.join("backend", "models", model_filename)
        joblib.dump(model, model_path)

        with open(log_path, "w") as log_file:
            log_file.write(f"Model trained: {model_filename}\n")
            log_file.write(f"Accuracy: {model.score(X_test, y_test)}\n")
            for entry in failed_date_parsing_log:
                log_file.write(f"[WARN] {entry}\n")

        return {"message": "Training complete", "model": model_filename, "log": log_path}

    elif file_type == 'text':
        results = text_classifier(data[:512])  # Optional: Add chunking logic

        with open(log_path, "w") as log_file:
            log_file.write(f"Sentiment: {results}\n")

        return {"message": "NLP model ran inference on uploaded text", "result": results, "log": log_path}

    return {"error": "Unsupported file type"}
