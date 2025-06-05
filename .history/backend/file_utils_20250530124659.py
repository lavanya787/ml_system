import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(df, model_type="random_forest"):
    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy
