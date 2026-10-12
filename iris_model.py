import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Iris.csv"
MODEL_PATH = BASE_DIR / "iris_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
DATA_OUTPUT_PATH = BASE_DIR / "iris_dataset.csv"


def train_model():
    # Load the Iris dataset
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    # Find the target column
    target = next(
        (column for column in data.columns
         if column.lower() == "species"),
        None
    )

    if target is None:
        raise ValueError("Dataset must contain a Species column.")

    # Use only numeric feature columns; exclude IDs
    features = data.drop(columns=[target]).select_dtypes(
        include="number"
    )

    features = features.loc[
        :, ~features.columns.str.lower().isin(["id", "unnamed: 0"])
    ]

    if features.shape[1] != 4:
        raise ValueError(
            f"Expected 4 numeric Iris features, found {features.shape[1]}."
        )

    X = features
    y = data[target].astype(str).str.strip()

    if y.nunique() != 3:
        raise ValueError("Expected 3 Iris species classes.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Build and train the ML pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)

    # Evaluate the trained model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": list(X.columns),
        "classes": sorted(y.unique().tolist()),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
    }

    # Save the trained model, metrics, and dataset copy
    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    data.to_csv(DATA_OUTPUT_PATH, index=False)

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {METRICS_PATH.name}")
    print(f"Saved dataset: {DATA_OUTPUT_PATH.name}")


if __name__ == "__main__":
    train_model()
