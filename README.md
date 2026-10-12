# iris-ml-artifacts-4

# Iris ML CI Pipeline

This project implements a machine learning CI pipeline using
the Iris flower dataset.

## Features
- Loads and validates the Iris dataset
- Trains a Logistic Regression classification model
- Evaluates model accuracy
- Runs a machine learning quality gate
- Executes automated unit tests
- Uploads validated ML artifacts through GitHub Actions

## Generated Artifacts
- iris_model.pkl
- metrics.json
- iris_dataset.csv

## Run Locally
Install dependencies:

pip install -r requirements.txt

Train the model:

python iris_model.py

Run the quality gate:

python quality_gate.py

Run tests:

python -m unittest discover -v

## Artifact Retention
GitHub Actions stores the uploaded artifact for 7 days.
