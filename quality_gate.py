import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

required_files = [
    BASE_DIR / "iris_model.pkl",
    BASE_DIR / "metrics.json",
    BASE_DIR / "iris_dataset.csv",
]

for file_path in required_files:
    if not file_path.is_file():
        raise SystemExit(f"QUALITY GATE FAILED: Missing {file_path.name}")

with open(BASE_DIR / "metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

accuracy = metrics.get("accuracy", 0)

if accuracy < 0.90:
    raise SystemExit(
        f"QUALITY GATE FAILED: Accuracy {accuracy:.4f} is below 0.90"
    )

if metrics.get("training_records", 0) <= 0:
    raise SystemExit("QUALITY GATE FAILED: No training records")

if metrics.get("testing_records", 0) <= 0:
    raise SystemExit("QUALITY GATE FAILED: No testing records")

print(f"QUALITY GATE PASSED: Accuracy = {accuracy:.4f}")
print("All required ML output files are available.")
