import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from preprocess import prepare, save_artifact, load_artifact

SEED = 42
MODELS_DIR = Path("models")


def build_models(seed=SEED):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=seed,
        ),
    }


def train_all(splits, artifact, models_dir=MODELS_DIR, seed=SEED):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(seed)
    val_scores = {}

    for name, clf in models.items():
        clf.fit(splits.X_train, splits.y_train)
        val_pred = clf.predict(splits.X_val)
        score = f1_score(splits.y_val, val_pred)
        val_scores[name] = float(score)
        joblib.dump(clf, models_dir / f"{name}.joblib")
        print(f"[{name}]  val_F1={score:.4f}")

    best_name = max(val_scores, key=val_scores.get)
    print(f"\nBest baseline (val F1): {best_name}")

    summary = {"val_f1": val_scores, "best_model": best_name}
    with open(models_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return models, best_name


def load_model(name, models_dir=MODELS_DIR):
    path = Path(models_dir) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Train baseline malware classifiers")
    parser.add_argument("--data", default="data/drebin215.csv")
    parser.add_argument("--preprocessor", default="models/preprocessor.joblib")
    args = parser.parse_args()

    splits, artifact = prepare(args.data)
    save_artifact(artifact, args.preprocessor)

    train_all(splits, artifact)


if __name__ == "__main__":
    main()
