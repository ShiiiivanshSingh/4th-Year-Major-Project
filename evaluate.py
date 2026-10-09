import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path("results")


def compute_metrics(y_true, y_pred, name="model"):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics


def print_metrics(metrics):
    print(f"\n=== {metrics['model']} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  FPR      : {metrics['false_positive_rate']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")


def save_metrics(metrics_list, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved to {path}")


def plot_confusion_matrix(metrics, save_path=None):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Malware"])
    ax.set_yticklabels(["Benign", "Malware"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {metrics['model']}")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def plot_metrics_comparison(metrics_list, save_path=None):
    keys = ["accuracy", "precision", "recall", "f1", "false_positive_rate"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    x = np.arange(len(labels))
    width = 0.8 / len(metrics_list)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k in keys]
        bars = ax.bar(x + i * width - (len(metrics_list) - 1) * width / 2, vals,
                      width, label=m["model"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def evaluate_models(models_dict, splits, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for name, clf in models_dict.items():
        y_pred = clf.predict(splits.X_test)
        m = compute_metrics(splits.y_test, y_pred, name=name)
        all_metrics.append(m)
        print_metrics(m)
        plot_confusion_matrix(m, save_path=results_dir / f"cm_{name}.png")

    plot_metrics_comparison(all_metrics, save_path=results_dir / "comparison.png")
    save_metrics(all_metrics, results_dir / "metrics.json")
    return all_metrics


if __name__ == "__main__":
    import joblib
    from preprocess import prepare

    splits, artifact = prepare("data/drebin215.csv")
    rf = joblib.load("models/random_forest.joblib")
    lr = joblib.load("models/logistic_regression.joblib")
    evaluate_models({"random_forest": rf, "logistic_regression": lr}, splits)
