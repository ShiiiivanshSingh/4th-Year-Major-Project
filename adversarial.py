import json
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from evaluate import compute_metrics, save_metrics, plot_metrics_comparison

SEED = 42
RESULTS_DIR = Path("results")


def feature_space_attack(X, binary_mask, n_features_to_flip=5, seed=SEED):
    rng = np.random.default_rng(seed)
    X_adv = X.copy()
    binary_indices = np.where(binary_mask)[0]

    if len(binary_indices) == 0:
        raise ValueError("No binary features available for attack")

    n_flip = min(n_features_to_flip, len(binary_indices))
    for i in range(len(X_adv)):
        chosen = rng.choice(binary_indices, size=n_flip, replace=False)
        X_adv.iloc[i, chosen] = 1 - X_adv.iloc[i, chosen]

    return X_adv


def evaluate_under_attack(clf, X_clean, X_adv, y_true, model_name="model"):
    y_clean_pred = clf.predict(X_clean)
    y_adv_pred = clf.predict(X_adv)

    clean_acc = accuracy_score(y_true, y_clean_pred)
    adv_acc = accuracy_score(y_true, y_adv_pred)
    malware_mask = y_true == 1
    attack_success = (
        (y_clean_pred[malware_mask] == 1) & (y_adv_pred[malware_mask] == 0)
    ).mean() if malware_mask.sum() > 0 else 0.0

    return {
        "model": model_name,
        "clean_accuracy": float(clean_acc),
        "adversarial_accuracy": float(adv_acc),
        "accuracy_drop": float(clean_acc - adv_acc),
        "attack_success_rate": float(attack_success),
    }


def adversarial_training(clf_class, X_train, y_train, binary_mask,
                         n_features_to_flip=5, aug_ratio=0.5, seed=SEED):
    n_aug = int(len(X_train) * aug_ratio)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=n_aug, replace=True)
    X_aug_subset = X_train.iloc[idx].copy()
    X_aug = feature_space_attack(X_aug_subset, binary_mask,
                                 n_features_to_flip=n_features_to_flip, seed=seed + 1)
    y_aug = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

    import pandas as pd
    X_combined = pd.concat([X_train, X_aug], ignore_index=True)
    y_combined = pd.concat(
        [y_train.reset_index(drop=True), y_aug.reset_index(drop=True)],
        ignore_index=True,
    )

    robust_clf = clf_class
    robust_clf.fit(X_combined, y_combined)
    return robust_clf


def plot_adversarial_comparison(results_list, save_path=None):
    names = [r["model"] for r in results_list]
    clean_accs = [r["clean_accuracy"] for r in results_list]
    adv_accs = [r["adversarial_accuracy"] for r in results_list]
    asr = [r["attack_success_rate"] for r in results_list]

    x = np.arange(len(names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width, clean_accs, width, label="Clean Acc", color="#4C9BE8")
    b2 = ax1.bar(x, adv_accs, width, label="Adversarial Acc", color="#E85454")
    b3 = ax2.bar(x + width, asr, width, label="Attack Success Rate", color="#F0A500", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Attack Success Rate")
    ax1.set_ylim(0, 1.15)
    ax2.set_ylim(0, 1.15)
    ax1.set_title("Adversarial Robustness Comparison")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


def run_adversarial_experiment(models_dict, splits, artifact,
                               n_flip=5, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    binary_mask = artifact["binary_mask"]

    X_test_adv = feature_space_attack(splits.X_test, binary_mask, n_features_to_flip=n_flip)

    adv_results = []
    for name, clf in models_dict.items():
        res = evaluate_under_attack(clf, splits.X_test, X_test_adv,
                                    splits.y_test.values, model_name=name)
        adv_results.append(res)
        print(f"[{name}]  clean_acc={res['clean_accuracy']:.4f}  "
              f"adv_acc={res['adversarial_accuracy']:.4f}  "
              f"ASR={res['attack_success_rate']:.4f}")

    save_path = results_dir / "adversarial_results.json"
    with open(save_path, "w") as f:
        json.dump(adv_results, f, indent=2)
    print(f"Adversarial results saved to {save_path}")

    plot_adversarial_comparison(adv_results, save_path=results_dir / "adversarial_comparison.png")
    return adv_results, X_test_adv


if __name__ == "__main__":
    from preprocess import prepare
    from sklearn.ensemble import RandomForestClassifier

    splits, artifact = prepare("data/drebin215.csv")
    rf = joblib.load("models/random_forest.joblib")
    robust_rf = adversarial_training(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=SEED, n_jobs=-1),
        splits.X_train, splits.y_train, artifact["binary_mask"],
    )
    joblib.dump(robust_rf, "models/robust_random_forest.joblib")

    models = {"baseline_rf": rf, "robust_rf": robust_rf}
    run_adversarial_experiment(models, splits, artifact)
