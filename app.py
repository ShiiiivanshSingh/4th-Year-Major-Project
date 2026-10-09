import json
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from preprocess import load_artifact, transform_inputs
from evaluate import compute_metrics, plot_confusion_matrix, plot_metrics_comparison
from adversarial import feature_space_attack, evaluate_under_attack

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def load_models():
    models = {}
    for name in ["random_forest", "logistic_regression"]:
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    robust_p = MODELS_DIR / "robust_random_forest.joblib"
    if robust_p.exists():
        models["robust_random_forest"] = joblib.load(robust_p)
    return models


def load_metrics():
    p = RESULTS_DIR / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def load_adv_results():
    p = RESULTS_DIR / "adversarial_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return []


def sidebar():
    st.sidebar.image(
        "https://img.shields.io/badge/Android-Malware%20Detector-red?style=for-the-badge&logo=android",
        use_column_width=True,
    )
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "",
        ["🏠 Home", "🔍 Predict", "📊 Evaluation", "⚔️ Adversarial Demo", "ℹ️ About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project Guide:** Mr. Kumar Nishant")
    st.sidebar.markdown("**Stack:** Python · scikit-learn · Streamlit")
    return page


def page_home():
    st.title("🛡️ Adversarially Robust Android Malware Detection")
    st.markdown(
        """
        This application demonstrates a machine-learning based Android malware detector
        and shows how **adversarial perturbations** affect model performance.

        ---
        ### What can you do here?
        | Page | Description |
        |------|-------------|
        | 🔍 **Predict** | Upload a CSV of feature vectors and get predictions |
        | 📊 **Evaluation** | View clean-data performance metrics and charts |
        | ⚔️ **Adversarial Demo** | See how feature-space attacks degrade detection |
        | ℹ️ **About** | Research context and limitations |

        ---
        ### Research Questions
        1. How effectively can ML models detect Android malware?
        2. How much does detection degrade under adversarial perturbations?
        3. Can adversarial training improve robustness without harming clean accuracy?
        """
    )

    col1, col2, col3 = st.columns(3)
    metrics = load_metrics()
    if metrics:
        best = max(metrics, key=lambda m: m["f1"])
        col1.metric("Best Model", best["model"].replace("_", " ").title())
        col2.metric("Best F1", f"{best['f1']:.4f}")
        col3.metric("Best Accuracy", f"{best['accuracy']:.4f}")
    else:
        col1.info("Train models first: `python train.py`")


def page_predict():
    st.title("🔍 Malware Prediction")
    st.markdown(
        "Upload a CSV file containing feature vectors. "
        "The file must have the **same feature columns** as the training dataset (label column optional)."
    )

    artifact_path = MODELS_DIR / "preprocessor.joblib"
    if not artifact_path.exists():
        st.error("Preprocessor not found. Run `python train.py` first.")
        return

    artifact = load_artifact(str(artifact_path))
    models = load_models()
    if not models:
        st.error("No trained models found. Run `python train.py` first.")
        return

    uploaded = st.file_uploader("Upload feature CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    label_col = artifact.get("label_column", None)
    y_true = None
    if label_col and label_col in df_raw.columns:
        from preprocess import encode_labels
        try:
            y_true = encode_labels(df_raw[label_col])
        except Exception:
            y_true = None

    try:
        X = transform_inputs(df_raw, artifact)
    except ValueError as e:
        st.error(str(e))
        return

    st.success(f"Loaded {len(X)} samples with {X.shape[1]} features.")

    model_name = st.selectbox("Choose model", list(models.keys()))
    clf = models[model_name]

    preds = clf.predict(X)
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[:, 1]

    results_df = pd.DataFrame({"prediction": preds}, index=range(len(preds)))
    results_df["label"] = results_df["prediction"].map({0: "✅ Benign", 1: "🚨 Malware"})
    if proba is not None:
        results_df["malware_probability"] = np.round(proba, 4)

    st.markdown(f"### Results — {model_name.replace('_', ' ').title()}")
    st.dataframe(results_df, use_container_width=True)

    n_mal = int(preds.sum())
    n_ben = len(preds) - n_mal
    st.markdown(f"**Malware:** {n_mal} &nbsp;&nbsp; **Benign:** {n_ben}")

    if y_true is not None:
        from evaluate import compute_metrics, print_metrics
        m = compute_metrics(y_true.values, preds, name=model_name)
        st.markdown("### Evaluation Against Provided Labels")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']:.4f}")
        col2.metric("Precision", f"{m['precision']:.4f}")
        col3.metric("Recall", f"{m['recall']:.4f}")
        col4.metric("F1", f"{m['f1']:.4f}")

    csv_bytes = results_df.to_csv(index=False).encode()
    st.download_button("Download Predictions CSV", csv_bytes, "predictions.csv", "text/csv")

    st.info(
        "⚠️ **Limitation**: Predictions are based on pre-extracted features only. "
        "APK files are never executed. This detector may miss novel malware families "
        "or be fooled by feature-space evasion. Do not use in production without additional validation."
    )


def page_evaluation():
    st.title("📊 Model Evaluation")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run `python evaluate.py` after training.")
        return

    st.markdown("### Clean-Data Performance on Held-Out Test Set")
    table_data = []
    for m in metrics:
        table_data.append({
            "Model": m["model"].replace("_", " ").title(),
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1": f"{m['f1']:.4f}",
            "FPR": f"{m['false_positive_rate']:.4f}",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    st.markdown("### Confusion Matrices")
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        img_path = RESULTS_DIR / f"cm_{m['model']}.png"
        if img_path.exists():
            col.image(str(img_path), caption=m["model"].replace("_", " ").title())

    comp_path = RESULTS_DIR / "comparison.png"
    if comp_path.exists():
        st.markdown("### Side-by-Side Comparison")
        st.image(str(comp_path), use_column_width=True)


def page_adversarial():
    st.title("⚔️ Adversarial Robustness Demo")
    st.markdown(
        """
        ### Threat Model
        - **Attacker goal**: Evade detection (malware classified as benign).
        - **Attacker knowledge**: Feature vector representation (black-box model access).
        - **Attack method**: Feature-space bit-flip — randomly toggle a fixed number of binary
          permissions/API-call flags from 0→1 or 1→0 on malware samples.
        - **Constraint**: Only binary features are perturbed; values remain 0 or 1.
        - **Limitation**: Feature-space perturbations do not guarantee valid APK modifications.
          These experiments measure classifier fragility in feature space only.
        """
    )

    artifact_path = MODELS_DIR / "preprocessor.joblib"
    if not artifact_path.exists():
        st.error("Run `python train.py` first.")
        return

    artifact = load_artifact(str(artifact_path))
    models = load_models()
    if not models:
        st.error("No trained models found. Run `python train.py` first.")
        return

    st.markdown("### Live Experiment")
    n_flip = st.slider("Number of features to flip per sample", 1, 20, 5)

    if st.button("Run Adversarial Experiment"):
        from preprocess import prepare
        with st.spinner("Preprocessing and attacking..."):
            splits, _ = prepare("data/drebin215.csv")
            X_adv = feature_space_attack(splits.X_test, artifact["binary_mask"],
                                         n_features_to_flip=n_flip)

        results = []
        for name, clf in models.items():
            res = evaluate_under_attack(clf, splits.X_test, X_adv,
                                        splits.y_test.values, model_name=name)
            results.append(res)

        table_data = []
        for r in results:
            table_data.append({
                "Model": r["model"].replace("_", " ").title(),
                "Clean Acc": f"{r['clean_accuracy']:.4f}",
                "Adversarial Acc": f"{r['adversarial_accuracy']:.4f}",
                "Acc Drop": f"{r['accuracy_drop']:.4f}",
                "Attack Success Rate": f"{r['attack_success_rate']:.4f}",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        names = [r["model"].replace("_", " ").title() for r in results]
        x = np.arange(len(names))
        w = 0.3
        ax.bar(x - w / 2, [r["clean_accuracy"] for r in results], w, label="Clean Acc", color="#4C9BE8")
        ax.bar(x + w / 2, [r["adversarial_accuracy"] for r in results], w, label="Adv Acc", color="#E85454")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.set_title(f"Accuracy Drop After {n_flip}-Feature Flip Attack")
        st.pyplot(fig)
        plt.close()

    adv_results = load_adv_results()
    if adv_results:
        st.markdown("### Pre-computed Adversarial Results")
        adv_img = RESULTS_DIR / "adversarial_comparison.png"
        if adv_img.exists():
            st.image(str(adv_img), use_column_width=True)


def page_about():
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ## Adversarially Robust Android Malware Detection

        **Final-year B.Tech CSE Major Project**  
        **Project Guide:** Mr. Kumar Nishant

        ---
        ### Dataset
        **Drebin-215** (Arp et al., NDSS 2014): ~15,000 Android apps, ~215 binary features
        (permissions, API calls, intents). Standard benchmark for Android malware classification.

        ---
        ### Literature Review (Key References)
        1. Arp et al. *"Drebin: Effective and Explainable Detection of Android Malware in Your Pocket."*
           NDSS 2014.
        2. Grosse et al. *"Adversarial Examples for Malware Detection."* ESORICS 2017.
        3. Pierazzi et al. *"Intriguing Properties of Adversarial ML Attacks in the Problem Space."*
           IEEE S&P 2020.
        4. Madry et al. *"Towards Deep Learning Models Resistant to Adversarial Attacks."* ICLR 2018.

        ---
        ### Research Questions
        1. How effectively can ML detect Android malware?
        2. How much does detection degrade under feature-space adversarial perturbations?
        3. Can adversarial training restore robustness without harming clean-data accuracy?

        ---
        ### Limitations
        - Feature-space attacks ≠ real APK modifications. Results reflect classifier fragility, not
          operational attack feasibility.
        - Models are trained on Drebin features. Unseen malware families may evade detection.
        - No deep learning or GPU required; may underperform neural-network baselines on large datasets.
        """
    )


def main():
    st.set_page_config(
        page_title="Android Malware Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        body { font-family: 'Inter', sans-serif; }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 8px;
            padding: 0.5rem 1.5rem; font-weight: 600;
        }
        .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
        .metric-container { background: #1e1e2e; border-radius: 12px; padding: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = sidebar()
    if page == "🏠 Home":
        page_home()
    elif page == "🔍 Predict":
        page_predict()
    elif page == "📊 Evaluation":
        page_evaluation()
    elif page == "⚔️ Adversarial Demo":
        page_adversarial()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
