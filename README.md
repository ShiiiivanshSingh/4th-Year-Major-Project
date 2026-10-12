

<div align="center">

# Adversarially Robust Android Malware Detection

</div>
<div align="center">
  
**Final-Year B.Tech CSE Major Project**  
**Guide:** Mr. Kumar Nishant


  <a href="https://docs.google.com/presentation/d/1Z5WuP13Uru8pjTM_ru-o6DLq4rsTjbP87kIcZGMPbyA/edit?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/View_Project_Presentation-4285F4?style=for-the-badge&logo=google-slides&logoColor=white" alt="View Project Presentation" />
  </a>
  &nbsp;
  <a href="https://android-malware-defense.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live_Web_App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Web App on Streamlit" />
  </a>

</div>


## Overview

A complete machine-learning pipeline that:
1. Detects Android malware from feature vectors (Drebin-215 dataset)
2. Demonstrates how adversarial perturbations degrade detection
3. Compares a baseline model against an adversarially trained robust model
4. Provides an interactive Streamlit web application




## Screenshots

### 1. Home / Overview

<div align="center">
  <img src="data/screenshots/1.png" width="800" alt="Home - Overview" />
</div>

### 2. Single App Prediction

<div align="center">
  <img src="data/screenshots/2.png" width="800" alt="Predict - Analysis" />
</div>

### 3. Model Evaluation

<div align="center">
  <img src="data/screenshots/3.png" width="800" alt="Evaluation - Confusion Matrix" />
</div>

### 4. Adversarial Attack Demo

<div align="center">
  <img src="data/screenshots/4.png" width="800" alt="Adversarial Attack Simulator" />
</div>

### 5. Project Architecture & About

<div align="center">
  <img src="data/screenshots/5.png" width="800" alt="About & Architecture" />
</div>


## Project Structure

```
android-malware-detection/
├── app.py              # Streamlit web application (5 pages)
├── train.py            # Train Random Forest + Logistic Regression
├── evaluate.py         # Metrics, confusion matrices, comparison plots
├── adversarial.py      # Feature-space attacks + adversarial training
├── preprocess.py       # Dataset loading, cleaning, splitting, artifacts
├── requirements.txt
├── README.md
├── data/               # Place drebin215.csv here
├── models/             # Saved .joblib model + preprocessor files
├── results/            # Metrics JSON + PNG plots
├── notebooks/          # Jupyter exploration notebooks
├── tests/              # pytest unit tests
└── report/             # Project report and slides
```

## Setup

```bash
cd "4th yr prtoj"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download **Drebin-215** from Kaggle or Mendeley and save as:

```
data/drebin215.csv
```

The CSV must have ~215 binary feature columns and a `class` column with values `S` (malware) or `B` (benign).

**Fallback:** TUANDROMD from UCI — `preprocess.py` auto-detects the label column.

## Running the Pipeline

### Step 1 — Preprocess and Train

```bash
python train.py --data data/drebin215.csv
```

This runs preprocessing, trains both models, and saves everything to `models/`.

### Step 2 — Evaluate

```bash
python evaluate.py
```

Outputs metrics to `results/metrics.json` and saves plots to `results/`.

### Step 3 — Adversarial Experiment

```bash
python adversarial.py
```

Runs feature-space attack, trains a robust model, and saves comparison plots.

### Step 4 — Launch Web App

**🌐 Live Cloud App:**
> 👉 **[Open Live Streamlit App](https://android-malware-defense.streamlit.app/)**

**Or Run Locally:**
```bash
cd "your directory"
source .venv/bin/activate
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Step 5 — Run Tests

```bash
python -m pytest tests -q
```


## Research Questions

1. How effectively can ML detect Android malware?
2. How much does performance degrade under feature-space adversarial perturbations?
3. Can adversarial training improve robustness without harming clean accuracy?

## Threat Model

- **Attack surface**: Feature vector representation (binary permissions / API-call flags)
- **Method**: Bit-flip attack — randomly toggle N binary features on malware samples
- **Goal**: Cause malware to be classified as benign (evasion attack)
- **Constraint**: Only binary (0/1) features are perturbed
- **Limitation**: Results reflect classifier fragility in feature space; not validated at APK level

## Key Design Decisions (Viva Notes)

| Decision | Rationale |
|----------|-----------|
| Duplicate rows removed before splitting | Prevents identical apps appearing in both train and test |
| Imputer fitted on train split only | Prevents data leakage |
| class_weight="balanced" | Handles class imbalance without resampling test data |
| Validation F1 used for model selection | More informative than accuracy on imbalanced data |
| Feature-space attack on binary features only | Preserves feature validity constraints |

## References

1. Arp et al. "Drebin: Effective and Explainable Detection of Android Malware in Your Pocket." NDSS 2014.
2. Grosse et al. "Adversarial Examples for Malware Detection." ESORICS 2017.
3. Pierazzi et al. "Intriguing Properties of Adversarial ML Attacks in the Problem Space." IEEE S&P 2020.
4. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.



---
title: Android Malware Detection
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---
