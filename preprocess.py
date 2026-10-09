import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_CANDIDATES = ["class", "label", "type", "malware", "category"]
MALWARE_TOKENS = {"s", "malware", "malicious", "m", "1", "1.0"}
BENIGN_TOKENS = {"b", "benign", "goodware", "0", "0.0"}


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def find_label_column(df):
    lowered = {c.lower(): c for c in df.columns}
    for name in LABEL_CANDIDATES:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"No label column found. Columns: {list(df.columns)[:10]}...")


def encode_labels(series):
    tokens = series.astype(str).str.strip().str.lower()
    mapping = {t: 1 for t in MALWARE_TOKENS}
    mapping.update({t: 0 for t in BENIGN_TOKENS})
    encoded = tokens.map(mapping)
    if encoded.isna().any():
        unknown = sorted(tokens[encoded.isna()].unique())
        raise ValueError(f"Unrecognised label values: {unknown}")
    return encoded.astype(int)


def clean_features(X):
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    return X.astype(float)


def prepare(path, test_size=0.15, val_size=0.15, seed=SEED):
    df = load_dataset(path)
    label_col = find_label_column(df)
    y = encode_labels(df[label_col])
    X = clean_features(df.drop(columns=[label_col]))

    full = X.copy()
    full["_y"] = y.values
    keep = ~full.duplicated()
    X, y = X[keep], y[keep]

    if y.nunique() < 2:
        raise ValueError("Only one class present after cleaning")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed
    )

    imputer = SimpleImputer(strategy="most_frequent")
    imputer.fit(X_train)
    cols = list(X.columns)

    def apply(frame):
        return pd.DataFrame(imputer.transform(frame), columns=cols, index=frame.index)

    X_train, X_val, X_test = apply(X_train), apply(X_val), apply(X_test)
    binary_mask = X_train.isin([0, 1]).all(axis=0).values

    artifact = {
        "feature_names": cols,
        "imputer": imputer,
        "binary_mask": binary_mask,
        "label_column": label_col,
        "seed": seed,
    }
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test), artifact


def transform_inputs(df, artifact):
    features = artifact["feature_names"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required columns, e.g. {missing[:5]}")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    out = artifact["imputer"].transform(X)
    return pd.DataFrame(out, columns=features, index=df.index)


def save_artifact(artifact, path="models/preprocessor.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path="models/preprocessor.joblib"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess the malware dataset")
    parser.add_argument("--data", default="data/drebin215.csv")
    parser.add_argument("--out", default="models/preprocessor.joblib")
    args = parser.parse_args()

    splits, artifact = prepare(args.data)
    save_artifact(artifact, args.out)

    print(f"features       : {len(artifact['feature_names'])}")
    print(f"binary features: {int(artifact['binary_mask'].sum())}")
    for name in ["train", "val", "test"]:
        y = getattr(splits, f"y_{name}")
        print(f"{name:5s}: n={len(y)}  malware_ratio={y.mean():.3f}")


if __name__ == "__main__":
    main()
