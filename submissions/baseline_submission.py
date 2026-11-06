"""CS506 music ratings Kaggle baseline: TF-IDF + Linear SVM.

Files expected alongside this script:
    train.csv: review features for ALL reviews; Score is missing for test reviews.
    test.csv:  id,Score (Score is blank); IDs identify reviews to predict.

Run:
    pip install pandas scikit-learn
    python baseline_submission.py

The script prints a local validation Macro F1, then fits on ALL labeled
reviews and writes baseline_submission.csv. No boosting or neural networks.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Files are resolved relative to this Python file, not the terminal's directory.
FOLDER = Path(__file__).resolve().parent
TRAIN_FILE = FOLDER / "train.csv"
TEST_FILE = FOLDER / "test.csv"
OUTPUT_FILE = FOLDER / "baseline_submission.csv"

RANDOM_STATE = 42


def make_text(df: pd.DataFrame) -> pd.Series:
    """Combine the three available text features, safely handling blanks."""
    summary = df["summary"].fillna("").astype(str)
    review = df["reviewText"].fillna("").astype(str)
    genres = df["genres"].fillna("").astype(str).str.replace(",", " ", regex=False)
    return "summary " + summary + " review " + review + " genres " + genres


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )


def main() -> None:
    train = pd.read_csv(TRAIN_FILE, dtype={"id": str})
    test = pd.read_csv(TEST_FILE, dtype={"id": str})

    needed = {"id", "Score", "summary", "reviewText", "genres"}
    missing = needed - set(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    if "id" not in test.columns:
        raise ValueError("test.csv must contain an 'id' column")

    # In this competition, train.csv includes BOTH labeled and unlabeled reviews.
    # test.csv contains the IDs of reviews whose Score must be predicted.
    train["id"] = train["id"].str.strip()
    test["id"] = test["id"].str.strip()
    if train["id"].isna().any() or train["id"].duplicated().any():
        raise ValueError("train.csv contains missing or duplicate review IDs")
    if test["id"].isna().any() or test["id"].duplicated().any():
        raise ValueError("test.csv contains missing or duplicate review IDs")

    absent = test.loc[~test["id"].isin(train["id"]), "id"]
    if not absent.empty:
        raise ValueError(
            f"{len(absent):,} test IDs are missing from train.csv (e.g. {absent.head().tolist()}). "
            "This script assumes train.csv contains the features for test IDs."
        )

    train["Score"] = pd.to_numeric(train["Score"], errors="coerce")
    test_ids = set(test["id"])

    # Never use test rows for supervised training, even if their Score was filled.
    labeled = train.loc[train["Score"].notna() & ~train["id"].isin(test_ids)].copy()
    if labeled.empty:
        raise ValueError("No labeled reviews available for training")
    if not labeled["Score"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Labeled Score values must be integers between 1 and 5")

    # Use the full-feature rows in train.csv, looked up by test.csv ID/order.
    predict_rows = train.set_index("id").loc[test["id"]].reset_index()
    X = make_text(labeled)
    y = labeled["Score"].astype(int)
    X_test = make_text(predict_rows)

    print(f"Labeled rows: {len(labeled):,} | Test rows: {len(test):,}", flush=True)
    print("Model: TF-IDF (summary + reviewText + genres) + LinearSVC", flush=True)

    # Local evaluation on labeled reviews ONLY, with an 80/20 split.
    counts = y.value_counts()
    val_size = max(1, round(0.2 * len(y)))
    stratify = y if counts.min() >= 2 and val_size >= len(counts) else None
    if stratify is None:
        print("Warning: too few examples for a stratified split; using random split.", flush=True)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    model = build_model()
    print("Training validation model...", flush=True)
    model.fit(X_fit, y_fit)
    validation_predictions = model.predict(X_val)
    macro_f1 = f1_score(
        y_val, validation_predictions, labels=[1, 2, 3, 4, 5],
        average="macro", zero_division=0
    )
    print(f"Local validation Macro F1: {macro_f1:.5f}", flush=True)

    # Refit using every labeled row, then predict the requested test IDs.
    print("Training final model on all labeled reviews...", flush=True)
    model.fit(X, y)
    predictions = model.predict(X_test).astype(int)

    submission = pd.DataFrame({"id": test["id"], "Score": predictions})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(submission):,} predictions to: {OUTPUT_FILE}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
