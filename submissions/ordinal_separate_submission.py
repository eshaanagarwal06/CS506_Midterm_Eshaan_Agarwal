"""Kaggle music ratings — separate-field ordinal Logistic Regression (C=4).

Run:
    python ordinal_separate_submission.py

Input:
    train.csv and test.csv beside this file, or in a data/ subfolder.
    As in the original competition files, train.csv contains all features for
    labeled AND unlabeled rows; test.csv lists the 80,000 test IDs in order.

Output:
    submissions/submission_ordinal_separate_C4_thresholds.csv

No previously fitted models, research caches, precomputed offsets, or research
predictions are loaded. The final four classifiers fit ALL labeled rows.

Pipeline reproduced from the separate_balanced_C4_thresholds experiment:
* Three independent word/bigram TF-IDF blocks: 20k summary, 90k review, 3k genres.
* Text block weights (1.0, 1.0, 0.5), L2-normalized as a group.
* 41 engineered metadata features with 4-fold OOF target encodings.
* Four independent logistic regressions, C=4, predict P(Score > 1..4).
* Nonincreasing cumulative probabilities and four nondecreasing thresholds.
* Thresholds tuned on held-out labeled reviews; fresh final training on ALL labels.

Requirements: numpy, pandas, scipy, scikit-learn.
No neural networks or boosting. No Kaggle upload is performed.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "submissions"
OUTPUT_FILE = "submission_ordinal_separate_C4_thresholds.csv"
SEED = 42
SCORES = np.arange(1, 6)
ID_COLUMNS = ("reviewerID", "album_mbid", "artist_mbid")
OOF_FOLDS = 4
GROUP_SMOOTHING = {"reviewerID": 12.0, "album_mbid": 15.0, "artist_mbid": 20.0}
GENRE_SMOOTHING = 20.0
USER_GENRE_SMOOTHING = 8.0
LOGISTIC_C = 4.0


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def locate_file(filename: str) -> Path:
    for path in (ROOT / filename, ROOT / "data" / filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {filename}; place it beside the script or in data/.")


def parse_genres(value: str) -> list[str]:
    return list(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(locate_file("train.csv"), dtype={"id": str}, low_memory=False)
    test = pd.read_csv(locate_file("test.csv"), dtype={"id": str}, low_memory=False)
    required = set(ID_COLUMNS) | {"id", "Score", "genres", "summary", "reviewText"}
    if required - set(train.columns):
        raise ValueError(f"Missing columns in train.csv: {sorted(required - set(train.columns))}")
    if "id" not in test.columns:
        raise ValueError("test.csv must contain an id column")

    for frame, name in ((train, "train"), (test, "test")):
        frame["id"] = frame["id"].str.strip()
        if frame["id"].isna().any() or (frame["id"] == "").any():
            raise ValueError(f"{name}.csv contains blank IDs")
        if frame["id"].duplicated().any():
            raise ValueError(f"{name}.csv contains duplicate IDs")
    if not test["id"].isin(train["id"]).all():
        raise ValueError("test.csv has IDs missing from train.csv")

    train["Score"] = pd.to_numeric(train["Score"], errors="coerce")
    test_ids = set(test["id"])
    labeled = train.loc[train["Score"].notna() & ~train["id"].isin(test_ids)].copy()
    if labeled.empty or not labeled["Score"].isin(SCORES).all():
        raise ValueError("Training scores must be integers from 1 to 5")
    labeled["Score"] = labeled["Score"].astype(np.int8)
    predict = train.set_index("id").loc[test["id"]].reset_index()

    for frame in (labeled, predict):
        for key in ID_COLUMNS:
            frame[key] = frame[key].fillna("").astype(str).str.strip()
        for key in ("summary", "reviewText", "genres"):
            frame[key] = frame[key].fillna("").astype(str)
        frame["genre_tokens"] = frame["genres"].map(parse_genres)

    labeled = labeled.reset_index(drop=True)
    predict = predict.reset_index(drop=True)
    if len(labeled["Score"].unique()) != 5:
        raise ValueError("All five rating classes must be present in training data")
    return labeled, predict, test["id"].copy()


def static_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(frame)))
    summary = frame["summary"]
    review = frame["reviewText"]
    result["summary_log_chars"] = np.log1p(summary.str.len().to_numpy())
    result["review_log_chars"] = np.log1p(review.str.len().to_numpy())
    result["summary_log_words"] = np.log1p(summary.str.count(r"\S+").to_numpy())
    result["review_log_words"] = np.log1p(review.str.count(r"\S+").to_numpy())
    result["review_exclamations"] = np.log1p(review.str.count("!").to_numpy())
    result["review_questions"] = np.log1p(review.str.count(r"\?").to_numpy())
    result["genre_count"] = frame["genre_tokens"].map(len).to_numpy()

    total = pd.to_numeric(frame.get("TotalVotes", pd.Series(0, index=frame.index)), errors="coerce")
    helpful = pd.to_numeric(frame.get("VotedHelpful", pd.Series(0, index=frame.index)), errors="coerce")
    result["has_vote_data"] = (total.notna() & helpful.notna()).to_numpy(dtype=np.float32)
    votes = total.fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
    helpful_votes = helpful.fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
    result["log_votes"] = np.log1p(votes)
    result["helpful_ratio"] = (np.minimum(helpful_votes, votes) + 1) / (votes + 2)

    timestamp = pd.to_numeric(
        frame.get("unixReviewTime", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    dates = pd.to_datetime(timestamp, unit="s", errors="coerce", utc=True)
    result["review_year"] = dates.dt.year.fillna(2000).to_numpy() - 2000
    result["review_month"] = dates.dt.month.fillna(6).to_numpy()
    result["missing_review_time"] = dates.isna().to_numpy(dtype=np.float32)
    return result.astype(np.float32)


def rating_group_features(
    reference: pd.DataFrame, query: pd.DataFrame, global_mean: float, global_probs: np.ndarray
) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(query)))
    for key in ID_COLUMNS:
        label = {"reviewerID": "reviewer", "album_mbid": "album", "artist_mbid": "artist"}[key]
        valid_ref = reference.loc[reference[key] != ""]
        aggregate = valid_ref.groupby(key, sort=False)["Score"].agg(["count", "sum", "std"])
        counts = query[key].map(aggregate["count"]).fillna(0).to_numpy(dtype=np.float64)
        sums = query[key].map(aggregate["sum"]).fillna(0).to_numpy(dtype=np.float64)
        variation = query[key].map(aggregate["std"]).fillna(0).to_numpy(dtype=np.float64)
        smooth = GROUP_SMOOTHING[key]
        output[f"{label}_log_count"] = np.log1p(counts)
        output[f"{label}_mean"] = (sums + smooth * global_mean) / (counts + smooth)
        output[f"{label}_std"] = variation * np.sqrt(counts / (counts + smooth))
        grouped = valid_ref.groupby([key, "Score"]).size().unstack(fill_value=0)
        for score in SCORES:
            column = grouped[score] if score in grouped else pd.Series(dtype=np.int64)
            class_counts = query[key].map(column).fillna(0).to_numpy(dtype=np.float64)
            output[f"{label}_p{score}"] = (
                class_counts + smooth * global_probs[score - 1]
            ) / (counts + smooth)
    return output


def genre_preference_features(
    reference: pd.DataFrame, query: pd.DataFrame, global_mean: float, reviewer_mean: pd.Series
) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(query)))
    reference_genres = reference[["reviewerID", "Score", "genre_tokens"]].explode("genre_tokens")
    reference_genres = reference_genres.loc[
        reference_genres["genre_tokens"].notna() & (reference_genres["genre_tokens"] != "")
    ].copy()
    reference_genres["reviewer_prior"] = (
        reference_genres["reviewerID"].map(reviewer_mean).fillna(global_mean)
    )
    reference_genres["relative_score"] = (
        reference_genres["Score"] - reference_genres["reviewer_prior"]
    )
    genre_agg = reference_genres.groupby("genre_tokens", sort=False)["Score"].agg(["sum", "count"])
    genre_mean = (genre_agg["sum"] + GENRE_SMOOTHING * global_mean) / (
        genre_agg["count"] + GENRE_SMOOTHING
    )
    profile = reference_genres.loc[reference_genres["reviewerID"] != ""].groupby(
        ["reviewerID", "genre_tokens"], sort=False
    )["relative_score"].agg(["sum", "count"])
    profile_effect = profile["sum"] / (profile["count"] + USER_GENRE_SMOOTHING)

    expanded = query[["reviewerID", "genre_tokens"]].copy()
    expanded["row_number"] = np.arange(len(query))
    expanded = expanded.explode("genre_tokens").reset_index(drop=True)
    expanded = expanded.loc[
        expanded["genre_tokens"].notna() & (expanded["genre_tokens"] != "")
    ].copy()
    if expanded.empty:
        output["genre_prior"] = global_mean
        output["genre_log_support"] = 0.0
        output["reviewer_genre_preference"] = 0.0
        output["reviewer_genre_log_support"] = 0.0
        return output

    expanded["genre_prior"] = expanded["genre_tokens"].map(genre_mean).fillna(global_mean)
    expanded["genre_support"] = expanded["genre_tokens"].map(genre_agg["count"]).fillna(0)
    pairs = pd.MultiIndex.from_frame(expanded[["reviewerID", "genre_tokens"]])
    expanded["profile"] = profile_effect.reindex(pairs).astype(float).to_numpy()
    expanded["profile_support"] = profile["count"].reindex(pairs).astype(float).to_numpy()
    expanded["profile"] = expanded["profile"].fillna(0)
    expanded["profile_support"] = expanded["profile_support"].fillna(0)
    aggregated = expanded.groupby("row_number", sort=False).agg(
        genre_prior=("genre_prior", "mean"),
        genre_support=("genre_support", "mean"),
        reviewer_genre_preference=("profile", "mean"),
        reviewer_genre_support=("profile_support", "sum"),
    )
    row_index = np.arange(len(query))
    output["genre_prior"] = aggregated["genre_prior"].reindex(row_index).fillna(global_mean).to_numpy()
    output["genre_log_support"] = np.log1p(
        aggregated["genre_support"].reindex(row_index).fillna(0).to_numpy()
    )
    output["reviewer_genre_preference"] = (
        aggregated["reviewer_genre_preference"].reindex(row_index).fillna(0).to_numpy()
    )
    output["reviewer_genre_log_support"] = np.log1p(
        aggregated["reviewer_genre_support"].reindex(row_index).fillna(0).to_numpy()
    )
    return output


def make_target_features(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(reference["Score"].mean())
    global_probs = np.bincount(reference["Score"].to_numpy(dtype=np.int64), minlength=6)[1:6]
    global_probs = global_probs / global_probs.sum()
    output = rating_group_features(reference, query, global_mean, global_probs)
    reviewer_agg = reference.loc[reference["reviewerID"] != ""].groupby("reviewerID")["Score"].agg(
        ["sum", "count"]
    )
    reviewer_mean = (
        reviewer_agg["sum"] + GROUP_SMOOTHING["reviewerID"] * global_mean
    ) / (reviewer_agg["count"] + GROUP_SMOOTHING["reviewerID"])
    genre = genre_preference_features(reference, query, global_mean, reviewer_mean)
    return pd.concat([output, genre], axis=1).astype(np.float32)


def build_metadata(labeled: pd.DataFrame, test: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    labels = labeled["Score"].to_numpy()
    folds = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=SEED + 1)
    oof = None
    feature_names = None

    for n, (train_idx, hold_idx) in enumerate(folds.split(np.zeros(len(labeled)), labels), 1):
        block = make_target_features(labeled.iloc[train_idx], labeled.iloc[hold_idx])
        if oof is None:
            feature_names = list(block.columns)
            oof = np.empty((len(labeled), len(feature_names)), dtype=np.float32)
        if list(block.columns) != feature_names:
            raise ValueError("Target feature columns changed between folds")
        oof[hold_idx] = block.to_numpy(dtype=np.float32)
        log(f"Target-feature fold {n}/{OOF_FOLDS} complete")

    test_target = make_target_features(labeled, test)[feature_names].to_numpy(dtype=np.float32)
    static_train = static_features(labeled)
    static_test = static_features(test)
    x_train = np.hstack((oof, static_train.to_numpy(dtype=np.float32)))
    x_test = np.hstack((test_target, static_test.to_numpy(dtype=np.float32)))
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Non-finite engineered features")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    scale = np.sqrt(x_train.shape[1])
    x_train = np.clip(x_train, -6, 6) / scale
    x_test = np.clip(x_test, -6, 6) / scale
    log(f"Engineered features: {x_train.shape[1]} columns")
    return sparse.csr_matrix(x_train), sparse.csr_matrix(x_test)



FIELD_SPECS = {
    "summary": dict(max_features=20_000, ngram_range=(1, 2), min_df=2),
    "review": dict(max_features=90_000, ngram_range=(1, 2), min_df=2),
    "genres": dict(max_features=3_000, ngram_range=(1, 2), min_df=2),
}
FIELD_WEIGHTS = (1.0, 1.0, 0.5)


def text_content(frame: pd.DataFrame, part: str) -> pd.Series:
    if part == "summary":
        return frame["summary"]
    if part == "review":
        return frame["reviewText"]
    if part == "genres":
        return frame["genres"].str.replace(",", " ", regex=False)
    raise ValueError(f"Unknown text field: {part}")


def prepare_hybrid(fit: pd.DataFrame, predict: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit ALL feature extractors from scratch on fit only, transform predict."""
    log(f"Building 41 engineered features (training rows={len(fit):,})...")
    m_fit, m_predict = build_metadata(fit, predict)

    weights = np.asarray(FIELD_WEIGHTS, dtype=np.float64)
    weights /= np.linalg.norm(weights)
    train_blocks = []
    predict_blocks = []
    for i, field in enumerate(("summary", "review", "genres")):
        log(f"Fitting {field} TF-IDF (max_features={FIELD_SPECS[field]['max_features']:,})...")
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
            **FIELD_SPECS[field],
        )
        fit_text = text_content(fit, field)
        predict_text = text_content(predict, field)
        x_train = vectorizer.fit_transform(fit_text).tocsr()
        x_predict = vectorizer.transform(predict_text).tocsr()
        if x_train.shape[1] == 0:
            raise ValueError(f"No TF-IDF vocabulary was learned for {field}")
        log(f"{field}: {x_train.shape[1]:,} learned columns, block weight={weights[i]:.3f}")
        train_blocks.append(x_train.multiply(weights[i]).tocsr())
        predict_blocks.append(x_predict.multiply(weights[i]).tocsr())
        del vectorizer, fit_text, predict_text, x_train, x_predict
        gc.collect()

    train_blocks.append(m_fit)
    predict_blocks.append(m_predict)
    x_fit = sparse.hstack(train_blocks, format="csr", dtype=np.float32)
    x_predict = sparse.hstack(predict_blocks, format="csr", dtype=np.float32)
    del train_blocks, predict_blocks, m_fit, m_predict
    gc.collect()
    if x_fit.shape[1] != x_predict.shape[1]:
        raise ValueError("Train and test feature dimensions do not match")
    log(f"Hybrid matrix: {x_fit.shape[0]:,} x {x_fit.shape[1]:,}; test rows={x_predict.shape[0]:,}")
    return x_fit, x_predict


def train_four_thresholds(
    x_fit: sparse.csr_matrix, y_fit: np.ndarray, x_predict: sparse.csr_matrix
) -> np.ndarray:
    """Fit 4 NEW classifiers on these labeled rows, predict cumulative probabilities."""
    cumulative = np.empty((x_predict.shape[0], 4), dtype=np.float32)
    for i, threshold in enumerate(range(1, 5)):
        log(f"Training C={LOGISTIC_C:g}: P(Score > {threshold}) ({i + 1}/4)...")
        target = (y_fit > threshold).astype(np.int8)
        if len(np.unique(target)) != 2:
            raise ValueError(f"Insufficient training classes for cutoff {threshold}")
        classifier = LogisticRegression(
            C=LOGISTIC_C,
            solver="liblinear",
            tol=1e-4,
            max_iter=350,
            class_weight=None,
            random_state=SEED,
        )
        classifier.fit(x_fit, target)
        cumulative[:, i] = classifier.predict_proba(x_predict)[:, 1].astype(np.float32)
        log(f"Completed cutoff {threshold}, iterations={int(classifier.n_iter_[0])}")
        del classifier, target
    return cumulative


def project_cumulative(cumulative: np.ndarray) -> np.ndarray:
    """Same cumulative-minimum consistency correction as research."""
    if cumulative.ndim != 2 or cumulative.shape[1] != 4:
        raise ValueError("Expected 4 cumulative probability columns")
    result = np.minimum.accumulate(np.clip(cumulative, 0.0, 1.0), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite cumulative probabilities")
    return result.astype(np.float32)


def threshold_predictions(cumulative: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Coherent ordinal decision: rating=1 + number of passed thresholds."""
    return (1 + (cumulative > thresholds[None, :]).sum(axis=1)).astype(np.int8)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fixed five-class macro F1 (including classes absent in a prediction)."""
    a = np.asarray(y_true, dtype=np.int32) - 1
    b = np.asarray(y_pred, dtype=np.int32) - 1
    confusion = np.bincount(a * 5 + b, minlength=25).reshape(5, 5)
    denom = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(2 * confusion.diagonal(), denom,
                         out=np.zeros(5, dtype=np.float64), where=denom > 0)
    return float(class_f1.mean())


def tune_ordinal_thresholds(cumulative: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Exactly the grid/coordinate-search procedure from ordinal_next_research.py."""
    grid = np.linspace(.25, .75, 21, dtype=np.float32)
    thresholds = np.full(4, .5, dtype=np.float32)
    best = macro_f1(labels, threshold_predictions(cumulative, thresholds))
    for _ in range(3):
        improved = False
        for i in range(4):
            candidates = grid[(grid >= (thresholds[i - 1] if i else 0)) &
                              (grid <= (thresholds[i + 1] if i < 3 else 1))]
            for candidate in candidates:
                proposed = thresholds.copy()
                proposed[i] = candidate
                value = macro_f1(labels, threshold_predictions(cumulative, proposed))
                if value > best + 1e-7:
                    best, thresholds, improved = value, proposed, True
        if not improved:
            break
    if np.any(np.diff(thresholds) < 0):
        raise AssertionError("Tuned decision thresholds are not nondecreasing")
    return thresholds, best


def calibrate_from_scratch(labeled: pd.DataFrame) -> np.ndarray:
    """Match research tuning split; no saved models or cached thresholds.

    80% fits the temporary calibration model. From the other 20%, 35% is used
    to tune decision thresholds (about 25,730 rows for this competition).
    All rows are used again later to fit the final four models.
    """
    fit_idx, validation_idx = train_test_split(
        np.arange(len(labeled)), test_size=0.20, random_state=SEED,
        stratify=labeled["Score"],
    )
    fit = labeled.iloc[fit_idx].reset_index(drop=True)
    validation = labeled.iloc[validation_idx].reset_index(drop=True)
    tune_idx, _ = train_test_split(
        np.arange(len(validation)), test_size=0.65, random_state=SEED + 7,
        stratify=validation["Score"],
    )
    tune = validation.iloc[tune_idx].reset_index(drop=True)
    log(f"CALIBRATION: {len(fit):,} fit / {len(tune):,} tuning reviews")
    x_fit, x_tune = prepare_hybrid(fit, tune)
    raw_cumulative = train_four_thresholds(x_fit, fit["Score"].to_numpy(), x_tune)
    corrected = project_cumulative(raw_cumulative)
    base = macro_f1(tune["Score"].to_numpy(),
                    threshold_predictions(corrected, np.full(4, .5, dtype=np.float32)))
    thresholds, tuned_f1 = tune_ordinal_thresholds(corrected, tune["Score"].to_numpy())
    log(f"Freshly calibrated cutoffs: {[round(float(x), 4) for x in thresholds]}")
    log(f"Calibration tuning Macro F1: {base:.5f} -> {tuned_f1:.5f}")
    log("Note: calibration F1 is NOT an independent final model evaluation.")
    del fit, validation, tune, x_fit, x_tune, raw_cumulative, corrected
    gc.collect()
    return thresholds


def write_submission(ids: pd.Series, scores: np.ndarray) -> Path:
    if len(ids) != len(scores):
        raise ValueError("Test ID count and prediction count differ")
    if not np.isin(scores, SCORES).all():
        raise ValueError("Predicted ratings outside 1..5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / OUTPUT_FILE
    submission = pd.DataFrame({"id": ids.to_numpy(), "Score": scores.astype(np.int8)})
    if submission["id"].isna().any() or submission["id"].duplicated().any():
        raise ValueError("Submission has blank or duplicate IDs")
    submission.to_csv(path, index=False)
    check = pd.read_csv(path, dtype={"id": str})
    if list(check.columns) != ["id", "Score"]:
        raise ValueError("CSV columns should be exactly id,Score")
    if not np.array_equal(check["id"].to_numpy(), ids.to_numpy()):
        raise ValueError("Test ID order was not preserved in saved CSV")
    if not np.array_equal(check["Score"].to_numpy(), scores):
        raise ValueError("CSV predictions differ from generated labels")
    log(f"Saved {path} ({len(scores):,} rows)")
    return path


def main() -> None:
    labeled, unlabeled, test_ids = load_data()
    log(f"Labeled reviews: {len(labeled):,}; test reviews: {len(unlabeled):,}")
    log("Selected: separate balanced TF-IDF, ordinal LogisticRegression C=4, tuned thresholds")
    thresholds = calibrate_from_scratch(labeled)

    log("FINAL TRAINING: rebuilding all features and training on ALL labeled reviews")
    x_train, x_test = prepare_hybrid(labeled, unlabeled)
    raw_cumulative = train_four_thresholds(x_train, labeled["Score"].to_numpy(), x_test)
    corrected = project_cumulative(raw_cumulative)
    scores = threshold_predictions(corrected, thresholds)
    write_submission(test_ids, scores)
    log("Done. No Kaggle submission was uploaded; don't commit predictions or datasets to Git.")


if __name__ == "__main__":
    main()
