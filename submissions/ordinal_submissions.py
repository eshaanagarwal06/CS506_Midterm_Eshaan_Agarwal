"""Train the ordinal-offset model from scratch and create one Kaggle CSV.

Usage:
    python ordinal_submissions.py

Requirements: pip install numpy pandas scipy scikit-learn

Place train.csv and test.csv alongside this script (or inside data/). This file
never imports research scripts or loads research caches, pretrained models, or
previously fitted offsets. It fits everything from labeled training data.

Produces submissions/submission_ordinal_offsets.csv in test.csv row order.
No neural networks, boosting, resource limits, or Kaggle uploads.
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
SEED = 42
SCORES = np.arange(1, 6)
ID_COLUMNS = ("reviewerID", "album_mbid", "artist_mbid")
OOF_FOLDS = 4
GROUP_SMOOTHING = {"reviewerID": 12.0, "album_mbid": 15.0, "artist_mbid": 20.0}
GENRE_SMOOTHING = 20.0
USER_GENRE_SMOOTHING = 8.0
LOGISTIC_C = 1.0


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


def make_text(frame: pd.DataFrame) -> pd.Series:
    genres = frame["genres"].str.replace(",", " ", regex=False)
    return "summary " + frame["summary"] + " review " + frame["reviewText"] + " genres " + genres



def prepare_hybrid(fit: pd.DataFrame, predict: pd.DataFrame) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build fresh OOF metadata, TF-IDF, and hybrid features for one training stage."""
    log(f"Building leakage-safe metadata from {len(fit):,} labeled rows...")
    m_fit, m_predict = build_metadata(fit, predict)

    log("Fitting TF-IDF on this stage's labeled training rows...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50_000, min_df=2,
        sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
    )
    t_fit = vectorizer.fit_transform(make_text(fit)).tocsr()
    t_predict = vectorizer.transform(make_text(predict)).tocsr()
    x_fit = sparse.hstack((t_fit, m_fit), format="csr", dtype=np.float32)
    x_predict = sparse.hstack((t_predict, m_predict), format="csr", dtype=np.float32)
    log(f"Hybrid features: {x_fit.shape[0]:,} rows x {x_fit.shape[1]:,} columns")
    return x_fit, x_predict


def train_four_thresholds(x_fit: sparse.csr_matrix, y_fit: np.ndarray,
                          x_predict: sparse.csr_matrix) -> np.ndarray:
    """Fresh binary logistic models for P(Score > 1), ..., P(Score > 4)."""
    cumulative = np.empty((x_predict.shape[0], 4), dtype=np.float32)
    for i, threshold in enumerate(range(1, 5)):
        log(f"Training LogisticRegression for P(Score > {threshold}) ({i + 1}/4)...")
        binary = (y_fit > threshold).astype(np.int8)
        if np.unique(binary).size != 2:
            raise ValueError(f"Only one binary class for threshold {threshold}")
        model = LogisticRegression(
            C=LOGISTIC_C, solver="liblinear", max_iter=250,
            tol=1e-4, class_weight=None, random_state=SEED,
        )
        model.fit(x_fit, binary)
        cumulative[:, i] = model.predict_proba(x_predict)[:, 1].astype(np.float32)
        log(f"Threshold {threshold} fitted (iterations: {int(model.n_iter_[0])})")
        del model
    return cumulative


def ordinal_probabilities(cumulative: np.ndarray) -> np.ndarray:
    """Project cumulative probabilities to nonincreasing order and derive 5-class probabilities."""
    if cumulative.ndim != 2 or cumulative.shape[1] != 4:
        raise ValueError("Expected probabilities for exactly four ordinal thresholds")
    corrected = np.minimum.accumulate(np.clip(cumulative, 0, 1), axis=1)
    p = np.column_stack((
        1 - corrected[:, 0],
        corrected[:, 0] - corrected[:, 1],
        corrected[:, 1] - corrected[:, 2],
        corrected[:, 2] - corrected[:, 3],
        corrected[:, 3],
    ))
    p = np.maximum(p, 0)
    p /= p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all():
        raise ValueError("Invalid predicted probabilities")
    return p.astype(np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast, fixed-5-class Macro F1 used only for calibration."""
    a = np.asarray(y_true, dtype=np.int32) - 1
    b = np.asarray(y_pred, dtype=np.int32) - 1
    cm = np.bincount(a * 5 + b, minlength=25).reshape(5, 5)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    scores = np.divide(2 * cm.diagonal(), denom,
                       out=np.zeros(5, dtype=np.float64), where=denom > 0)
    return float(scores.mean())


def optimize_offsets(log_probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Reproduce the research search *method*, but re-learn offsets from train.csv."""
    offsets = np.zeros(5, dtype=np.float32)
    best = macro_f1(labels, SCORES[np.argmax(log_probs, axis=1)])
    for step in (0.40, 0.20, 0.10, 0.05, 0.02):
        for _ in range(3):
            improved = False
            for col in range(5):
                for movement in (-2, -1, 1, 2):
                    trial = offsets.copy()
                    trial[col] += movement * step
                    predicted = SCORES[np.argmax(log_probs + trial, axis=1)]
                    performance = macro_f1(labels, predicted)
                    if performance > best + 1e-7:
                        best, offsets, improved = performance, trial, True
            if not improved:
                break
    # Adding a constant to all five offsets doesn't change the argmax.
    offsets -= offsets.mean()
    return offsets, best


def fit_fresh_offsets(all_labeled: pd.DataFrame) -> tuple[np.ndarray, float]:
    """No saved research models or research-fitted offsets; calibration uses labeled data only.

    Match the prior research split: train on 80% of labeled data, select a
    stratified 35% of the remaining 20% to tune class offsets. The other 65%
    is NOT used for tuning. The final classifier will later be refitted on ALL
    labeled data, including these held-out rows.
    """
    log("CALIBRATION STAGE: fitting four completely new models on 80% of labeled reviews")
    indices = np.arange(len(all_labeled))
    train_ids, validation_ids = train_test_split(
        indices, test_size=0.20, random_state=SEED,
        stratify=all_labeled["Score"],
    )
    calibration_fit = all_labeled.iloc[train_ids].reset_index(drop=True)
    validation = all_labeled.iloc[validation_ids].reset_index(drop=True)
    tune_ids, _ = train_test_split(
        np.arange(len(validation)), test_size=0.65, random_state=SEED + 7,
        stratify=validation["Score"],
    )
    tune = validation.iloc[tune_ids].reset_index(drop=True)
    log(f"Calibration training: {len(calibration_fit):,}; offset tuning: {len(tune):,}")
    x_fit, x_tune = prepare_hybrid(calibration_fit, tune)
    cumulative = train_four_thresholds(x_fit, calibration_fit["Score"].to_numpy(), x_tune)
    probs = ordinal_probabilities(cumulative)
    log_probs = np.log(np.clip(probs, 1e-8, 1.0))
    y_tune = tune["Score"].to_numpy()
    raw_f1 = macro_f1(y_tune, SCORES[np.argmax(log_probs, axis=1)])
    offsets, tuned_f1 = optimize_offsets(log_probs, y_tune)
    log(f"Offsets learned from scratch: {[round(float(v), 4) for v in offsets]}")
    log(f"Calibration tuning Macro F1: {raw_f1:.5f} -> {tuned_f1:.5f}")
    del calibration_fit, validation, tune, x_fit, x_tune, cumulative, probs, log_probs
    gc.collect()
    return offsets, tuned_f1


def write_csv(ids: pd.Series, labels: np.ndarray, filename: str) -> None:
    if len(ids) != len(labels):
        raise ValueError("Submission row count does not match test.csv")
    if not np.isin(labels, SCORES).all():
        raise ValueError("Submission contains predictions outside 1..5")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    output = pd.DataFrame({"id": ids.to_numpy(), "Score": labels.astype(np.int8)})
    if output["id"].duplicated().any() or output["id"].isna().any():
        raise ValueError("Invalid submission IDs")
    output.to_csv(path, index=False)
    verify = pd.read_csv(path, dtype={"id": str})
    if list(verify.columns) != ["id", "Score"] or not np.array_equal(
        verify["id"].to_numpy(), ids.to_numpy()
    ):
        raise ValueError("Written submission did not preserve test IDs or column order")
    log(f"Saved {path} ({len(labels):,} predictions)")


def main() -> None:
    labeled, unlabeled, test_ids = load_data()
    log(f"Loaded {len(labeled):,} labeled rows and {len(unlabeled):,} test rows")
    log("Selected: ordinal logistic regression with freshly calibrated class offsets")

    # Fit the calibration-stage models only to obtain offsets from labeled data.
    offsets, _ = fit_fresh_offsets(labeled)

    # Fit four NEW models on every labeled review for the Kaggle prediction.
    log("FINAL STAGE: training four new ordinal models on ALL labeled reviews")
    x_train, x_test = prepare_hybrid(labeled, unlabeled)
    cumulative = train_four_thresholds(x_train, labeled["Score"].to_numpy(), x_test)
    class_probs = ordinal_probabilities(cumulative)
    adjusted_log_probs = np.log(np.clip(class_probs, 1e-8, 1.0)) + offsets
    predictions = SCORES[np.argmax(adjusted_log_probs, axis=1)]
    write_csv(test_ids, predictions, "submission_ordinal_offsets.csv")

    log("Finished. No Kaggle upload performed. Do not commit generated CSVs or raw data.")


if __name__ == "__main__":
    main()
