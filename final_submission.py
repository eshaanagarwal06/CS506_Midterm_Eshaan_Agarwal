"""Final submission for CS506 music-rating Kaggle midterm.

Pipeline: TF-IDF + cross-fitted metadata -> four ordinal logistic models ->
out-of-fold residual learning with ExtraTrees -> adjusted rating thresholds.

Place train.csv and test.csv next to this script, then run:
    python final_submission.py

Requires: numpy, pandas, scipy, scikit-learn.
"""

from pathlib import Path
import gc
import math

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Model Configuration developed from research runs
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20261107
N_FOLDS = 4
RATINGS = np.arange(1, 6, dtype=np.int8)
ID_COLUMNS = ("reviewerID", "album_mbid", "artist_mbid")

# Selected during the earlier development experiments; not retuned on test data.
TEXT_FEATURE_LIMITS = {"summary": 20_000, "review": 90_000, "genres": 3_000}
TEXT_FIELD_WEIGHTS = {"summary": 1.0, "review": 1.0, "genres": 0.5}
METADATA_FEATURE_COUNT = 41
METADATA_WEIGHT = 1.4
LOGISTIC_C = 4.0

TREE_COUNT = 800
TREE_MIN_LEAF = 5
TREE_MAX_FEATURES = 0.8
TREE_JOBS = 2  # Keep memory usage manageable on my laptop (16GB). U can remove it on your run if needed.
PROBABILITY_TILT = 1.2
TILT_SIGMA = 0.85

# Frozen development-selected thresholds; keep the original float32 values.
RATING_THRESHOLDS = np.array(
    [0.4500000476837158, 0.5750000476837158,
     0.5750000476837158, 0.5750000476837158],
    dtype=np.float32,
)

# Smoothed historical-rating features, retained from the selected experiment.
GROUP_SMOOTHING = {"reviewerID": 12.0, "album_mbid": 15.0, "artist_mbid": 20.0}
GENRE_SMOOTHING = 20.0
REVIEWER_GENRE_SMOOTHING = 8.0


# -----------------------------------------------------------------------------
# 2. Load and preprocess the competition data
# -----------------------------------------------------------------------------

def parse_genres(genres: str) -> list[str]:
    """Return unique, lowercase genres while preserving their original order."""
    return list(dict.fromkeys(genre.strip().lower()
                              for genre in genres.split(",") if genre.strip()))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Use test.csv IDs to find their feature rows in train.csv."""
    train_path = SCRIPT_DIR / "train.csv"
    test_path = SCRIPT_DIR / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Place {path.name} in {SCRIPT_DIR}")

    all_reviews = pd.read_csv(train_path, dtype={"id": str}, low_memory=False)
    test_requests = pd.read_csv(test_path, dtype={"id": str}, low_memory=False)
    required = {"id", "Score", "summary", "reviewText", "genres", *ID_COLUMNS}
    if not required.issubset(all_reviews.columns):
        raise ValueError(f"train.csv is missing columns: {sorted(required - set(all_reviews))}")
    if "id" not in test_requests:
        raise ValueError("test.csv must contain an id column")

    for name, frame in (("train.csv", all_reviews), ("test.csv", test_requests)):
        if frame["id"].isna().any():
            raise ValueError(f"{name} contains missing IDs")
        frame["id"] = frame["id"].str.strip()
        if frame["id"].eq("").any() or frame["id"].duplicated().any():
            raise ValueError(f"{name} contains blank or duplicate IDs")

    missing_ids = ~test_requests["id"].isin(all_reviews["id"])
    if missing_ids.any():
        raise ValueError(f"{missing_ids.sum()} test IDs are missing from train.csv")

    all_reviews["Score"] = pd.to_numeric(all_reviews["Score"], errors="coerce")
    requested_ids = set(test_requests["id"])
    labeled = all_reviews.loc[
        all_reviews["Score"].notna() & ~all_reviews["id"].isin(requested_ids)
    ].copy()
    test = all_reviews.set_index("id", drop=False).loc[test_requests["id"]].reset_index(drop=True).copy()

    if labeled.empty or not labeled["Score"].isin(RATINGS).all():
        raise ValueError("Labeled scores must be integers from 1 to 5")
    labeled["Score"] = labeled["Score"].astype(np.int8)
    counts = labeled["Score"].value_counts()
    if labeled["Score"].nunique() != 5 or (counts < N_FOLDS).any():
        raise ValueError("Each rating needs at least four examples for cross-fitting")

    for frame in (labeled, test):
        for column in (*ID_COLUMNS, "summary", "reviewText", "genres"):
            frame[column] = frame[column].fillna("").astype(str)
        frame["genre_tokens"] = frame["genres"].map(parse_genres)

    return (labeled.reset_index(drop=True), test.reset_index(drop=True),
            test_requests["id"].copy())


# -----------------------------------------------------------------------------
# 3. Engineer 41 metadata features (13 static + 28 target-derived)
# -----------------------------------------------------------------------------

def static_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe review length, punctuation, helpfulness, date, and genre count."""
    summary = frame["summary"]
    review = frame["reviewText"]
    features = pd.DataFrame(index=np.arange(len(frame)))
    features["summary_log_chars"] = np.log1p(summary.str.len().to_numpy())
    features["review_log_chars"] = np.log1p(review.str.len().to_numpy())
    features["summary_log_words"] = np.log1p(summary.str.count(r"\S+").to_numpy())
    features["review_log_words"] = np.log1p(review.str.count(r"\S+").to_numpy())
    features["review_exclamations"] = np.log1p(review.str.count("!").to_numpy())
    features["review_questions"] = np.log1p(review.str.count(r"\?").to_numpy())
    features["genre_count"] = frame["genre_tokens"].map(len).to_numpy()

    total_votes = pd.to_numeric(
        frame.get("TotalVotes", pd.Series(0, index=frame.index)), errors="coerce"
    )
    helpful_votes = pd.to_numeric(
        frame.get("VotedHelpful", pd.Series(0, index=frame.index)), errors="coerce"
    )
    features["has_vote_data"] = (
        total_votes.notna() & helpful_votes.notna()
    ).to_numpy(dtype=np.float32)
    votes = total_votes.fillna(0).clip(lower=0).to_numpy(dtype=float)
    helpful = helpful_votes.fillna(0).clip(lower=0).to_numpy(dtype=float)
    features["log_votes"] = np.log1p(votes)
    features["helpful_ratio"] = (np.minimum(helpful, votes) + 1) / (votes + 2)

    timestamps = pd.to_numeric(
        frame.get("unixReviewTime", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    dates = pd.to_datetime(timestamps, unit="s", errors="coerce", utc=True)
    features["review_year"] = dates.dt.year.fillna(2000).to_numpy() - 2000
    features["review_month"] = dates.dt.month.fillna(6).to_numpy()
    features["missing_review_time"] = dates.isna().to_numpy(dtype=np.float32)
    return features.astype(np.float32)


def group_rating_metadata(reference: pd.DataFrame, query: pd.DataFrame,
                          global_mean: float, global_probs: np.ndarray) -> pd.DataFrame:
    """Create 8 smoothed rating-history features for each reviewer, album, artist."""
    features = pd.DataFrame(index=np.arange(len(query)))
    prefixes = {"reviewerID": "reviewer", "album_mbid": "album", "artist_mbid": "artist"}

    for id_column in ID_COLUMNS:
        known = reference.loc[reference[id_column] != ""]
        aggregate = known.groupby(id_column, sort=False)["Score"].agg(["count", "sum", "std"])
        counts = query[id_column].map(aggregate["count"]).fillna(0).to_numpy(dtype=float)
        sums = query[id_column].map(aggregate["sum"]).fillna(0).to_numpy(dtype=float)
        standard_deviation = query[id_column].map(aggregate["std"]).fillna(0).to_numpy(dtype=float)
        smoothing = GROUP_SMOOTHING[id_column]
        prefix = prefixes[id_column]

        features[f"{prefix}_log_count"] = np.log1p(counts)
        features[f"{prefix}_mean"] = (sums + smoothing * global_mean) / (counts + smoothing)
        features[f"{prefix}_std"] = standard_deviation * np.sqrt(counts / (counts + smoothing))

        class_counts = known.groupby([id_column, "Score"]).size().unstack(fill_value=0)
        for rating in RATINGS:
            class_column = (class_counts[rating] if rating in class_counts
                            else pd.Series(dtype=np.float64))
            count_at_rating = query[id_column].map(class_column).fillna(0).to_numpy(dtype=float)
            features[f"{prefix}_p{rating}"] = (
                count_at_rating + smoothing * global_probs[rating - 1]
            ) / (counts + smoothing)

    return features


def genre_rating_metadata(reference: pd.DataFrame, query: pd.DataFrame,
                          global_mean: float, reviewer_means: pd.Series) -> pd.DataFrame:
    """Create genre priors and reviewer-specific genre preference features."""
    features = pd.DataFrame(index=np.arange(len(query)))
    historical = reference[["reviewerID", "Score", "genre_tokens"]].explode("genre_tokens")
    historical = historical.loc[
        historical["genre_tokens"].notna() & historical["genre_tokens"].ne("")
    ].copy()

    if historical.empty:
        return pd.DataFrame({
            "genre_prior": np.full(len(query), global_mean),
            "genre_log_support": np.zeros(len(query)),
            "reviewer_genre_preference": np.zeros(len(query)),
            "reviewer_genre_log_support": np.zeros(len(query)),
        })

    historical["reviewer_prior"] = historical["reviewerID"].map(
        reviewer_means
    ).fillna(global_mean)
    historical["relative_score"] = historical["Score"] - historical["reviewer_prior"]
    genre_aggregate = historical.groupby("genre_tokens", sort=False)["Score"].agg(["sum", "count"])
    genre_means = (
        (genre_aggregate["sum"] + GENRE_SMOOTHING * global_mean)
        / (genre_aggregate["count"] + GENRE_SMOOTHING)
    )
    reviewer_genre = historical.loc[historical["reviewerID"] != ""].groupby(
        ["reviewerID", "genre_tokens"], sort=False
    )["relative_score"].agg(["sum", "count"])
    reviewer_genre_effect = reviewer_genre["sum"] / (
        reviewer_genre["count"] + REVIEWER_GENRE_SMOOTHING
    )

    expanded = query[["reviewerID", "genre_tokens"]].copy()
    expanded["row_number"] = np.arange(len(query))
    expanded = expanded.explode("genre_tokens").reset_index(drop=True)
    expanded = expanded.loc[
        expanded["genre_tokens"].notna() & expanded["genre_tokens"].ne("")
    ].copy()
    if expanded.empty:
        features["genre_prior"] = global_mean
        features["genre_log_support"] = 0.0
        features["reviewer_genre_preference"] = 0.0
        features["reviewer_genre_log_support"] = 0.0
        return features

    expanded["genre_prior"] = expanded["genre_tokens"].map(genre_means).fillna(global_mean)
    expanded["genre_support"] = expanded["genre_tokens"].map(genre_aggregate["count"]).fillna(0)
    reviewer_genre_pairs = pd.MultiIndex.from_frame(expanded[["reviewerID", "genre_tokens"]])
    expanded["profile"] = reviewer_genre_effect.reindex(reviewer_genre_pairs).astype(float).to_numpy()
    expanded["profile_support"] = reviewer_genre["count"].reindex(
        reviewer_genre_pairs
    ).astype(float).to_numpy()
    expanded["profile"] = expanded["profile"].fillna(0)
    expanded["profile_support"] = expanded["profile_support"].fillna(0)

    per_review = expanded.groupby("row_number", sort=False).agg(
        genre_prior=("genre_prior", "mean"),
        genre_support=("genre_support", "mean"),
        reviewer_genre_preference=("profile", "mean"),
        reviewer_genre_support=("profile_support", "sum"),
    )
    row_indices = np.arange(len(query))
    features["genre_prior"] = (
        per_review["genre_prior"].reindex(row_indices).fillna(global_mean).to_numpy()
    )
    features["genre_log_support"] = np.log1p(
        per_review["genre_support"].reindex(row_indices).fillna(0).to_numpy()
    )
    features["reviewer_genre_preference"] = per_review[
        "reviewer_genre_preference"
    ].reindex(row_indices).fillna(0).to_numpy()
    features["reviewer_genre_log_support"] = np.log1p(
        per_review["reviewer_genre_support"].reindex(row_indices).fillna(0).to_numpy()
    )
    return features


def target_metadata(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """Derive 28 rating-dependent features using reference labels only."""
    global_mean = float(reference["Score"].mean())
    rating_counts = np.bincount(reference["Score"].to_numpy(dtype=np.int64), minlength=6)[1:6]
    global_probs = rating_counts / rating_counts.sum()
    group_features = group_rating_metadata(reference, query, global_mean, global_probs)

    reviewer_stats = reference.loc[reference["reviewerID"] != ""].groupby(
        "reviewerID"
    )["Score"].agg(["sum", "count"])
    smoothing = GROUP_SMOOTHING["reviewerID"]
    reviewer_means = (
        (reviewer_stats["sum"] + smoothing * global_mean)
        / (reviewer_stats["count"] + smoothing)
    )
    genre_features = genre_rating_metadata(reference, query, global_mean, reviewer_means)
    return pd.concat([group_features, genre_features], axis=1).astype(np.float32)


def outer_folds(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reuse these splits for global cross-fitting and ordinal OOF predictions."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + 1)
    return list(splitter.split(np.zeros(len(labels)), labels))


def scale_metadata(train_features: np.ndarray,
                   query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize with training statistics, clip, and normalize by sqrt(41)."""
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_query = scaler.transform(query_features)
    divisor = np.sqrt(float(METADATA_FEATURE_COUNT))
    return (np.clip(scaled_train, -6, 6).astype(np.float32) / divisor,
            np.clip(scaled_query, -6, 6).astype(np.float32) / divisor)


def build_full_metadata(labeled: pd.DataFrame, test: pd.DataFrame,
                        folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Use out-of-fold target features for training; all labels for test features."""
    target_oof = None
    feature_names = None

    for train_indices, held_indices in folds:
        chunk = target_metadata(labeled.iloc[train_indices], labeled.iloc[held_indices])
        if target_oof is None:
            feature_names = list(chunk.columns)
            target_oof = np.empty((len(labeled), len(feature_names)), dtype=np.float32)
        target_oof[held_indices] = chunk[feature_names].to_numpy(dtype=np.float32)

    test_target = target_metadata(labeled, test)[feature_names].to_numpy(dtype=np.float32)
    train_raw = np.column_stack((target_oof, static_metadata(labeled).to_numpy(dtype=np.float32)))
    test_raw = np.column_stack((test_target, static_metadata(test).to_numpy(dtype=np.float32)))
    if train_raw.shape[1] != METADATA_FEATURE_COUNT:
        raise ValueError("Unexpected metadata feature count")
    return scale_metadata(train_raw, test_raw)


def build_fold_metadata(training: pd.DataFrame, held_out: pd.DataFrame,
                        fold_number: int) -> tuple[np.ndarray, np.ndarray]:
    """Nest cross-fitting inside the outer fold to avoid own-label leakage."""
    labels = training["Score"].to_numpy()
    inner_splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + 200 + fold_number
    )
    inner_oof = None
    feature_names = None
    for train_indices, val_indices in inner_splitter.split(np.zeros(len(training)), labels):
        chunk = target_metadata(training.iloc[train_indices], training.iloc[val_indices])
        if inner_oof is None:
            feature_names = list(chunk.columns)
            inner_oof = np.empty((len(training), len(feature_names)), dtype=np.float32)
        inner_oof[val_indices] = chunk[feature_names].to_numpy(dtype=np.float32)

    held_features = target_metadata(training, held_out)[feature_names].to_numpy(dtype=np.float32)
    train_raw = np.column_stack((inner_oof, static_metadata(training).to_numpy(dtype=np.float32)))
    held_raw = np.column_stack((held_features, static_metadata(held_out).to_numpy(dtype=np.float32)))
    return scale_metadata(train_raw, held_raw)


# -----------------------------------------------------------------------------
# 4. Separate-field TF-IDF and four ordinal logistic regressions
# -----------------------------------------------------------------------------

def text_for_field(frame: pd.DataFrame, field: str) -> pd.Series:
    if field == "review":
        return frame["reviewText"]
    if field == "genres":
        return frame["genres"].str.replace(",", " ", regex=False)
    return frame["summary"]


def vectorize_text(training: pd.DataFrame, query: pd.DataFrame
                  ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit each TF-IDF vocabulary on training reviews, never held-out reviews."""
    train_blocks, query_blocks = [], []
    weight_norm = math.sqrt(sum(weight * weight for weight in TEXT_FIELD_WEIGHTS.values()))

    for field in ("summary", "review", "genres"):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=TEXT_FEATURE_LIMITS[field],
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        vectorizer.fit(text_for_field(training, field))
        train_matrix = vectorizer.transform(text_for_field(training, field)).astype(np.float32)
        query_matrix = vectorizer.transform(text_for_field(query, field)).astype(np.float32)
        weight = TEXT_FIELD_WEIGHTS[field] / weight_norm
        train_blocks.append(train_matrix.multiply(weight).tocsr())
        query_blocks.append(query_matrix.multiply(weight).tocsr())

    return (sparse.hstack(train_blocks, format="csr", dtype=np.float32),
            sparse.hstack(query_blocks, format="csr", dtype=np.float32))


def project_cumulative(probabilities: np.ndarray) -> np.ndarray:
    """Enforce P(Score>1) >= ... >= P(Score>4)."""
    return np.minimum.accumulate(
        np.clip(np.asarray(probabilities, dtype=np.float32), 0, 1), axis=1
    )


def cumulative_to_class_probs(cumulative: np.ndarray) -> np.ndarray:
    cumulative = project_cumulative(cumulative)
    class_probs = np.column_stack((
        1 - cumulative[:, 0],
        cumulative[:, 0] - cumulative[:, 1],
        cumulative[:, 1] - cumulative[:, 2],
        cumulative[:, 2] - cumulative[:, 3],
        cumulative[:, 3],
    ))
    class_probs = np.maximum(class_probs, 1e-7).astype(np.float32)
    return class_probs / np.maximum(class_probs.sum(axis=1, keepdims=True), 1e-10)


def class_to_cumulative(class_probs: np.ndarray) -> np.ndarray:
    class_probs = np.asarray(class_probs, dtype=np.float32)
    return project_cumulative(np.column_stack((
        class_probs[:, 1:].sum(axis=1),
        class_probs[:, 2:].sum(axis=1),
        class_probs[:, 3:].sum(axis=1),
        class_probs[:, 4],
    )))


def predict_ordinal(training: pd.DataFrame, query: pd.DataFrame,
                    train_metadata: np.ndarray, query_metadata: np.ndarray) -> np.ndarray:
    """Predict P(Score>1), ..., P(Score>4) using independent logistic models."""
    train_text, query_text = vectorize_text(training, query)
    train_features = sparse.hstack(
        (train_text, sparse.csr_matrix(train_metadata * METADATA_WEIGHT)),
        format="csr", dtype=np.float32,
    )
    query_features = sparse.hstack(
        (query_text, sparse.csr_matrix(query_metadata * METADATA_WEIGHT)),
        format="csr", dtype=np.float32,
    )
    labels = training["Score"].to_numpy(dtype=np.int8)
    cumulative = np.zeros((query_features.shape[0], 4), dtype=np.float32)

    for index, cutoff in enumerate(range(1, 5)):
        classifier = LogisticRegression(
            C=LOGISTIC_C, solver="liblinear", max_iter=350,
            tol=1e-4, random_state=RANDOM_SEED,
        )
        classifier.fit(train_features, (labels > cutoff).astype(np.int8))
        cumulative[:, index] = classifier.predict_proba(query_features)[:, 1].astype(np.float32)

    return project_cumulative(cumulative)


def out_of_fold_predictions(labeled: pd.DataFrame,
                            folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Refit vocabulary, metadata encodings, scaler, and all four heads per fold."""
    cumulative_oof = np.empty((len(labeled), 4), dtype=np.float32)
    for fold_number, (train_indices, held_indices) in enumerate(folds, start=1):
        print(f"Training ordinal out-of-fold model {fold_number}/{N_FOLDS}", flush=True)
        training = labeled.iloc[train_indices].reset_index(drop=True)
        held_out = labeled.iloc[held_indices].reset_index(drop=True)
        train_metadata, held_metadata = build_fold_metadata(training, held_out, fold_number)
        cumulative_oof[held_indices] = predict_ordinal(
            training, held_out, train_metadata, held_metadata
        )
        del training, held_out, train_metadata, held_metadata
        gc.collect()
    return cumulative_oof


# -----------------------------------------------------------------------------
# 5. Train ExtraTrees on out-of-fold ordinal prediction residuals
# -----------------------------------------------------------------------------

def stacking_features(cumulative: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    """Combine 12 ordinal uncertainty features with the 41 metadata columns."""
    class_probs = cumulative_to_class_probs(cumulative)
    entropy = -np.sum(class_probs * np.log(np.maximum(class_probs, 1e-7)),
                      axis=1, keepdims=True)
    sorted_probs = np.sort(class_probs, axis=1)
    margin = (sorted_probs[:, -1] - sorted_probs[:, -2])[:, None]
    expected_score = (class_probs @ np.arange(1, 6, dtype=np.float32))[:, None]
    ordinal_features = np.hstack((cumulative, class_probs, expected_score, entropy, margin))
    return np.nan_to_num(np.hstack((ordinal_features, metadata)).astype(np.float32),
                         posinf=0.0, neginf=0.0)


def correct_probabilities(cumulative_oof: np.ndarray,
                          cumulative_test: np.ndarray,
                          train_metadata: np.ndarray,
                          test_metadata: np.ndarray,
                          ratings: np.ndarray) -> np.ndarray:
    """Fit residuals to true scores and tilt the final test class probabilities."""
    train_stack = stacking_features(cumulative_oof, train_metadata)
    test_stack = stacking_features(cumulative_test, test_metadata)
    expected_training_scores = (
        cumulative_to_class_probs(cumulative_oof) @ np.arange(1, 6, dtype=np.float32)
    )
    residuals = ratings.astype(np.float32) - expected_training_scores

    forest = ExtraTreesRegressor(
        n_estimators=TREE_COUNT,
        min_samples_leaf=TREE_MIN_LEAF,
        max_features=TREE_MAX_FEATURES,
        random_state=RANDOM_SEED + TREE_MIN_LEAF,
        n_jobs=TREE_JOBS,
    )
    forest.fit(train_stack, residuals)
    predicted_residuals = forest.predict(test_stack).astype(np.float32)

    class_probs = cumulative_to_class_probs(cumulative_test)
    expected_test_scores = class_probs @ np.arange(1, 6, dtype=np.float32)
    rating_values = np.arange(1, 6, dtype=np.float32)[None, :]
    adjusted_residuals = predicted_residuals * PROBABILITY_TILT
    # Original exponential tilt: shift class mass without changing rating support.
    exponents = np.clip(
        adjusted_residuals[:, None] * (rating_values - expected_test_scores[:, None])
        / (TILT_SIGMA * TILT_SIGMA),
        -5, 5,
    )
    tilted_probs = class_probs * np.exp(exponents)
    normalized_probs = tilted_probs / np.maximum(tilted_probs.sum(axis=1, keepdims=True), 1e-8)
    return class_to_cumulative(normalized_probs)


# -----------------------------------------------------------------------------
# 6. Final full-data fit and Kaggle submission
# -----------------------------------------------------------------------------

def main() -> None:
    labeled, test, test_ids = load_data()
    labels = labeled["Score"].to_numpy(dtype=np.int8)
    folds = outer_folds(labels)
    print(f"Loaded {len(labeled):,} labeled and {len(test):,} test reviews", flush=True)

    print("Cross-fitting metadata features", flush=True)
    train_metadata, test_metadata = build_full_metadata(labeled, test, folds)
    cumulative_oof = out_of_fold_predictions(labeled, folds)

    print("Training final ordinal model on all labeled reviews", flush=True)
    cumulative_test = predict_ordinal(labeled, test, train_metadata, test_metadata)

    print("Fitting ExtraTrees residual correction", flush=True)
    corrected = correct_probabilities(
        cumulative_oof, cumulative_test, train_metadata, test_metadata, labels
    )
    predictions = (1 + (project_cumulative(corrected) > RATING_THRESHOLDS[None, :])
                   .sum(axis=1)).astype(np.int8)

    # test.csv determines both the set and the exact order of submission IDs.
    submission = pd.DataFrame({"id": test_ids.to_numpy(), "Score": predictions.astype(int)})
    output_path = SCRIPT_DIR / "final_submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
