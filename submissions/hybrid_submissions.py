"""Create reproducible Kaggle submissions from the recommender research models.

Install: pip install numpy pandas scipy scikit-learn

Run:
    python hybrid_submissions.py --model both
    python hybrid_submissions.py --model hybrid
    python hybrid_submissions.py --model hybrid_mf_artist

Reads train.csv and test.csv beside this script or in data/.
Writes predictions into submissions/ without modifying research results.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "submissions"
SEED = 42
SCORES = np.arange(1, 6)
ID_COLUMNS = ("reviewerID", "album_mbid", "artist_mbid")
OOF_FOLDS = 4
GROUP_SMOOTHING = {"reviewerID": 12.0, "album_mbid": 15.0, "artist_mbid": 20.0}
GENRE_SMOOTHING = 20.0
USER_GENRE_SMOOTHING = 8.0
MF_FACTORS = 16
MF_EPOCHS = 16
MF_BATCH_SIZE = 4096
MF_LEARNING_RATE = 0.008
MF_REGULARIZATION = 0.08


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


def fit_hybrid(labeled: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    log("Building out-of-fold metadata...")
    meta_train, meta_test = build_metadata(labeled, test)
    log("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50_000, min_df=2,
        sublinear_tf=True, strip_accents="unicode"
    )
    text_train = vectorizer.fit_transform(make_text(labeled))
    text_test = vectorizer.transform(make_text(test))
    x_train = sparse.hstack((text_train, meta_train), format="csr")
    x_test = sparse.hstack((text_test, meta_test), format="csr")
    log(f"Hybrid matrix: {x_train.shape[0]:,} rows x {x_train.shape[1]:,} columns")

    model = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
    log("Fitting final LinearSVC on all labeled reviews...")
    model.fit(x_train, labeled["Score"].to_numpy())
    decisions = model.decision_function(x_test)
    predictions = model.predict(x_test).astype(np.int8)
    return predictions, decisions


class BiasModel:
    def __init__(self):
        self.mean = 0.0
        self.penalty = {"reviewerID": 15.0, "artist_mbid": 20.0, "album_mbid": 25.0}
        self.biases = {key: pd.Series(dtype=np.float64) for key in ID_COLUMNS}

    def fit(self, frame: pd.DataFrame) -> "BiasModel":
        y = frame["Score"].to_numpy(dtype=np.float64)
        self.mean = float(y.mean())
        for _ in range(5):
            for key in ("reviewerID", "artist_mbid", "album_mbid"):
                residual = y - self.mean
                for other in ID_COLUMNS:
                    if other != key:
                        residual -= frame[other].map(self.biases[other]).fillna(0).to_numpy()
                active = frame[key].to_numpy() != ""
                aggregates = pd.DataFrame({
                    "key": frame[key].to_numpy()[active],
                    "residual": residual[active],
                })
                grouped = aggregates.groupby("key", sort=False)["residual"].agg(["sum", "count"])
                self.biases[key] = grouped["sum"] / (grouped["count"] + self.penalty[key])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        result = np.full(len(frame), self.mean, dtype=np.float64)
        for key in ID_COLUMNS:
            result += frame[key].map(self.biases[key]).fillna(0).to_numpy()
        return np.clip(result, 1, 5)


class ArtistMatrixFactorization:
    def __init__(self):
        self.user_index = pd.Index([])
        self.artist_index = pd.Index([])
        self.user_factors = np.empty((0, MF_FACTORS), dtype=np.float32)
        self.artist_factors = np.empty((0, MF_FACTORS), dtype=np.float32)

    def fit(self, frame: pd.DataFrame, residual: np.ndarray) -> "ArtistMatrixFactorization":
        users = frame["reviewerID"]
        artists = frame["artist_mbid"]
        user_count = users[users != ""].value_counts()
        artist_count = artists[artists != ""].value_counts()
        eligible = (users.map(user_count).fillna(0).to_numpy() >= 2) & (
            artists.map(artist_count).fillna(0).to_numpy() >= 2
        )
        selection = np.flatnonzero(eligible)
        if not len(selection):
            raise ValueError("No repeated reviewer-artist interactions")

        user_codes, names = pd.factorize(users.iloc[selection], sort=False)
        artist_codes, artist_names = pd.factorize(artists.iloc[selection], sort=False)
        self.user_index = pd.Index(names)
        self.artist_index = pd.Index(artist_names)
        targets = residual[selection].astype(np.float32)
        rng = np.random.default_rng(SEED)
        self.user_factors = rng.normal(
            0, 0.10, (len(self.user_index), MF_FACTORS)
        ).astype(np.float32)
        self.artist_factors = rng.normal(
            0, 0.10, (len(self.artist_index), MF_FACTORS)
        ).astype(np.float32)

        for epoch in range(MF_EPOCHS):
            order = np.random.default_rng(SEED + 1000 + epoch).permutation(len(selection))
            for start in range(0, len(order), MF_BATCH_SIZE):
                batch = order[start:start + MF_BATCH_SIZE]
                u, i, y = user_codes[batch], artist_codes[batch], targets[batch]
                p, q = self.user_factors[u].copy(), self.artist_factors[i].copy()
                error = y - np.einsum("ij,ij->i", p, q)
                step = MF_LEARNING_RATE / (1 + epoch * 0.04)
                np.add.at(self.user_factors, u, step * (error[:, None] * q - MF_REGULARIZATION * p))
                np.add.at(self.artist_factors, i, step * (error[:, None] * p - MF_REGULARIZATION * q))
            fitted = np.einsum(
                "ij,ij->i", self.user_factors[user_codes], self.artist_factors[artist_codes]
            )
            rmse = float(np.sqrt(np.mean((targets - fitted) ** 2)))
            if not np.isfinite(rmse):
                raise FloatingPointError("Matrix factorization diverged")
            log(f"Artist MF epoch {epoch + 1}/{MF_EPOCHS}: residual RMSE={rmse:.4f}")
        return self

    def predict_residual(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        users = self.user_index.get_indexer(frame["reviewerID"])
        artists = self.artist_index.get_indexer(frame["artist_mbid"])
        known = (users >= 0) & (artists >= 0)
        residual = np.zeros(len(frame), dtype=np.float32)
        residual[known] = np.einsum(
            "ij,ij->i", self.user_factors[users[known]], self.artist_factors[artists[known]]
        )
        return residual, known


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_hybrid_mf_artist(
    labeled: pd.DataFrame, test: pd.DataFrame, decisions: np.ndarray
) -> np.ndarray:
    log("Fitting reviewer/album/artist bias model...")
    bias = BiasModel().fit(labeled)
    residual = labeled["Score"].to_numpy(dtype=np.float64) - bias.predict(labeled)
    log("Training artist matrix factorization...")
    model = ArtistMatrixFactorization().fit(labeled, residual)
    latent_residual, warm = model.predict_residual(test)
    rating = np.clip(bias.predict(test) + latent_residual, 1, 5).reshape(-1, 1)
    text_probs = softmax(decisions)
    cf_probs = np.exp(-0.5 * ((SCORES.reshape(1, -1) - rating) / 0.85) ** 2)
    cf_probs /= cf_probs.sum(axis=1, keepdims=True)
    blended = text_probs.copy()
    blended[warm] = 0.80 * text_probs[warm] + 0.20 * cf_probs[warm]
    log(f"Artist MF applied to {warm.sum():,}/{len(test):,} test reviews ({warm.mean():.1%})")
    return SCORES[np.argmax(blended, axis=1)].astype(np.int8)


def save_submission(ids: pd.Series, predictions: np.ndarray, filename: str) -> None:
    if len(predictions) != len(ids) or not np.isin(predictions, SCORES).all():
        raise ValueError("Invalid submission predictions")
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    submission = pd.DataFrame({"id": ids.to_numpy(), "Score": predictions.astype(int)})
    submission.to_csv(path, index=False)
    log(f"Saved {len(submission):,} rows: {path}")
    print(submission.head(5).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate competition submissions")
    parser.add_argument(
        "--model", choices=("hybrid", "hybrid_mf_artist", "both"), default="both",
        help="Which submission to generate (default: both)",
    )
    args = parser.parse_args()
    labeled, test, ids = load_data()
    log(f"Labeled={len(labeled):,}; test={len(test):,}; model={args.model}")
    hybrid_predictions, decisions = fit_hybrid(labeled, test)

    if args.model in ("hybrid", "both"):
        save_submission(ids, hybrid_predictions, "submission_hybrid.csv")
    if args.model in ("hybrid_mf_artist", "both"):
        predictions = predict_hybrid_mf_artist(labeled, test, decisions)
        save_submission(ids, predictions, "submission_hybrid_mf_artist.csv")
    log("Done. No competition files were uploaded automatically.")


if __name__ == "__main__":
    main()
