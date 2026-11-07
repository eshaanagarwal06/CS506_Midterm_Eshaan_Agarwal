r"""CS506 music-review Kaggle submission: ordinal LR + OOF ExtraTrees residual correction.

This is a STANDALONE, full-data submission trainer. It reads only the original
train.csv / test.csv, not old research models, validation predictions, or caches.
It creates its OWN optional restart cache and, when run, writes a Kaggle CSV.

Dataset contract (as in the class competition):
    train.csv contains features for labeled AND unlabeled reviews;
    test.csv lists the 'id' values to predict (other columns are ignored).

The architecture reproduces the selected overnight research configuration:
    - 20k summary + 90k review + 3k genre TF-IDF (word 1-2 grams), weighted 1/1/.5
    - 41 cross-fitted metadata features, multiplied by 1.4
    - Four independent logistic cutoffs, C=4, liblinear
    - Four outer OOF folds with *complete* base model refits (including vocabulary,
      metadata encodings, scaler) to create a clean residual target
    - ExtraTreesRegressor (800 trees, min_samples_leaf=5, max_features=.8)
    - Exponential probability tilt 1.2, sigma=.85
    - Frozen research decision thresholds [0.45, .575, .575, .575]

No model is fit on the original competition test labels. No neural nets or
boosting. No further threshold tuning on the previously audited reviews.

Run (PowerShell, preferably with data outside OneDrive):
    pip install numpy pandas scipy scikit-learn joblib
    python .\ordinal_residual_submission.py

Optionally:
    python .\ordinal_residual_submission.py --cache-dir "C:\\KaggleResearch\\residual_submission" --jobs 2
    python .\ordinal_residual_submission.py --train "C:\\data\\train.csv" --test "C:\\data\\test.csv"

A rerun REUSES ONLY checkpoints produced by this particular script if the
script, original CSVs, and scikit-learn version have not changed. To start
completely fresh, delete its dedicated cache directory, or use a new one.

Outputs:
    submissions/submission_ordinal_C4_extratrees_residual_leaf5.csv
    residual_submission_cache/submission_diagnostics.json (by default)

IMPORTANT: This script is intentionally NOT executed on the real competition
files here. Run it locally to generate your own predictions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SEED = 20261107
VERSION = "C4-OOF-residual-leaf5-final-v1"
N_FOLDS = 4
N_META = 41
C = 4.0
META_WEIGHT = 1.4
TREE_COUNT = 800
TREE_LEAF = 5
TREE_MAX_FEATURES = 0.8
TILT = 1.2
TILT_SIGMA = 0.85
# Taken directly from the development-selected (NOT audit-retuned) winning candidate.
THRESHOLDS = np.array([
    0.4500000476837158,
    0.5750000476837158,
    0.5750000476837158,
    0.5750000476837158,
], dtype=np.float32)
CLASSES = np.arange(1, 6, dtype=np.int8)
ID_COLUMNS = ("reviewerID", "album_mbid", "artist_mbid")
GROUP_SMOOTHING = {"reviewerID": 12., "album_mbid": 15., "artist_mbid": 20.}
GENRE_SMOOTHING = 20.
USER_GENRE_SMOOTHING = 8.
TEXT_WEIGHTS = {"summary": 1., "review": 1., "genres": .5}
FIELD_CAPS = {"summary": 20000, "review": 90000, "genres": 3000}


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    tmp.replace(path)


def save_object(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(model, tmp, compress=1)
    tmp.replace(path)


def verify_manifest(cache: Path, train_path: Path, test_path: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": VERSION,
        "script_sha256": sha256(Path(__file__).resolve()),
        "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path),
        "sklearn_version": sklearn.__version__,
    }
    path = cache / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(
                "Cache manifest differs from the input CSVs, script, or sklearn version. "
                "Choose a NEW --cache-dir (or delete the dedicated old cache directory) "
                "to ensure no mismatched artifacts are reused."
            )
    else:
        if any(cache.iterdir()):
            raise RuntimeError("Cache dir contains files but no manifest; choose an empty --cache-dir.")
        atomic_json(path, manifest)


def field_text(frame: pd.DataFrame, field: str) -> pd.Series:
    if field == "review":
        return frame["reviewText"]
    if field == "genres":
        return frame["genres"].str.replace(",", " ", regex=False)
    return frame["summary"]


def project_q(q: np.ndarray) -> np.ndarray:
    return np.minimum.accumulate(np.clip(np.asarray(q, dtype=np.float32), 0, 1), axis=1)


def q_to_p(q: np.ndarray) -> np.ndarray:
    q = project_q(q)
    p = np.column_stack((
        1 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1] - q[:, 2],
        q[:, 2] - q[:, 3], q[:, 3],
    ))
    p = np.maximum(p, 1e-7).astype(np.float32)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-10)


def p_to_q(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    return project_q(np.column_stack((
        p[:, 1:].sum(axis=1), p[:, 2:].sum(axis=1),
        p[:, 3:].sum(axis=1), p[:, 4],
    )))


def predict_threshold(q: np.ndarray, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    return (1 + (project_q(q) > thresholds[None, :]).sum(axis=1)).astype(np.int8)


def shift_probabilities(q: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """EXACT research probability tilt, with delta already multiplied by TILT."""
    p = q_to_p(q)
    expected = p @ np.arange(1, 6, dtype=np.float32)
    k = np.arange(1, 6, dtype=np.float32)[None, :]
    tilted = p * np.exp(np.clip(
        delta[:, None] * (k - expected[:, None]) / (TILT_SIGMA * TILT_SIGMA),
        -5, 5,
    ))
    return p_to_q(tilted / np.maximum(tilted.sum(axis=1, keepdims=True), 1e-8))


# METADATA FEATURE IMPLEMENTATION
# Copied without modifying feature definitions, smoothing, or column ordering
# from the user's completed overnight research script. Those details must
# match between training, OOF targets, and final test predictions.

def parse_genres(text: str) -> list[str]:
    return list(dict.fromkeys(s.strip().lower() for s in text.split(",") if s.strip()))


def static_features(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame["summary"]
    review = frame["reviewText"]
    result = pd.DataFrame(index=np.arange(len(frame)))
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
    votes = total.fillna(0).clip(lower=0).to_numpy(dtype=float)
    helpful_votes = helpful.fillna(0).clip(lower=0).to_numpy(dtype=float)
    result["log_votes"] = np.log1p(votes)
    result["helpful_ratio"] = (np.minimum(helpful_votes, votes) + 1) / (votes + 2)

    timestamp = pd.to_numeric(
        frame.get("unixReviewTime", pd.Series(np.nan, index=frame.index)), errors="coerce")
    dates = pd.to_datetime(timestamp, unit="s", errors="coerce", utc=True)
    result["review_year"] = dates.dt.year.fillna(2000).to_numpy() - 2000
    result["review_month"] = dates.dt.month.fillna(6).to_numpy()
    result["missing_review_time"] = dates.isna().to_numpy(dtype=np.float32)
    return result.astype(np.float32)


def rating_group_features(ref: pd.DataFrame, query: pd.DataFrame,
                          global_mean: float, global_probs: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(query)))
    aliases = {"reviewerID": "reviewer", "album_mbid": "album", "artist_mbid": "artist"}
    for key in ID_COLUMNS:
        active = ref.loc[ref[key] != ""]
        aggregates = active.groupby(key, sort=False)["Score"].agg(["count", "sum", "std"])
        count = query[key].map(aggregates["count"]).fillna(0).to_numpy(dtype=float)
        sums = query[key].map(aggregates["sum"]).fillna(0).to_numpy(dtype=float)
        std = query[key].map(aggregates["std"]).fillna(0).to_numpy(dtype=float)
        smoothing = GROUP_SMOOTHING[key]
        prefix = aliases[key]
        result[f"{prefix}_log_count"] = np.log1p(count)
        result[f"{prefix}_mean"] = (sums + smoothing * global_mean) / (count + smoothing)
        result[f"{prefix}_std"] = std * np.sqrt(count / (count + smoothing))
        grouped = active.groupby([key, "Score"]).size().unstack(fill_value=0)
        for rating in CLASSES:
            column = grouped[rating] if rating in grouped else pd.Series(dtype=np.float64)
            class_count = query[key].map(column).fillna(0).to_numpy(dtype=float)
            result[f"{prefix}_p{rating}"] = (
                class_count + smoothing * global_probs[rating - 1]
            ) / (count + smoothing)
    return result


def genre_features(ref: pd.DataFrame, query: pd.DataFrame,
                   global_mean: float, reviewer_mean: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=np.arange(len(query)))
    exploded = ref[["reviewerID", "Score", "genre_tokens"]].explode("genre_tokens")
    exploded = exploded.loc[exploded["genre_tokens"].notna() &
                            (exploded["genre_tokens"] != "")].copy()
    if exploded.empty:
        return pd.DataFrame({"genre_prior": np.full(len(query), global_mean),
                             "genre_log_support": np.zeros(len(query)),
                             "reviewer_genre_preference": np.zeros(len(query)),
                             "reviewer_genre_log_support": np.zeros(len(query))})
    exploded["reviewer_prior"] = (
        exploded["reviewerID"].map(reviewer_mean).fillna(global_mean))
    exploded["relative_score"] = exploded["Score"] - exploded["reviewer_prior"]
    genre_agg = exploded.groupby("genre_tokens", sort=False)["Score"].agg(["sum", "count"])
    genre_mean = ((genre_agg["sum"] + GENRE_SMOOTHING * global_mean) /
                  (genre_agg["count"] + GENRE_SMOOTHING))
    profile = exploded.loc[exploded["reviewerID"] != ""].groupby(
        ["reviewerID", "genre_tokens"], sort=False)["relative_score"].agg(["sum", "count"])
    profile_effect = profile["sum"] / (profile["count"] + USER_GENRE_SMOOTHING)

    expanded = query[["reviewerID", "genre_tokens"]].copy()
    expanded["row_number"] = np.arange(len(query))
    expanded = expanded.explode("genre_tokens").reset_index(drop=True)
    expanded = expanded.loc[expanded["genre_tokens"].notna() &
                            (expanded["genre_tokens"] != "")].copy()
    if expanded.empty:
        result["genre_prior"] = global_mean
        result["genre_log_support"] = 0.0
        result["reviewer_genre_preference"] = 0.0
        result["reviewer_genre_log_support"] = 0.0
        return result
    expanded["genre_prior"] = expanded["genre_tokens"].map(genre_mean).fillna(global_mean)
    expanded["genre_support"] = expanded["genre_tokens"].map(genre_agg["count"]).fillna(0)
    pairs = pd.MultiIndex.from_frame(expanded[["reviewerID", "genre_tokens"]])
    expanded["profile"] = profile_effect.reindex(pairs).astype(float).to_numpy()
    expanded["profile_support"] = profile["count"].reindex(pairs).astype(float).to_numpy()
    expanded["profile"] = expanded["profile"].fillna(0)
    expanded["profile_support"] = expanded["profile_support"].fillna(0)
    grouped = expanded.groupby("row_number", sort=False).agg(
        genre_prior=("genre_prior", "mean"),
        genre_support=("genre_support", "mean"),
        reviewer_genre_preference=("profile", "mean"),
        reviewer_genre_support=("profile_support", "sum"),
    )
    idx = np.arange(len(query))
    result["genre_prior"] = grouped["genre_prior"].reindex(idx).fillna(global_mean).to_numpy()
    result["genre_log_support"] = np.log1p(
        grouped["genre_support"].reindex(idx).fillna(0).to_numpy())
    result["reviewer_genre_preference"] = (
        grouped["reviewer_genre_preference"].reindex(idx).fillna(0).to_numpy())
    result["reviewer_genre_log_support"] = np.log1p(
        grouped["reviewer_genre_support"].reindex(idx).fillna(0).to_numpy())
    return result


def target_features(reference: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    global_mean = float(reference["Score"].mean())
    counts = np.bincount(reference["Score"].to_numpy(dtype=np.int64), minlength=6)[1:6]
    global_probs = counts / counts.sum()
    groups = rating_group_features(reference, query, global_mean, global_probs)
    reviewer_agg = reference.loc[reference["reviewerID"] != ""].groupby("reviewerID")["Score"].agg(
        ["sum", "count"])
    reviewer_mean = ((reviewer_agg["sum"] + GROUP_SMOOTHING["reviewerID"] * global_mean) /
                     (reviewer_agg["count"] + GROUP_SMOOTHING["reviewerID"]))
    return pd.concat([groups, genre_features(reference, query, global_mean, reviewer_mean)],
                     axis=1).astype(np.float32)



def read_original_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    log("Reading ORIGINAL train.csv and test.csv; no earlier research outputs are loaded")
    frame = pd.read_csv(train_path, dtype={"id": str}, low_memory=False)
    request = pd.read_csv(test_path, dtype={"id": str}, low_memory=False)
    required = {"id", "Score", "summary", "reviewText", "genres", *ID_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError(f"train.csv is missing: {sorted(required - set(frame.columns))}")
    if "id" not in request.columns:
        raise ValueError("test.csv is missing an 'id' column")
    for title, obj in (("train.csv", frame), ("test.csv", request)):
        if obj["id"].isna().any():
            raise ValueError(f"{title} contains missing id values")
        obj["id"] = obj["id"].str.strip()
        if (obj["id"] == "").any() or obj["id"].duplicated().any():
            raise ValueError(f"{title} contains blank or duplicate ids")
    missing = ~request["id"].isin(frame["id"])
    if missing.any():
        examples = request.loc[missing, "id"].head().tolist()
        raise ValueError(f"{int(missing.sum())} test IDs have no feature row in train.csv: {examples}")
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce")
    test_ids = set(request["id"])
    # This also excludes accidentally populated scores for requested test IDs.
    labeled = frame.loc[frame["Score"].notna() & ~frame["id"].isin(test_ids)].copy()
    test_rows = frame.set_index("id", drop=False).loc[request["id"]].reset_index(drop=True).copy()
    if labeled.empty or not labeled["Score"].isin(CLASSES).all():
        raise ValueError("No valid labeled examples, or Score is not in {1,2,3,4,5}")
    labeled["Score"] = labeled["Score"].astype(np.int8)
    if (labeled["Score"].value_counts() < N_FOLDS).any() or labeled["Score"].nunique() != 5:
        raise ValueError("Need at least four labeled examples of each rating for 4-fold training")
    for f in (labeled, test_rows):
        for col in (*ID_COLUMNS, "summary", "reviewText", "genres"):
            f[col] = f[col].fillna("").astype(str)
        f["genre_tokens"] = f["genres"].map(parse_genres)
    labeled.reset_index(drop=True, inplace=True)
    test_rows.reset_index(drop=True, inplace=True)
    log(f"Training on {len(labeled):,} labeled reviews; predicting {len(test_rows):,} test reviews")
    log("All supervised label-dependent features use only the labeled training data")
    return labeled, test_rows, request["id"].copy()


def outer_folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED+1
    ).split(np.zeros(len(y)), y))


def build_global_metadata(
    labeled: pd.DataFrame,
    test: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cache: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """41 OOF metadata features for TREE training and FINAL ordinal training.

    Important: all outer held examples get label-dependent features using
    strictly other outer-fold reviews, then ONE global scaler is fitted to
    these OOF rows. This mirrors research meta_A_BC() exactly.
    """
    path_train = cache / "metadata_train_oof.npy"
    path_test = cache / "metadata_test.npy"
    if path_train.exists() and path_test.exists():
        log("Reusing this submission's cross-fitted global metadata")
        x = np.load(path_train)
        z = np.load(path_test)
        if x.shape != (len(labeled), N_META) or z.shape != (len(test), N_META):
            raise ValueError("Cached metadata shape mismatch")
        return x, z

    log("Building 41 metadata features with four-fold out-of-fold target statistics")
    target_oof = None
    names = None
    for fold, (tr, te) in enumerate(folds, 1):
        raw_path = cache / f"outer_fold{fold}_target_raw.npz"
        if raw_path.exists():
            with np.load(raw_path, allow_pickle=False) as data:
                if not np.array_equal(data["indices"], te):
                    raise ValueError("Outer-fold metadata indices don't match")
                vals = data["values"]
                these_names = list(data["names"])
        else:
            source = labeled.iloc[tr]
            target = labeled.iloc[te]
            chunk = target_features(source, target)
            these_names = list(chunk.columns)
            vals = chunk.to_numpy(dtype=np.float32)
            temp = raw_path.with_name(raw_path.name+".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, indices=te, values=vals, names=np.asarray(these_names))
            temp.replace(raw_path)
            del source, target, chunk
        if names is None:
            names = these_names
            target_oof = np.empty((len(labeled), len(names)), dtype=np.float32)
        elif names != these_names:
            raise ValueError("Metadata column ordering changed")
        target_oof[te] = vals
        log(f"  Global metadata outer fold {fold}/4")
        gc.collect()

    assert target_oof is not None and names is not None
    query = target_features(labeled, test)[names].to_numpy(dtype=np.float32)
    raw_train = np.column_stack((target_oof, static_features(labeled).to_numpy(dtype=np.float32)))
    raw_test = np.column_stack((query, static_features(test).to_numpy(dtype=np.float32)))
    if raw_train.shape[1] != N_META:
        raise RuntimeError(f"Expected {N_META} metadata features; got {raw_train.shape[1]}")
    scaler = StandardScaler()
    x = scaler.fit_transform(raw_train)
    z = scaler.transform(raw_test)
    x = (np.clip(x, -6, 6).astype(np.float32) / np.sqrt(float(N_META)))
    z = (np.clip(z, -6, 6).astype(np.float32) / np.sqrt(float(N_META)))
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise RuntimeError("Metadata contains non-finite values")
    save_array(path_train, x)
    save_array(path_test, z)
    save_object(cache / "metadata_scaler.joblib", scaler)
    atomic_json(cache / "metadata_column_order.json", {
        "columns": names + list(static_features(labeled.iloc[:1]).columns),
    })
    del raw_train, raw_test, target_oof, query, scaler
    return x, z


def fold_local_metadata(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    fold: int,
    cached_held_raw: Path,
    held_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FULLY nested metadata for each outer ordinal fold; no own-label leakage."""
    y = training["Score"].to_numpy()
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED+200+fold)
    raw_train = None
    names = None
    for subtr, subval in splitter.split(np.zeros(len(training)), y):
        chunk = target_features(training.iloc[subtr], training.iloc[subval])
        if raw_train is None:
            names = list(chunk.columns)
            raw_train = np.empty((len(training), len(names)), dtype=np.float32)
        raw_train[subval] = chunk[names].to_numpy(dtype=np.float32)
    if cached_held_raw.exists():
        with np.load(cached_held_raw, allow_pickle=False) as data:
            if not np.array_equal(data["indices"], held_indices) or list(data["names"]) != names:
                raise ValueError("Held metadata cache does not match this outer fold")
            raw_validation = data["values"]
    else:
        raw_validation = target_features(training, validation)[names].to_numpy(dtype=np.float32)
    x = np.column_stack((raw_train, static_features(training).to_numpy(dtype=np.float32)))
    z = np.column_stack((raw_validation, static_features(validation).to_numpy(dtype=np.float32)))
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    z = scaler.transform(z)
    x = np.clip(x, -6, 6).astype(np.float32) / np.sqrt(float(N_META))
    z = np.clip(z, -6, 6).astype(np.float32) / np.sqrt(float(N_META))
    return x, z


def fit_text_matrices(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    vectorizer_dir: Path | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Separate-field TF-IDF, exactly as used in the overnight experiment."""
    x_blocks = []
    z_blocks = []
    denominator = math.sqrt(sum(weight*weight for weight in TEXT_WEIGHTS.values()))
    for field in ("summary", "review", "genres"):
        if vectorizer_dir is not None:
            saved = vectorizer_dir / f"tfidf_{field}.joblib"
        else:
            saved = None
        if saved is not None and saved.exists():
            vectorizer = joblib.load(saved)
        else:
            log(f"  TF-IDF {field}: maximum {FIELD_CAPS[field]:,} features")
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=FIELD_CAPS[field],
                min_df=2, sublinear_tf=True, strip_accents="unicode", dtype=np.float32,
            )
            vectorizer.fit(field_text(training, field))
            if saved is not None:
                save_object(saved, vectorizer)
        x = vectorizer.transform(field_text(training, field)).astype(np.float32)
        z = vectorizer.transform(field_text(validation, field)).astype(np.float32)
        weight = TEXT_WEIGHTS[field] / denominator
        x_blocks.append(x.multiply(weight).tocsr())
        z_blocks.append(z.multiply(weight).tocsr())
        log(f"  {field}: {x.shape[1]:,} features")
        del vectorizer, x, z
    X = sparse.hstack(x_blocks, format="csr", dtype=np.float32)
    Z = sparse.hstack(z_blocks, format="csr", dtype=np.float32)
    del x_blocks, z_blocks
    return X, Z


def train_ordinal_heads(
    X: sparse.csr_matrix,
    Z: sparse.csr_matrix,
    y: np.ndarray,
    model_dir: Path,
    label: str,
) -> np.ndarray:
    """Train and save four cutoffs, or reuse ONLY this submission's saved heads."""
    model_dir.mkdir(parents=True, exist_ok=True)
    q = np.zeros((len(Z.indptr)-1, 4), dtype=np.float32)
    for j in range(4):
        model_path = model_dir / f"cutoff{j+1}_C4.joblib"
        score_path = model_dir / f"cutoff{j+1}_pred.npy"
        if score_path.exists():
            scores = np.load(score_path)
            if scores.shape != (Z.shape[0],):
                raise ValueError(f"Wrong cached cutoff prediction shape: {score_path}")
            log(f"  {label}: reused Score>{j+1} predictions")
        else:
            if model_path.exists():
                model = joblib.load(model_path)
                log(f"  {label}: reused Score>{j+1} fitted classifier")
            else:
                log(f"  {label}: FIT Score>{j+1}, C=4 (liblinear)")
                model = LogisticRegression(
                    C=C, solver="liblinear", max_iter=350, tol=1e-4,
                    random_state=SEED,
                )
                model.fit(X, (y > j+1).astype(np.int8))
                save_object(model_path, model)
                log(f"  optimizer iterations={int(model.n_iter_[0])}")
            scores = model.predict_proba(Z)[:, 1].astype(np.float32)
            save_array(score_path, scores)
            del model
        q[:, j] = scores
    return project_q(q)


def build_ordinal_oof(
    labeled: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cache: Path,
) -> np.ndarray:
    path = cache / "ordinal_C4_full_pipeline_oof.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape != (len(labeled), 4):
            raise ValueError("OOF predictions have the wrong shape")
        log("Reusing completed, full-pipeline OOF ordinal predictions")
        return arr
    y = labeled["Score"].to_numpy(dtype=np.int8)
    q = np.empty((len(labeled), 4), dtype=np.float32)
    for fold, (tr, held) in enumerate(folds, 1):
        checkpoint = cache / f"ordinal_oof_fold{fold}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as data:
                if not np.array_equal(data["indices"], held):
                    raise ValueError(f"Cached OOF fold {fold} indices changed")
                scores = data["values"]
            if scores.shape != (len(held), 4):
                raise ValueError("Bad OOF fold prediction shape")
            log(f"Reusing outer ordinal fold {fold}/4")
        else:
            log(f"OOF fold {fold}/4: {len(tr):,} training; {len(held):,} strictly held out")
            train_fold = labeled.iloc[tr].reset_index(drop=True)
            held_fold = labeled.iloc[held].reset_index(drop=True)
            meta_tr, meta_held = fold_local_metadata(
                train_fold, held_fold, fold,
                cache / f"outer_fold{fold}_target_raw.npz", held,
            )
            text_tr, text_held = fit_text_matrices(train_fold, held_fold)
            X = sparse.hstack((text_tr, sparse.csr_matrix(meta_tr * META_WEIGHT)),
                              format="csr", dtype=np.float32)
            Z = sparse.hstack((text_held, sparse.csr_matrix(meta_held * META_WEIGHT)),
                              format="csr", dtype=np.float32)
            scores = train_ordinal_heads(
                X, Z, train_fold["Score"].to_numpy(dtype=np.int8),
                cache / f"ordinal_oof_models_fold{fold}", f"OOF {fold}/4",
            )
            temp = checkpoint.with_name(checkpoint.name + ".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, indices=held, values=scores)
            temp.replace(checkpoint)
            log(f"OOF fold {fold}/4 saved successfully")
            del train_fold, held_fold, X, Z, text_tr, text_held, meta_tr, meta_held
            gc.collect()
        q[held] = scores
    save_array(path, q)
    return q


def final_ordinal(
    labeled: pd.DataFrame, test: pd.DataFrame,
    meta_train: np.ndarray, meta_test: np.ndarray,
    cache: Path,
) -> np.ndarray:
    path = cache / "final_ordinal_C4_test_q.npy"
    if path.exists():
        q = np.load(path)
        if q.shape != (len(test), 4):
            raise ValueError("Cached test prediction shape mismatch")
        log("Reusing final C=4 ordinal predictions")
        return q
    log("Final ordinal model: fitting from ALL labeled reviews")
    txt_x, txt_z = fit_text_matrices(labeled, test, vectorizer_dir=cache / "final_vectorizers")
    X = sparse.hstack((txt_x, sparse.csr_matrix(meta_train * META_WEIGHT)),
                      format="csr", dtype=np.float32)
    Z = sparse.hstack((txt_z, sparse.csr_matrix(meta_test * META_WEIGHT)),
                      format="csr", dtype=np.float32)
    q = train_ordinal_heads(
        X, Z, labeled["Score"].to_numpy(dtype=np.int8),
        cache / "final_ordinal_models", "FINAL ORDINAL",
    )
    save_array(path, q)
    del X, Z, txt_x, txt_z
    gc.collect()
    return q


def stack_features(q: np.ndarray, metadata: np.ndarray) -> np.ndarray:
    """EXACT stack_features(q,meta) from the research code: 53 dense float32 columns."""
    p = q_to_p(q)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-7)), axis=1, keepdims=True)
    sorted_p = np.sort(p, axis=1)
    margin = (sorted_p[:, -1] - sorted_p[:, -2])[:, None]
    expected = (p @ np.arange(1, 6, dtype=np.float32))[:, None]
    core = np.hstack((q, p, expected, entropy, margin)).astype(np.float32)
    X = np.nan_to_num(np.hstack((core, metadata)).astype(np.float32), posinf=0., neginf=0.)
    if X.shape[1] != 53:
        raise ValueError(f"Wrong stack feature count {X.shape[1]} (expected 53)")
    return X


def residual_correction(
    q_oof: np.ndarray, q_test: np.ndarray,
    meta_train: np.ndarray, meta_test: np.ndarray,
    y: np.ndarray, cache: Path, jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    saved = cache / "final_extratrees_delta_test.npy"
    if saved.exists():
        delta = np.load(saved)
        if delta.shape != (len(q_test),):
            raise ValueError("Cached residual output has wrong shape")
        log("Reusing ExtraTrees residual predictions on test reviews")
    else:
        log("Building 53-column stacking representation (OOF train; full-data test)")
        X = stack_features(q_oof, meta_train)
        Z = stack_features(q_test, meta_test)
        expected_train = q_to_p(q_oof) @ np.arange(1, 6, dtype=np.float32)
        residual = y.astype(np.float32) - expected_train
        log(f"OOF residual mean={residual.mean():+.5f}; std={residual.std():.5f}")
        log(f"Training ExtraTrees residual regressor: {TREE_COUNT} trees, "
            f"leaf={TREE_LEAF}, max_features={TREE_MAX_FEATURES}, jobs={jobs}")
        forest = ExtraTreesRegressor(
            n_estimators=TREE_COUNT, min_samples_leaf=TREE_LEAF,
            max_features=TREE_MAX_FEATURES, random_state=SEED+TREE_LEAF,
            n_jobs=jobs,
        )
        forest.fit(X, residual)
        delta = forest.predict(Z).astype(np.float32)
        save_array(saved, delta)
        save_object(cache / "final_extratrees_residual_leaf5.joblib", forest)
        del forest, X, Z, expected_train, residual
        gc.collect()
    q_final = shift_probabilities(q_test, delta * TILT)
    return q_final, delta


def locate_csv(requested: str | None, name: str) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for parent in (ROOT, ROOT / "data"):
        path = parent / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not locate {name}; place it alongside this script, "
                            f"inside data/, or pass --{name.split('.')[0]} PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe full-data residual ExtraTrees Kaggle trainer")
    parser.add_argument("--train", help="Original competition train.csv")
    parser.add_argument("--test", help="Original competition test.csv")
    parser.add_argument("--output", help="Kaggle CSV path; default submissions/...csv")
    parser.add_argument("--cache-dir", help="Dedicated cache for this submission script only")
    parser.add_argument("--jobs", type=int, default=2, help="ExtraTrees workers (default 2 for 16 GB RAM)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    train_path = locate_csv(args.train, "train.csv")
    test_path = locate_csv(args.test, "test.csv")
    out = Path(args.output).expanduser().resolve() if args.output else ROOT / "submissions" / "submission_ordinal_C4_extratrees_residual_leaf5.csv"
    cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else ROOT / "residual_submission_cache"
    if cache == train_path.parent or cache == test_path.parent or cache == ROOT:
        parser.error("--cache-dir must be its own dedicated directory, not a CSV or script parent")

    start = time.perf_counter()
    log("START: C=4 ordinal logistic + OOF ExtraTrees residual correction")
    log("Frozen development-selected thresholds: " + repr(THRESHOLDS.tolist()))
    verify_manifest(cache, train_path, test_path)
    labeled, test, test_ids = read_original_data(train_path, test_path)
    y = labeled["Score"].to_numpy(dtype=np.int8)
    folds = outer_folds(y)

    # Stage 1: cross-fitted ID features for BOTH clean OOF training and final LR.
    meta_train, meta_test = build_global_metadata(labeled, test, folds, cache)
    # Stage 2: all four cutoffs fit 4 times with FULLY nested pipeline per outer fold.
    q_oof = build_ordinal_oof(labeled, folds, cache)
    log(f"OOF ordinal probabilities created for {len(q_oof):,} labeled reviews")
    # Stage 3: final ordinal model learns from every labeled review, not old research fits.
    q_test = final_ordinal(labeled, test, meta_train, meta_test, cache)
    # Stage 4: correct residual errors learned ONLY from out-of-fold predictions.
    q_final, delta = residual_correction(q_oof, q_test, meta_train, meta_test, y, cache, args.jobs)
    predictions = predict_threshold(q_final)
    baseline = predict_threshold(q_test, np.full(4, .5, dtype=np.float32))
    if predictions.shape != (len(test_ids),) or not np.isin(predictions, CLASSES).all():
        raise RuntimeError("Invalid submission predictions")

    # Write an exact two-column Kaggle file, preserving test.csv ID order.
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids.to_numpy(), "Score": predictions.astype(int)})
    tmp = out.with_name(out.name + ".tmp")
    submission.to_csv(tmp, index=False)
    tmp.replace(out)
    report = {
        "architecture": "C4 ordinal + OOF ExtraTrees residual tilt",
        "seed": SEED,
        "n_labeled": int(len(labeled)),
        "n_test": int(len(test)),
        "features": {"text": FIELD_CAPS, "metadata": N_META, "metadata_weight": META_WEIGHT},
        "ordinal": {"C": C, "solver": "liblinear", "outer_oof_folds": N_FOLDS},
        "residual": {"trees": TREE_COUNT, "min_samples_leaf": TREE_LEAF,
                     "max_features": TREE_MAX_FEATURES, "tilt": TILT, "sigma": TILT_SIGMA,
                     "test_correction_mean": float(np.mean(delta)),
                     "test_correction_std": float(np.std(delta))},
        "thresholds": THRESHOLDS.tolist(),
        "prediction_counts": {str(k): int(np.sum(predictions == k)) for k in CLASSES},
        "baseline_prediction_counts": {str(k): int(np.sum(baseline == k)) for k in CLASSES},
        "fraction_changed_vs_plain_C4": float(np.mean(predictions != baseline)),
        "submission_path": str(out),
        "elapsed_hours": (time.perf_counter()-start)/3600.,
        "validation_score": "Not estimated here; frozen architecture selected on earlier research development data",
    }
    atomic_json(cache / "submission_diagnostics.json", report)
    log(f"Saved {len(submission):,} predictions: {out}")
    log("Predicted rating counts: " + str(report["prediction_counts"]))
    log(f"Changed vs plain C4: {100*report['fraction_changed_vs_plain_C4']:.2f}%")
    log(f"Research-only diagnostics: {cache / 'submission_diagnostics.json'}")
    log(f"Done in {report['elapsed_hours']:.2f} hours")


if __name__ == "__main__":
    main()
