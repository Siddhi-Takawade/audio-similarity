"""
audio_engine.py
================
All ML computation for the Audio Similarity Search system.
Handles:
  - Feature extraction (librosa)
  - Semantic embeddings (CLAP)
  - Database persistence (embeddings + metadata)
  - Similarity search & reason generation

No Streamlit imports here — pure Python / ML.
"""

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_DIR          = Path("audio_db")
EMBEDDINGS_FILE = DB_DIR / "embeddings.pkl"
METADATA_FILE   = DB_DIR / "metadata.json"

SAMPLE_RATE       = 22050
TOP_K             = 5
REASON_THRESHOLD  = 0.80   # min per-feature cosine sim to count as a match reason

DB_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE METADATA  (used by both engine and UI for labels / descriptions)
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_META: dict[str, tuple[str, str]] = {
    "chroma":            ("🎵 Tune / Melody",        "Similar harmonic / melodic structure"),
    "mfcc":              ("🎤 Tone / Vocals",         "Similar voice timbre or instrument tone"),
    "rhythm":            ("🥁 Rhythm / Tempo",        "Similar beat pattern and tempo"),
    "spectral_contrast": ("🌊 Texture / Background",  "Similar frequency texture and background sound"),
    "zcr":               ("🗣️ Vocal Presence",         "Similar vocal vs non-vocal content"),
    "rms":               ("🔊 Loudness Pattern",       "Similar energy and loudness dynamics"),
}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_clap_model() -> tuple:
    """
    Attempt to load CLAP (LAION) for semantic audio embeddings.

    Returns
    -------
    (model, model_type)
        model_type is "clap" on success, "librosa" on failure.
    """
    try:
        from msclap import CLAP  # type: ignore
        model = CLAP(version="2023", use_cuda=False)
        return model, "clap"
    except Exception as exc:
        return None, "librosa"


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_librosa_features(audio_path: str) -> dict[str, np.ndarray]:
    """
    Extract named, interpretable feature vectors from an audio file.

    Features extracted
    ------------------
    mfcc              : timbre / vocal quality / tone colour  (20-d)
    chroma            : melody / harmonic content             (12-d)
    spectral_contrast : foreground vs background texture      (7-d)
    rhythm            : tempo + beat regularity + onset stats (4-d)
    zcr               : zero-crossing rate – vocal indicator  (2-d)
    rms               : loudness / energy dynamics            (2-d)
    combined          : L2-normalised concatenation of all    (47-d)

    Returns
    -------
    dict mapping feature name → numpy array
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)   # remove leading/trailing silence

    features: dict[str, np.ndarray] = {}

    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features["mfcc"] = np.mean(mfcc, axis=1)

    # 2. Chroma (CQT-based for better pitch accuracy)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    features["chroma"] = np.mean(chroma, axis=1)

    # 3. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast"] = np.mean(contrast, axis=1)

    # 4. Rhythm / tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
    features["rhythm"] = np.array([
        float(tempo),
        float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0,
        float(np.mean(onset_env)),
        float(np.std(onset_env)),
    ])

    # 5. Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr"] = np.array([float(np.mean(zcr)), float(np.std(zcr))])

    # 6. RMS energy
    rms = librosa.feature.rms(y=y)
    features["rms"] = np.array([float(np.mean(rms)), float(np.std(rms))])

    # Combined normalised vector
    concat = np.concatenate([
        features["mfcc"],
        features["chroma"],
        features["spectral_contrast"],
        features["rhythm"],
        features["zcr"],
        features["rms"],
    ])
    norm = np.linalg.norm(concat)
    features["combined"] = concat / (norm + 1e-8)

    return features


def extract_clap_embedding(model, audio_path: str) -> np.ndarray | None:
    """
    Extract a L2-normalised CLAP semantic embedding.

    Returns None if extraction fails.
    """
    try:
        embeddings = model.get_audio_embeddings([audio_path])
        vec = np.array(embeddings[0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)
    except Exception:
        return None


def build_embedding(audio_path: str, clap_model, model_type: str) -> dict:
    """
    Build the full embedding bundle for one audio file.

    Returns
    -------
    dict with keys:
        "librosa"  : dict of named feature arrays  (always present)
        "clap"     : 1-D ndarray                   (present only when model_type == "clap")
    """
    bundle: dict = {"librosa": extract_librosa_features(audio_path)}
    if model_type == "clap" and clap_model is not None:
        vec = extract_clap_embedding(clap_model, audio_path)
        if vec is not None:
            bundle["clap"] = vec
    return bundle


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE  (flat-file, pickle + JSON)
# ──────────────────────────────────────────────────────────────────────────────

def load_database() -> tuple[list, list]:
    """
    Load embeddings and metadata from disk.

    Returns
    -------
    (embeddings, metadata)
        embeddings : list of embedding bundles (dicts)
        metadata   : list of dicts {label, path, hash}
    """
    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return embeddings, metadata
    return [], []


def save_database(embeddings: list, metadata: list) -> None:
    """Persist embeddings and metadata to disk."""
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def clear_database() -> None:
    """Delete all stored embeddings and metadata."""
    for fp in (EMBEDDINGS_FILE, METADATA_FILE):
        if fp.exists():
            fp.unlink()


def add_audio_to_database(
    audio_path: str,
    label: str,
    clap_model,
    model_type: str,
) -> tuple[bool, str]:
    """
    Embed an audio file and add it to the persistent database.

    Returns
    -------
    (success, message)
    """
    embeddings, metadata = load_database()

    # Duplicate check by MD5
    file_hash = hashlib.md5(Path(audio_path).read_bytes()).hexdigest()
    if any(m.get("hash") == file_hash for m in metadata):
        return False, "File already exists in the database."

    bundle = build_embedding(audio_path, clap_model, model_type)
    embeddings.append(bundle)
    metadata.append({"label": label, "path": str(audio_path), "hash": file_hash})
    save_database(embeddings, metadata)

    return True, f"'{label}' added. Database now has {len(metadata)} file(s)."


# ──────────────────────────────────────────────────────────────────────────────
# SIMILARITY & REASON GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def explain_match(
    query_feats: dict[str, np.ndarray],
    db_feats: dict[str, np.ndarray],
    threshold: float = REASON_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """
    Compare per-feature similarity to generate human-readable match reasons.

    Returns
    -------
    list of (display_label, description, similarity_score) sorted descending.
    Only features whose similarity meets `threshold` are included.
    """
    reasons: list[tuple[str, str, float]] = []
    for key, (label, desc) in FEATURE_META.items():
        qv, dv = query_feats.get(key), db_feats.get(key)
        if qv is None or dv is None:
            continue
        sim = _cosine(qv, dv)
        if sim >= threshold:
            reasons.append((label, desc, sim))
    reasons.sort(key=lambda x: x[2], reverse=True)
    return reasons


def search(
    query_path: str,
    clap_model,
    model_type: str,
    top_k: int = TOP_K,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Find the top-K most similar audio files in the database.

    Scoring
    -------
    When CLAP is available  : 60 % CLAP semantic  +  40 % librosa combined
    Librosa-only            : 100 % librosa combined

    Parameters
    ----------
    query_path      : path to the query audio file
    clap_model      : loaded CLAP model or None
    model_type      : "clap" | "librosa"
    top_k           : maximum results to return
    min_similarity  : filter out results below this threshold (0–1)

    Returns
    -------
    list of result dicts, each containing:
        rank        : int
        label       : str
        path        : str
        similarity  : float  (0–1)
        reasons     : list of (label, desc, score) tuples
    """
    embeddings, metadata = load_database()
    if not embeddings:
        return []

    query_bundle = build_embedding(query_path, clap_model, model_type)
    results: list[dict] = []

    for db_bundle, meta in zip(embeddings, metadata):
        # Overall similarity score
        if model_type == "clap" and "clap" in query_bundle and "clap" in db_bundle:
            clap_sim = _cosine(query_bundle["clap"], db_bundle["clap"])
            lib_sim  = _cosine(
                query_bundle["librosa"]["combined"],
                db_bundle["librosa"]["combined"],
            )
            overall = 0.60 * clap_sim + 0.40 * lib_sim
        else:
            overall = _cosine(
                query_bundle["librosa"]["combined"],
                db_bundle["librosa"]["combined"],
            )

        if overall < min_similarity:
            continue

        reasons = explain_match(query_bundle["librosa"], db_bundle["librosa"])

        results.append({
            "rank":       0,           # assigned below after sorting
            "label":      meta["label"],
            "path":       meta["path"],
            "similarity": overall,
            "reasons":    reasons,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1

    return results[:top_k]
