"""
audio_engine.py — Qdrant backend
==================================
Same public API as the original audio_engine.py but backed by Qdrant,
a purpose-built vector database with native support for multiple named
vectors per record, rich JSON payloads, and payload-filtered search.

All vectors (CLAP + 6 librosa features + combined) live on **one Qdrant
point** — no metadata flattening or blob serialisation required.

Storage layout
--------------
audio_db/
└── qdrant_storage/        — Qdrant persistent local directory
    └── collection: "audio"
          Each point:
            id      : int (first 8 hex chars of MD5 hash)
            vectors : {clap, librosa_combined, mfcc, chroma,
                       spectral_contrast, rhythm, zcr, rms}
            payload : {label, path, hash}
"""

import hashlib
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_DIR      = Path("audio_db")
QDRANT_PATH = str(DB_DIR / "qdrant_storage")
COLLECTION  = "audio"

SAMPLE_RATE      = 22050
TOP_K            = 5
REASON_THRESHOLD = 0.80

DB_DIR.mkdir(exist_ok=True)

FEATURE_META: dict[str, tuple[str, str]] = {
    "chroma":            ("🎵 Tune / Melody",        "Similar harmonic / melodic structure"),
    "mfcc":              ("🎤 Tone / Vocals",         "Similar voice timbre or instrument tone"),
    "rhythm":            ("🥁 Rhythm / Tempo",        "Similar beat pattern and tempo"),
    "spectral_contrast": ("🌊 Texture / Background",  "Similar frequency texture and background sound"),
    "zcr":               ("🗣️ Vocal Presence",         "Similar vocal vs non-vocal content"),
    "rms":               ("🔊 Loudness Pattern",       "Similar energy and loudness dynamics"),
}

# Named vector sizes for collection creation
_VECTOR_SIZES: dict[str, int] = {
    "clap":               512,
    "librosa_combined":   47,
    "mfcc":               20,
    "chroma":             12,
    "spectral_contrast":  7,
    "rhythm":             4,
    "zcr":                2,
    "rms":                2,
}

# ──────────────────────────────────────────────────────────────────────────────
# QDRANT CLIENT  (lazy singleton)
# ──────────────────────────────────────────────────────────────────────────────

_qdrant = None


def _get_qdrant():
    """Initialise QdrantClient and ensure the collection exists on first call."""
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_qdrant)
    return _qdrant


def _ensure_collection(client) -> None:
    """Create the 'audio' collection with all named vectors if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                name: VectorParams(size=size, distance=Distance.COSINE)
                for name, size in _VECTOR_SIZES.items()
            },
        )


def _md5(audio_path: str) -> str:
    return hashlib.md5(Path(audio_path).read_bytes()).hexdigest()


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
    except Exception:
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
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=20)

    features: dict[str, np.ndarray] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features["mfcc"] = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    features["chroma"] = np.mean(chroma, axis=1)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["spectral_contrast"] = np.mean(contrast, axis=1)

    tempo, beats  = librosa.beat.beat_track(y=y, sr=sr)
    onset_env     = librosa.onset.onset_strength(y=y, sr=sr)
    features["rhythm"] = np.array([
        float(tempo),
        float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0,
        float(np.mean(onset_env)),
        float(np.std(onset_env)),
    ])

    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr"] = np.array([float(np.mean(zcr)), float(np.std(zcr))])

    rms = librosa.feature.rms(y=y)
    features["rms"] = np.array([float(np.mean(rms)), float(np.std(rms))])

    concat = np.concatenate([
        features["mfcc"], features["chroma"], features["spectral_contrast"],
        features["rhythm"], features["zcr"], features["rms"],
    ])
    norm = np.linalg.norm(concat)
    features["combined"] = concat / (norm + 1e-8)

    return features


def extract_clap_embedding(model, audio_path: str) -> Optional[np.ndarray]:
    """Extract an L2-normalised CLAP semantic embedding. Returns None on failure."""
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
# DATABASE  (Qdrant)
# ──────────────────────────────────────────────────────────────────────────────

def load_database() -> tuple[list, list]:
    """
    Return ([], metadata_list).
    Vectors live in Qdrant; only the lightweight payload is returned here
    for use by the UI (listing, counting, etc.).
    """
    client = _get_qdrant()
    records, _ = client.scroll(
        collection_name=COLLECTION,
        with_payload=True,
        with_vectors=False,
        limit=10_000,
    )
    metadata = [
        {"label": r.payload["label"], "path": r.payload["path"], "hash": r.payload["hash"]}
        for r in records
    ]
    return [], metadata


def save_database(embeddings: list, metadata: list) -> None:
    """No-op — Qdrant persists automatically on every upsert."""
    pass


def clear_database() -> None:
    """Delete and recreate the Qdrant collection."""
    client = _get_qdrant()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    _ensure_collection(client)


def add_audio_to_database(
    audio_path: str,
    label: str,
    clap_model,
    model_type: str,
) -> tuple[bool, str]:
    """
    Embed an audio file and upsert it as a single Qdrant point with all
    named vectors attached.

    Returns
    -------
    (success, message)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

    client    = _get_qdrant()
    file_hash = _md5(audio_path)

    # Duplicate check via payload filter
    existing, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="hash", match=MatchValue(value=file_hash))
        ]),
        limit=1,
    )
    if existing:
        return False, "File already exists in the database."

    bundle = build_embedding(audio_path, clap_model, model_type)
    lib    = bundle["librosa"]

    # All named vectors on one point — no metadata hacks needed
    vectors: dict = {
        "librosa_combined":   lib["combined"].tolist(),
        "mfcc":               lib["mfcc"].tolist(),
        "chroma":             lib["chroma"].tolist(),
        "spectral_contrast":  lib["spectral_contrast"].tolist(),
        "rhythm":             lib["rhythm"].tolist(),
        "zcr":                lib["zcr"].tolist(),
        "rms":                lib["rms"].tolist(),
    }
    if "clap" in bundle:
        vectors["clap"] = bundle["clap"].tolist()

    # Qdrant IDs must be integers or UUIDs — convert first 8 hex chars of hash
    point_id = int(file_hash[:8], 16)

    point = PointStruct(
        id=point_id,
        vector=vectors,
        payload={"label": label, "path": str(audio_path), "hash": file_hash},
    )
    client.upsert(collection_name=COLLECTION, points=[point])

    _, all_meta = load_database()
    return True, f"'{label}' added. Database now has {len(all_meta)} file(s)."


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
    for key, (feat_label, desc) in FEATURE_META.items():
        qv, dv = query_feats.get(key), db_feats.get(key)
        if qv is None or dv is None:
            continue
        sim = _cosine(qv, dv)
        if sim >= threshold:
            reasons.append((feat_label, desc, sim))
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
    Find the top-K most similar audio files using Qdrant HNSW search.

    Scoring
    -------
    When CLAP is available  : 60 % CLAP semantic  +  40 % librosa_combined
    Librosa-only            : 100 % librosa_combined

    Qdrant with COSINE distance returns scores that are cosine similarities.
    Per-feature vectors are returned directly in `hit.vector` — no
    reconstruction step required.
    """
    from qdrant_client.models import NamedVector

    client       = _get_qdrant()
    query_bundle = build_embedding(query_path, clap_model, model_type)

    info = client.get_collection(COLLECTION)
    if (info.points_count or 0) == 0:
        return []

    score_threshold = min_similarity if min_similarity > 0.0 else None
    results: list[dict] = []

    if model_type == "clap" and "clap" in query_bundle:
        # Semantic ANN search, over-fetch for re-ranking with librosa
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=NamedVector(name="clap", vector=query_bundle["clap"].tolist()),
            limit=top_k * 2,
            with_vectors=True,
            with_payload=True,
            score_threshold=score_threshold,
        )
        for hit in hits:
            lib_vec = np.array(hit.vector["librosa_combined"])
            lib_sim = _cosine(query_bundle["librosa"]["combined"], lib_vec)
            overall = 0.60 * hit.score + 0.40 * lib_sim
            if overall < min_similarity:
                continue
            # Per-feature vectors come back directly — no reconstruction needed
            db_feats = {
                feat: np.array(hit.vector[feat])
                for feat in ["mfcc", "chroma", "spectral_contrast", "rhythm", "zcr", "rms"]
                if feat in hit.vector
            }
            reasons = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      hit.payload["label"],
                "path":       hit.payload["path"],
                "similarity": overall,
                "reasons":    reasons,
            })

    else:
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=NamedVector(
                name="librosa_combined",
                vector=query_bundle["librosa"]["combined"].tolist(),
            ),
            limit=top_k,
            with_vectors=True,
            with_payload=True,
            score_threshold=score_threshold,
        )
        for hit in hits:
            db_feats = {
                feat: np.array(hit.vector[feat])
                for feat in ["mfcc", "chroma", "spectral_contrast", "rhythm", "zcr", "rms"]
                if feat in hit.vector
            }
            reasons = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      hit.payload["label"],
                "path":       hit.payload["path"],
                "similarity": hit.score,
                "reasons":    reasons,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
    return results[:top_k]
