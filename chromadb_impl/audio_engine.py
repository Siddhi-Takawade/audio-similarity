"""
audio_engine.py — ChromaDB backend
====================================
Same public API as the original audio_engine.py but backed by ChromaDB
for persistent HNSW-indexed vector storage. No server required — ChromaDB
runs in-process and persists data to a local directory.
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

DB_DIR     = Path("audio_db")
CHROMA_DIR = DB_DIR / "chroma"

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

# Dimensions for each per-feature array (used for metadata serialisation)
_PER_FEATURE_DIMS: dict[str, int] = {
    "mfcc": 20, "chroma": 12, "spectral_contrast": 7,
    "rhythm": 4, "zcr": 2, "rms": 2,
}

# ──────────────────────────────────────────────────────────────────────────────
# CHROMADB CLIENT  (lazy singleton)
# ──────────────────────────────────────────────────────────────────────────────

_client   = None
_clap_col = None
_lib_col  = None


def _get_collections():
    """Initialise ChromaDB client and collections on first call."""
    global _client, _clap_col, _lib_col
    if _client is None:
        import chromadb
        _client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _clap_col = _client.get_or_create_collection(
            "clap_vectors", metadata={"hnsw:space": "cosine"}
        )
        _lib_col = _client.get_or_create_collection(
            "librosa_vectors", metadata={"hnsw:space": "cosine"}
        )
    return _clap_col, _lib_col


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
# METADATA SERIALISATION HELPERS
# ChromaDB metadata values must be scalar — per-feature arrays are flattened.
# ──────────────────────────────────────────────────────────────────────────────

def _feats_to_meta(lib_feats: dict[str, np.ndarray]) -> dict:
    """Flatten per-feature numpy arrays into individual scalar metadata keys."""
    meta: dict = {}
    for feat_name in _PER_FEATURE_DIMS:
        vec = lib_feats.get(feat_name)
        if vec is not None:
            for i, v in enumerate(vec.tolist()):
                meta[f"{feat_name}_{i}"] = float(v)
    return meta


def _meta_to_feats(meta: dict) -> dict[str, np.ndarray]:
    """Reconstruct per-feature numpy arrays from scalar metadata keys."""
    feats: dict[str, np.ndarray] = {}
    for feat_name, dims in _PER_FEATURE_DIMS.items():
        keys = [f"{feat_name}_{i}" for i in range(dims)]
        if all(k in meta for k in keys):
            feats[feat_name] = np.array([meta[k] for k in keys])
    return feats


def _md5(audio_path: str) -> str:
    return hashlib.md5(Path(audio_path).read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE  (ChromaDB)
# ──────────────────────────────────────────────────────────────────────────────

def load_database() -> tuple[list, list]:
    """
    Return ([], metadata_list).
    Embeddings live in ChromaDB; only the lightweight metadata is returned here
    for use by the UI (listing, counting, etc.).
    """
    _, lib_col = _get_collections()
    result = lib_col.get(include=["metadatas"])
    metadata = [
        {"label": m["label"], "path": m["path"], "hash": m["hash"]}
        for m in (result["metadatas"] or [])
    ]
    return [], metadata


def save_database(embeddings: list, metadata: list) -> None:
    """No-op — ChromaDB persists automatically on every write."""
    pass


def clear_database() -> None:
    """Delete both ChromaDB collections and recreate them empty."""
    global _client, _clap_col, _lib_col
    import chromadb
    # Re-use or create client
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for name in ("clap_vectors", "librosa_vectors"):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    _client   = client
    _clap_col = client.get_or_create_collection(
        "clap_vectors", metadata={"hnsw:space": "cosine"}
    )
    _lib_col  = client.get_or_create_collection(
        "librosa_vectors", metadata={"hnsw:space": "cosine"}
    )


def add_audio_to_database(
    audio_path: str,
    label: str,
    clap_model,
    model_type: str,
) -> tuple[bool, str]:
    """
    Embed an audio file and add it to the ChromaDB collections.

    Returns
    -------
    (success, message)
    """
    clap_col, lib_col = _get_collections()
    file_hash = _md5(audio_path)

    # Duplicate check via metadata filter (no full scan needed)
    existing = lib_col.get(where={"hash": file_hash})
    if existing["ids"]:
        return False, "File already exists in the database."

    bundle    = build_embedding(audio_path, clap_model, model_type)
    lib_feats = bundle["librosa"]
    doc_id    = file_hash   # stable unique ID

    base_meta = {"label": label, "path": str(audio_path), "hash": file_hash}

    # Store CLAP vector (if available)
    if "clap" in bundle:
        clap_col.add(
            ids=[doc_id],
            embeddings=[bundle["clap"].tolist()],
            metadatas=[base_meta],
        )

    # Store librosa combined vector; per-feature arrays flattened into metadata
    lib_meta = {**base_meta, **_feats_to_meta(lib_feats)}
    lib_col.add(
        ids=[doc_id],
        embeddings=[lib_feats["combined"].tolist()],
        metadatas=[lib_meta],
    )

    count = lib_col.count()
    return True, f"'{label}' added. Database now has {count} file(s)."


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
    Find the top-K most similar audio files using ChromaDB HNSW ANN search.

    Scoring
    -------
    When CLAP is available  : 60 % CLAP semantic  +  40 % librosa combined
    Librosa-only            : 100 % librosa combined

    ChromaDB returns cosine *distance* in [0, 2]; similarity = 1 - distance.
    """
    clap_col, lib_col = _get_collections()

    if lib_col.count() == 0:
        return []

    query_bundle = build_embedding(query_path, clap_model, model_type)
    results: list[dict] = []

    if model_type == "clap" and "clap" in query_bundle and clap_col.count() > 0:
        # Over-fetch from CLAP index, then re-rank with librosa similarity
        n_fetch = min(top_k * 2, clap_col.count())
        clap_results = clap_col.query(
            query_embeddings=[query_bundle["clap"].tolist()],
            n_results=n_fetch,
            include=["metadatas", "distances"],
        )
        candidate_ids    = clap_results["ids"][0]
        clap_distances   = clap_results["distances"][0]

        # Batch-fetch librosa vectors + metadata for all candidates
        lib_results = lib_col.get(
            ids=candidate_ids,
            include=["embeddings", "metadatas"],
        )
        lib_map = {
            rid: (emb, meta)
            for rid, emb, meta in zip(
                lib_results["ids"],
                lib_results["embeddings"],
                lib_results["metadatas"],
            )
        }

        for cid, dist in zip(candidate_ids, clap_distances):
            if cid not in lib_map:
                continue
            lib_emb, lib_meta = lib_map[cid]
            clap_sim = 1.0 - dist   # ChromaDB cosine distance = 1 - similarity
            lib_sim  = float(np.dot(
                query_bundle["librosa"]["combined"],
                np.array(lib_emb, dtype=np.float64),
            ))
            overall = 0.60 * clap_sim + 0.40 * lib_sim
            if overall < min_similarity:
                continue
            db_feats = _meta_to_feats(lib_meta)
            reasons  = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      lib_meta["label"],
                "path":       lib_meta["path"],
                "similarity": overall,
                "reasons":    reasons,
            })

    else:
        # Librosa-only ANN search
        n_fetch = min(top_k, lib_col.count())
        lib_results = lib_col.query(
            query_embeddings=[query_bundle["librosa"]["combined"].tolist()],
            n_results=n_fetch,
            include=["metadatas", "distances"],
        )
        for meta, dist in zip(lib_results["metadatas"][0], lib_results["distances"][0]):
            sim = 1.0 - dist
            if sim < min_similarity:
                continue
            db_feats = _meta_to_feats(meta)
            reasons  = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      meta["label"],
                "path":       meta["path"],
                "similarity": sim,
                "reasons":    reasons,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
    return results[:top_k]
