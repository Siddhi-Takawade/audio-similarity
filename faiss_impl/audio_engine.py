"""
audio_engine.py — FAISS + SQLite backend
==========================================
Same public API as the original audio_engine.py but backed by FAISS
for fast ANN vector search and SQLite for metadata + per-feature blobs.

Storage layout
--------------
audio_db/
├── clap.index      — FAISS IndexFlatIP for CLAP vectors    (binary, mmap-able)
├── librosa.index   — FAISS IndexFlatIP for librosa combined vectors
└── metadata.db     — SQLite: labels, paths, hashes, per-feature blobs
"""

import hashlib
import sqlite3
import warnings
from pathlib import Path
from typing import Optional

import faiss
import librosa
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DB_DIR             = Path("audio_db")
CLAP_INDEX_FILE    = DB_DIR / "clap.index"
LIBROSA_INDEX_FILE = DB_DIR / "librosa.index"
METADATA_DB_FILE   = DB_DIR / "metadata.db"

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

# ──────────────────────────────────────────────────────────────────────────────
# SQLITE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _get_db_conn() -> sqlite3.Connection:
    """Open (or create) the SQLite database, ensure schema exists."""
    conn = sqlite3.connect(METADATA_DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audio (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_idx         INTEGER NOT NULL,
            label             TEXT    NOT NULL,
            path              TEXT    NOT NULL,
            hash              TEXT    NOT NULL UNIQUE,
            mfcc              BLOB,
            chroma            BLOB,
            spectral_contrast BLOB,
            rhythm            BLOB,
            zcr               BLOB,
            rms               BLOB
        )
    """)
    conn.commit()
    return conn


# ──────────────────────────────────────────────────────────────────────────────
# FAISS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_or_create_index(path: Path, dim: int) -> faiss.Index:
    """Load an existing FAISS index from disk, or create a new IndexFlatIP."""
    if path.exists():
        return faiss.read_index(str(path))
    # IndexFlatIP computes inner product — cosine similarity for L2-normalised vectors
    return faiss.IndexFlatIP(dim)


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
# DATABASE  (FAISS + SQLite)
# ──────────────────────────────────────────────────────────────────────────────

def load_database() -> tuple[list, list]:
    """
    Return ([], metadata_list).
    Vectors live in FAISS index files; only the lightweight metadata is returned
    here for use by the UI (listing, counting, etc.).
    """
    conn = _get_db_conn()
    rows = conn.execute("SELECT label, path, hash FROM audio ORDER BY id").fetchall()
    conn.close()
    metadata = [{"label": r["label"], "path": r["path"], "hash": r["hash"]} for r in rows]
    return [], metadata


def save_database(embeddings: list, metadata: list) -> None:
    """No-op — FAISS index and SQLite are updated incrementally on each add."""
    pass


def clear_database() -> None:
    """Delete all SQLite records and FAISS index files."""
    conn = _get_db_conn()
    conn.execute("DELETE FROM audio")
    conn.commit()
    conn.close()
    for fp in (CLAP_INDEX_FILE, LIBROSA_INDEX_FILE):
        if fp.exists():
            fp.unlink()


def add_audio_to_database(
    audio_path: str,
    label: str,
    clap_model,
    model_type: str,
) -> tuple[bool, str]:
    """
    Embed an audio file and add it to the FAISS indices + SQLite metadata store.

    Returns
    -------
    (success, message)
    """
    conn      = _get_db_conn()
    file_hash = _md5(audio_path)

    if conn.execute("SELECT 1 FROM audio WHERE hash=?", (file_hash,)).fetchone():
        conn.close()
        return False, "File already exists in the database."

    bundle    = build_embedding(audio_path, clap_model, model_type)
    lib_feats = bundle["librosa"]

    # Add combined librosa vector to FAISS index
    lib_vec   = lib_feats["combined"].reshape(1, -1).astype(np.float32)
    lib_index = _load_or_create_index(LIBROSA_INDEX_FILE, lib_vec.shape[1])
    lib_index.add(lib_vec)
    faiss.write_index(lib_index, str(LIBROSA_INDEX_FILE))
    faiss_idx = lib_index.ntotal - 1   # row index in the FAISS flat array

    # Add CLAP vector to its own FAISS index (if available)
    if "clap" in bundle:
        clap_vec   = bundle["clap"].reshape(1, -1).astype(np.float32)
        clap_index = _load_or_create_index(CLAP_INDEX_FILE, clap_vec.shape[1])
        clap_index.add(clap_vec)
        faiss.write_index(clap_index, str(CLAP_INDEX_FILE))

    # Persist metadata + per-feature blobs to SQLite
    conn.execute("""
        INSERT INTO audio
            (faiss_idx, label, path, hash,
             mfcc, chroma, spectral_contrast, rhythm, zcr, rms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        faiss_idx, label, str(audio_path), file_hash,
        lib_feats["mfcc"].astype(np.float64).tobytes(),
        lib_feats["chroma"].astype(np.float64).tobytes(),
        lib_feats["spectral_contrast"].astype(np.float64).tobytes(),
        lib_feats["rhythm"].astype(np.float64).tobytes(),
        lib_feats["zcr"].astype(np.float64).tobytes(),
        lib_feats["rms"].astype(np.float64).tobytes(),
    ))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM audio").fetchone()[0]
    conn.close()

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


def _row_to_feats(row: sqlite3.Row) -> dict[str, np.ndarray]:
    """Deserialise per-feature numpy arrays from SQLite BLOB columns."""
    return {
        "mfcc":              np.frombuffer(row["mfcc"],              dtype=np.float64),
        "chroma":            np.frombuffer(row["chroma"],            dtype=np.float64),
        "spectral_contrast": np.frombuffer(row["spectral_contrast"], dtype=np.float64),
        "rhythm":            np.frombuffer(row["rhythm"],            dtype=np.float64),
        "zcr":               np.frombuffer(row["zcr"],               dtype=np.float64),
        "rms":               np.frombuffer(row["rms"],               dtype=np.float64),
    }


def search(
    query_path: str,
    clap_model,
    model_type: str,
    top_k: int = TOP_K,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Find the top-K most similar audio files using FAISS ANN search.

    Scoring
    -------
    When CLAP is available  : 60 % CLAP (IndexFlatIP inner product)
                            + 40 % librosa combined (re-computed from index)
    Librosa-only            : 100 % librosa combined

    FAISS IndexFlatIP returns inner product scores; for L2-normalised vectors
    this equals cosine similarity directly.
    """
    conn  = _get_db_conn()
    total = conn.execute("SELECT COUNT(*) FROM audio").fetchone()[0]
    if total == 0:
        conn.close()
        return []

    query_bundle = build_embedding(query_path, clap_model, model_type)
    results: list[dict] = []

    if model_type == "clap" and "clap" in query_bundle and CLAP_INDEX_FILE.exists():
        clap_index = faiss.read_index(str(CLAP_INDEX_FILE))
        lib_index  = faiss.read_index(str(LIBROSA_INDEX_FILE))
        clap_vec   = query_bundle["clap"].reshape(1, -1).astype(np.float32)
        n_fetch    = min(top_k * 3, clap_index.ntotal)

        clap_scores, clap_idxs = clap_index.search(clap_vec, n_fetch)

        for score, faiss_idx in zip(clap_scores[0], clap_idxs[0]):
            if faiss_idx < 0:
                continue
            row = conn.execute(
                "SELECT * FROM audio WHERE faiss_idx=?", (int(faiss_idx),)
            ).fetchone()
            if not row:
                continue

            # Reconstruct librosa combined vector from FAISS for blended score
            lib_vec  = lib_index.reconstruct(int(faiss_idx)).astype(np.float64)
            lib_sim  = float(np.dot(query_bundle["librosa"]["combined"], lib_vec))
            clap_sim = float(score)
            overall  = 0.60 * clap_sim + 0.40 * lib_sim

            if overall < min_similarity:
                continue

            db_feats = _row_to_feats(row)
            reasons  = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      row["label"],
                "path":       row["path"],
                "similarity": overall,
                "reasons":    reasons,
            })

    else:
        if not LIBROSA_INDEX_FILE.exists():
            conn.close()
            return []
        lib_index = faiss.read_index(str(LIBROSA_INDEX_FILE))
        lib_vec   = query_bundle["librosa"]["combined"].reshape(1, -1).astype(np.float32)
        n_fetch   = min(top_k, lib_index.ntotal)

        scores, idxs = lib_index.search(lib_vec, n_fetch)

        for score, faiss_idx in zip(scores[0], idxs[0]):
            if faiss_idx < 0:
                continue
            row = conn.execute(
                "SELECT * FROM audio WHERE faiss_idx=?", (int(faiss_idx),)
            ).fetchone()
            if not row:
                continue
            if float(score) < min_similarity:
                continue

            db_feats = _row_to_feats(row)
            reasons  = explain_match(query_bundle["librosa"], db_feats)
            results.append({
                "rank":       0,
                "label":      row["label"],
                "path":       row["path"],
                "similarity": float(score),
                "reasons":    reasons,
            })

    conn.close()
    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
    return results[:top_k]
