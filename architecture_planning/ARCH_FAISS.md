# Architecture Option B — FAISS Vector Index

## Summary

Replace the linear cosine loop with **FAISS** (Facebook AI Similarity Search), the fastest open-source ANN library available. FAISS operates on raw numpy arrays and delivers sub-millisecond search even at millions of vectors. It does not store metadata — a lightweight SQLite sidecar handles labels, paths, and per-feature arrays.

---

## Why FAISS

| Property | Value |
|---|---|
| Setup | pip install faiss-cpu (or faiss-gpu) |
| Search | IVF + HNSW — fastest ANN available in Python |
| Metadata | Not supported — needs sidecar (SQLite) |
| Scale | Millions of vectors, production-grade |
| Migration effort | Medium — need to manage index + SQLite separately |

---

## Storage Layout

```
audio_db/
├── clap.index          ← FAISS index for CLAP vectors  (binary, mmap-able)
├── librosa.index       ← FAISS index for librosa combined vectors
└── metadata.db         ← SQLite: labels, paths, hashes, per-feature blobs
```

### SQLite Schema

```sql
CREATE TABLE audio (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    faiss_idx   INTEGER NOT NULL,        -- row index in FAISS index
    label       TEXT NOT NULL,
    path        TEXT NOT NULL,
    hash        TEXT NOT NULL UNIQUE,
    -- per-feature arrays stored as BLOB (np.ndarray → bytes)
    mfcc        BLOB,
    chroma      BLOB,
    spectral_contrast BLOB,
    rhythm      BLOB,
    zcr         BLOB,
    rms         BLOB
);
```

FAISS indices are integer-addressed; `faiss_idx` maps a FAISS row back to a SQLite record.

---

## Component Design

### Index types by DB size

| DB Size | Recommended Index | Notes |
|---|---|---|
| < 1 000 | `IndexFlatIP` | Exact cosine, no training needed |
| 1 000 – 100 000 | `IndexIVFFlat` | Train on sample, ~10–50× speedup |
| > 100 000 | `IndexIVFPQ` | Compressed, ~100× speedup, slight accuracy loss |

For most audio libraries (< 10 000 files), `IndexFlatIP` is exact and still very fast.

---

### `audio_engine.py` changes

```python
import faiss
import sqlite3

CLAP_INDEX_FILE    = DB_DIR / "clap.index"
LIBROSA_INDEX_FILE = DB_DIR / "librosa.index"
METADATA_DB_FILE   = DB_DIR / "metadata.db"

def _get_db_conn():
    conn = sqlite3.connect(METADATA_DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_idx INTEGER NOT NULL,
            label TEXT, path TEXT, hash TEXT UNIQUE,
            mfcc BLOB, chroma BLOB, spectral_contrast BLOB,
            rhythm BLOB, zcr BLOB, rms BLOB
        )
    """)
    return conn

def _load_or_create_index(path, dim):
    if Path(path).exists():
        return faiss.read_index(str(path))
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine on L2-normalised vectors
    return index
```

#### Adding audio

```python
def add_audio_to_database(audio_path, label, clap_model, model_type):
    file_hash = md5(audio_path)
    conn = _get_db_conn()

    if conn.execute("SELECT 1 FROM audio WHERE hash=?", (file_hash,)).fetchone():
        return False, "File already exists."

    bundle = build_embedding(audio_path, clap_model, model_type)
    lib_feats = bundle["librosa"]

    # Add to CLAP FAISS index
    if "clap" in bundle:
        clap_index = _load_or_create_index(CLAP_INDEX_FILE, len(bundle["clap"]))
        clap_vec = bundle["clap"].reshape(1, -1).astype(np.float32)
        clap_index.add(clap_vec)
        faiss.write_index(clap_index, str(CLAP_INDEX_FILE))

    # Add to librosa FAISS index
    lib_index = _load_or_create_index(LIBROSA_INDEX_FILE, len(lib_feats["combined"]))
    lib_vec = lib_feats["combined"].reshape(1, -1).astype(np.float32)
    lib_index.add(lib_vec)
    faiss.write_index(lib_index, str(LIBROSA_INDEX_FILE))

    faiss_idx = lib_index.ntotal - 1   # last added row

    # Persist metadata + per-feature blobs
    conn.execute("""
        INSERT INTO audio (faiss_idx, label, path, hash, mfcc, chroma, spectral_contrast, rhythm, zcr, rms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        faiss_idx, label, audio_path, file_hash,
        lib_feats["mfcc"].tobytes(), lib_feats["chroma"].tobytes(),
        lib_feats["spectral_contrast"].tobytes(), lib_feats["rhythm"].tobytes(),
        lib_feats["zcr"].tobytes(), lib_feats["rms"].tobytes(),
    ))
    conn.commit()
    return True, f"'{label}' added."
```

#### Searching

```python
def search(query_path, clap_model, model_type, top_k=5, min_similarity=0.0):
    query_bundle = build_embedding(query_path, clap_model, model_type)
    conn = _get_db_conn()

    if model_type == "clap" and "clap" in query_bundle:
        # Step 1: CLAP ANN search — get top_k*3 candidates
        clap_index = faiss.read_index(str(CLAP_INDEX_FILE))
        clap_vec = query_bundle["clap"].reshape(1, -1).astype(np.float32)
        clap_scores, clap_idxs = clap_index.search(clap_vec, top_k * 3)

        # Step 2: fetch librosa vectors for candidates only
        lib_index = faiss.read_index(str(LIBROSA_INDEX_FILE))
        candidate_idxs = clap_idxs[0].tolist()

        results = []
        for i, faiss_idx in enumerate(candidate_idxs):
            row = conn.execute("SELECT * FROM audio WHERE faiss_idx=?", (faiss_idx,)).fetchone()
            if not row:
                continue
            # Reconstruct librosa combined from index
            lib_vec = lib_index.reconstruct(faiss_idx)
            lib_sim = float(np.dot(query_bundle["librosa"]["combined"], lib_vec))
            clap_sim = float(clap_scores[0][i])
            overall = 0.60 * clap_sim + 0.40 * lib_sim

            if overall < min_similarity:
                continue

            # Reconstruct per-feature vectors from SQLite blobs for reason generation
            db_feats = {
                "mfcc":              np.frombuffer(row["mfcc"],              dtype=np.float64),
                "chroma":            np.frombuffer(row["chroma"],            dtype=np.float64),
                "spectral_contrast": np.frombuffer(row["spectral_contrast"], dtype=np.float64),
                "rhythm":            np.frombuffer(row["rhythm"],            dtype=np.float64),
                "zcr":               np.frombuffer(row["zcr"],               dtype=np.float64),
                "rms":               np.frombuffer(row["rms"],               dtype=np.float64),
            }
            reasons = explain_match(query_bundle["librosa"], db_feats)
            results.append({"label": row["label"], "path": row["path"],
                            "similarity": overall, "reasons": reasons})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]
```

---

## Data Flow

```
Upload audio
     │
     ▼
build_embedding()
     │
     ├──► faiss clap.index.add(clap_vec)        ← append to binary index
     ├──► faiss librosa.index.add(combined_vec) ← append to binary index
     └──► sqlite INSERT (label, path, hash, per-feature blobs)

─────────────────────────────────────────────────────────

Query audio
     │
     ▼
build_embedding()  (query)
     │
     ▼
clap.index.search(query_clap, k=top_k*3)     ← returns (scores, faiss_row_ids)
     │
     ▼
sqlite SELECT WHERE faiss_idx IN (...)        ← fetch only candidates
     │
     ▼
librosa.index.reconstruct(faiss_idx)         ← get librosa vec for each candidate
     │
     ▼
Blend CLAP + librosa scores
     │
     ▼
explain_match() from SQLite blobs → top_k
     │
     ▼
Return results
```

---

## Performance Comparison

| Operation | Current (pickle) | FAISS + SQLite |
|---|---|---|
| Load DB (1000 files) | ~200–500 ms | ~5–10 ms (mmap index) |
| Search (1000 files) | O(n) Python loop | O(log n) IVF / exact flat |
| Search (100 000 files) | seconds | < 5 ms |
| Add single file | full pickle read+write | O(1) index append + 1 SQL insert |
| GPU support | No | Yes (faiss-gpu) |

---

## Dependencies

```
faiss-cpu>=1.7.4      # or faiss-gpu for CUDA
# sqlite3 is stdlib — no extra install
```

---

## Migration Steps

1. `pip install faiss-cpu`
2. Create SQLite schema on first run
3. Replace `build_embedding` + `save_database` with FAISS `.add()` + SQLite `INSERT`
4. Replace linear search loop with `index.search()` + SQLite candidate fetch
5. Reconstruct per-feature vectors from SQLite blobs for `explain_match()`
6. Delete old `embeddings.pkl` and `metadata.json`

---

## Tradeoffs

| Pro | Con |
|---|---|
| Fastest ANN search available | Two storage systems to manage (FAISS + SQLite) |
| GPU acceleration available | No native metadata filtering in FAISS |
| Memory-mappable index (low RAM) | Index rebuild needed if vectors are deleted |
| Exact or approximate modes | More complex code than ChromaDB |
| Production-battle-tested (Meta) | Deletes require index rebuild (FAISS doesn't support removes natively) |
