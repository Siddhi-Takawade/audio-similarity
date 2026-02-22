# Architecture Option A — ChromaDB Vector Store

## Summary

Replace the monolithic `embeddings.pkl` with **ChromaDB**, an embedded local vector database with built-in HNSW indexing. No server required — ChromaDB runs in-process and persists data to a local directory.

---

## Why ChromaDB

| Property | Value |
|---|---|
| Setup | Zero — pip install, no server |
| Search | HNSW (Approximate Nearest Neighbor) |
| Metadata | Stored alongside vectors natively |
| Scale | Handles ~1M vectors comfortably |
| Migration effort | Low — minimal changes to `audio_engine.py` |

---

## Storage Layout

```
audio_db/
└── chroma/                        ← ChromaDB persistent directory
    ├── clap_vectors/              ← Collection 1: CLAP semantic embeddings
    │     vectors   : (N × D_clap) HNSW-indexed
    │     metadata  : {label, path, hash}
    └── librosa_vectors/           ← Collection 2: librosa combined + per-feature
          vectors   : (N × 47) HNSW-indexed
          metadata  : {label, path, hash,
                       mfcc_0..19, chroma_0..11,
                       spectral_contrast_0..6,
                       rhythm_0..3, zcr_0..1, rms_0..1}
```

Individual librosa feature arrays (mfcc, chroma, etc.) are serialised into the metadata payload so reason generation can be done without a separate file.

---

## Component Design

### `audio_engine.py` changes

```python
import chromadb

DB_DIR = Path("audio_db")
_chroma_client = chromadb.PersistentClient(path=str(DB_DIR / "chroma"))
clap_col   = _chroma_client.get_or_create_collection("clap_vectors",   metadata={"hnsw:space": "cosine"})
lib_col    = _chroma_client.get_or_create_collection("librosa_vectors", metadata={"hnsw:space": "cosine"})
```

#### Adding audio

```python
def add_audio_to_database(audio_path, label, clap_model, model_type):
    file_hash = md5(audio_path)

    # Duplicate check — query metadata, no full load needed
    existing = clap_col.get(where={"hash": file_hash})
    if existing["ids"]:
        return False, "File already exists."

    bundle = build_embedding(audio_path, clap_model, model_type)
    doc_id = file_hash  # use hash as stable ID

    # Store CLAP vector
    if "clap" in bundle:
        clap_col.add(
            ids=[doc_id],
            embeddings=[bundle["clap"].tolist()],
            metadatas=[{"label": label, "path": audio_path, "hash": file_hash}]
        )

    # Store librosa combined vector + per-feature in metadata
    lib_meta = {"label": label, "path": audio_path, "hash": file_hash}
    for feat_name, vec in bundle["librosa"].items():
        if feat_name != "combined":
            for i, v in enumerate(vec.tolist()):
                lib_meta[f"{feat_name}_{i}"] = v
    lib_col.add(
        ids=[doc_id],
        embeddings=[bundle["librosa"]["combined"].tolist()],
        metadatas=[lib_meta]
    )
    return True, f"'{label}' added."
```

#### Searching

```python
def search(query_path, clap_model, model_type, top_k=5, min_similarity=0.0):
    query_bundle = build_embedding(query_path, clap_model, model_type)

    if model_type == "clap" and "clap" in query_bundle:
        # ANN search on CLAP collection
        clap_results = clap_col.query(
            query_embeddings=[query_bundle["clap"].tolist()],
            n_results=top_k * 2,        # over-fetch, re-rank with librosa
            include=["metadatas", "distances"]
        )
        # Re-rank: blend CLAP distance with librosa similarity
        candidates = clap_results["ids"][0]
        lib_results = lib_col.get(ids=candidates, include=["embeddings", "metadatas"])
        # compute weighted score and reason generation only on top_k*2 candidates
    else:
        lib_results = lib_col.query(
            query_embeddings=[query_bundle["librosa"]["combined"].tolist()],
            n_results=top_k,
            include=["embeddings", "metadatas", "distances"]
        )

    # Reason generation — reconstruct per-feature vectors from metadata payload
    # Sort → return top_k
```

---

## Data Flow

```
Upload audio
     │
     ▼
build_embedding()
     │
     ├──► clap_col.add(id, clap_vec, metadata)       ← HNSW indexed
     └──► lib_col.add(id, combined_vec, metadata)     ← HNSW indexed
                                                         (per-feature in payload)

─────────────────────────────────────────────────────────

Query audio
     │
     ▼
build_embedding()  (query)
     │
     ▼
clap_col.query(clap_vec, n=top_k*2)   ← ANN, O(log N)
     │
     ▼
lib_col.get(candidate_ids)            ← fetch only top_k*2, not all N
     │
     ▼
Blend scores → sort → take top_k
     │
     ▼
explain_match() on top_k only         ← NOT on all N entries
     │
     ▼
Return results with reasons
```

---

## Performance Comparison

| Operation | Current (pickle) | ChromaDB |
|---|---|---|
| Load DB (1000 files) | ~200–500 ms | ~0 ms (persistent index) |
| Search (1000 files) | O(n) loop | O(log n) HNSW |
| Add single file | full read + full write | single insert |
| Reason generation | all N entries | top K only |

---

## Dependencies

```
chromadb>=0.4.0
```

---

## Migration Steps

1. `pip install chromadb`
2. Replace `load_database()` / `save_database()` with ChromaDB client init
3. Replace `add_audio_to_database()` with `clap_col.add()` + `lib_col.add()`
4. Replace linear search loop with `col.query()`
5. Reconstruct per-feature arrays from metadata for reason generation
6. Delete old `embeddings.pkl` and `metadata.json`

---

## Tradeoffs

| Pro | Con |
|---|---|
| Zero server setup | ChromaDB metadata values must be scalar (floats/strings) — per-feature vectors need to be flattened into individual keys |
| Metadata + vectors co-located | Slightly more complex metadata serialisation for reason generation |
| Built-in duplicate detection via metadata filter | First query slightly slower (HNSW index warm-up) |
| Easy to inspect / debug | Limited filtering compared to Qdrant |
