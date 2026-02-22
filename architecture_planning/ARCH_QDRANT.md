# Architecture Option C — Qdrant Vector Store

## Summary

Replace `embeddings.pkl` with **Qdrant**, a purpose-built vector database that supports rich payload filtering, multiple named vectors per record, and both local embedded mode and a full server mode. Qdrant is the most feature-complete option — it handles CLAP and all 6 librosa feature vectors as separate named vectors on a single "point", with no metadata serialisation tricks required.

---

## Why Qdrant

| Property | Value |
|---|---|
| Setup | Local embedded mode (no server) or Docker server |
| Search | HNSW with payload filtering |
| Metadata | Rich JSON payload per vector point |
| Multiple vectors | One record can carry multiple named vectors natively |
| Scale | Millions of vectors; horizontal scaling in server mode |
| Migration effort | Medium-High — richer API surface |

---

## Storage Layout

```
audio_db/
└── qdrant_storage/          ← Qdrant persistent local directory
    └── collection: "audio"
          Each point contains:
            id       : md5 hash of file (stable unique ID)
            vectors  : {
                "clap"              : float[] (D_clap dims),
                "librosa_combined"  : float[] (47 dims),
                "mfcc"              : float[] (20 dims),
                "chroma"            : float[] (12 dims),
                "spectral_contrast" : float[] (7 dims),
                "rhythm"            : float[] (4 dims),
                "zcr"               : float[] (2 dims),
                "rms"               : float[] (2 dims),
            }
            payload  : {label, path, hash}
```

All vectors (CLAP + 6 librosa features + combined) live on **one point**. This eliminates the need to flatten per-feature arrays into metadata keys or blobs.

---

## Component Design

### `audio_engine.py` changes

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, NamedVector,
    PointStruct, Filter, FieldCondition, MatchValue
)

QDRANT_PATH   = str(DB_DIR / "qdrant_storage")
COLLECTION    = "audio"

# Lazy singleton client
_qdrant: QdrantClient | None = None

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_qdrant)
    return _qdrant

def _ensure_collection(client: QdrantClient) -> None:
    """Create collection with named vectors if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "clap":               VectorParams(size=512,  distance=Distance.COSINE),
                "librosa_combined":   VectorParams(size=47,   distance=Distance.COSINE),
                "mfcc":               VectorParams(size=20,   distance=Distance.COSINE),
                "chroma":             VectorParams(size=12,   distance=Distance.COSINE),
                "spectral_contrast":  VectorParams(size=7,    distance=Distance.COSINE),
                "rhythm":             VectorParams(size=4,    distance=Distance.COSINE),
                "zcr":                VectorParams(size=2,    distance=Distance.COSINE),
                "rms":                VectorParams(size=2,    distance=Distance.COSINE),
            }
        )
```

#### Adding audio

```python
def add_audio_to_database(audio_path, label, clap_model, model_type):
    client = _get_qdrant()
    file_hash = md5(audio_path)

    # Duplicate check
    results = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="hash", match=MatchValue(value=file_hash))]),
        limit=1
    )
    if results[0]:
        return False, "File already exists."

    bundle = build_embedding(audio_path, clap_model, model_type)
    lib = bundle["librosa"]

    vectors = {
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

    point = PointStruct(
        id=int(file_hash[:8], 16),          # convert first 8 hex chars to int ID
        vector=vectors,
        payload={"label": label, "path": audio_path, "hash": file_hash}
    )
    client.upsert(collection_name=COLLECTION, points=[point])
    return True, f"'{label}' added."
```

#### Searching

```python
def search(query_path, clap_model, model_type, top_k=5, min_similarity=0.0):
    client = _get_qdrant()
    query_bundle = build_embedding(query_path, clap_model, model_type)

    if model_type == "clap" and "clap" in query_bundle:
        # Search by CLAP vector — semantic ANN
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=NamedVector(name="clap", vector=query_bundle["clap"].tolist()),
            limit=top_k * 2,
            with_vectors=True,           # fetch all named vectors for re-ranking + reasons
            with_payload=True,
            score_threshold=min_similarity
        )
    else:
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=NamedVector(name="librosa_combined",
                                     vector=query_bundle["librosa"]["combined"].tolist()),
            limit=top_k,
            with_vectors=True,
            with_payload=True,
            score_threshold=min_similarity
        )

    results = []
    for hit in hits:
        # Re-rank: blend CLAP score with librosa combined score
        if model_type == "clap":
            lib_vec = np.array(hit.vector["librosa_combined"])
            lib_sim = _cosine(query_bundle["librosa"]["combined"], lib_vec)
            overall = 0.60 * hit.score + 0.40 * lib_sim
        else:
            overall = hit.score

        # Reconstruct per-feature dicts directly from returned named vectors
        db_feats = {
            feat: np.array(hit.vector[feat])
            for feat in ["mfcc", "chroma", "spectral_contrast", "rhythm", "zcr", "rms"]
            if feat in hit.vector
        }
        reasons = explain_match(query_bundle["librosa"], db_feats)

        results.append({
            "label":      hit.payload["label"],
            "path":       hit.payload["path"],
            "similarity": overall,
            "reasons":    reasons,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    for i, r in enumerate(results[:top_k]):
        r["rank"] = i + 1
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
     ▼
client.upsert(PointStruct(
    id      = hash_int,
    vectors = {clap, librosa_combined, mfcc, chroma,
               spectral_contrast, rhythm, zcr, rms},
    payload = {label, path, hash}
))                             ← single write, all vectors on one point

─────────────────────────────────────────────────────────

Query audio
     │
     ▼
build_embedding()  (query)
     │
     ▼
client.search(named_vector="clap", k=top_k*2)
         ← HNSW ANN, returns hits with all named vectors + payload
     │
     ▼
Re-rank: blend clap score + librosa_combined score
     │
     ▼
explain_match() using named vectors from hit directly
         ← no reconstruction step, vectors are already there
     │
     ▼
Sort → top_k with reasons
```

---

## Bonus: Payload Filtering

Qdrant supports filtering during search — unique to this option:

```python
# Search only within a specific category (if you tag files)
client.search(
    collection_name=COLLECTION,
    query_vector=NamedVector(name="clap", vector=...),
    query_filter=Filter(must=[
        FieldCondition(key="category", match=MatchValue(value="music"))
    ]),
    limit=top_k
)
```

Useful for: separating music vs speech vs SFX, per-user databases, date-based filtering, etc.

---

## Performance Comparison

| Operation | Current (pickle) | Qdrant (local) |
|---|---|---|
| Load DB (1000 files) | ~200–500 ms | ~0 ms (persistent index) |
| Search (1000 files) | O(n) loop | O(log n) HNSW |
| Add single file | full read + write | single upsert |
| Per-feature reconstruction | deserialise pickle | returned directly in hit |
| Filtered search | not supported | native payload filter |
| Server mode (scale-out) | not supported | Docker / cloud |

---

## Dependencies

```
qdrant-client>=1.9.0
```

---

## Local vs Server Mode

```python
# Local embedded (no server — default, same as ChromaDB/FAISS)
client = QdrantClient(path="audio_db/qdrant_storage")

# Server mode (for production or shared access)
client = QdrantClient(host="localhost", port=6333)
# docker run -p 6333:6333 qdrant/qdrant
```

Switching from local to server is a one-line change.

---

## Migration Steps

1. `pip install qdrant-client`
2. Define collection with named vectors (run once)
3. Replace `add_audio_to_database()` with `client.upsert()`
4. Replace linear search with `client.search()` with named vector
5. Use returned `hit.vector` directly for reason generation — no reconstruction needed
6. Delete old `embeddings.pkl` and `metadata.json`

---

## Tradeoffs

| Pro | Con |
|---|---|
| Named multi-vectors per point — no metadata hacks | Heavier dependency than ChromaDB or FAISS alone |
| Per-feature vectors returned directly in search hit | Qdrant IDs must be integers or UUIDs (hash conversion needed) |
| Native payload filtering | Slight overhead in local mode vs raw FAISS |
| Zero-change upgrade to server mode | More verbose collection setup |
| Best for future features (tags, filters, categories) | Overkill if DB stays small (< 500 files) |
