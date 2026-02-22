# audio-similarity
Here's a walkthrough of how the whole system works, from start to finish.

---

### Big Picture

You have two files working together: `audio_engine.py` does all the math and ML, and `app.py` is just the interface that calls it. When you interact with the UI, `app.py` collects your inputs and hands them off to `audio_engine.py`, which does the heavy lifting and returns results back to the UI.

---

### Step 1 — Model Loading (once at startup)

When the app starts, it tries to load **CLAP** (a model by LAION that understands audio semantically, similar to how CLIP understands images). If CLAP isn't installed, it quietly falls back to using librosa features only. This is cached so the model only loads once per session.

---

### Step 2 — Adding Audio to the Database

When you upload a file and click "Add to Database":

1. The file is saved to a temp path on disk
2. `audio_engine.py` computes an **embedding bundle** for it — this is two things combined:
   - A **CLAP vector**: one big number array representing the "meaning" of the whole audio (like a fingerprint of what the audio sounds like semantically)
   - Six **librosa feature vectors**, each capturing something specific about the audio

The six librosa features are what power the reason explanations later:

| Feature | What it captures |
|---|---|
| MFCC | Timbre — the "colour" of the sound, vocal quality |
| Chroma | Which musical notes/pitches are present — the melody |
| Spectral Contrast | Difference between loud and quiet frequencies — background vs foreground |
| Rhythm/Tempo | Speed of the beat and how regular it is |
| Zero-Crossing Rate | How often the signal crosses zero — indicates vocal vs non-vocal |
| RMS Energy | Overall loudness and how it changes over time |

3. All of this gets saved to disk — the vectors go into `embeddings.pkl` (pickle file) and the labels/paths go into `metadata.json`.

---

### Step 3 — Searching

When you upload a query audio and click Search:

1. The same embedding process runs on your query file — CLAP vector + 6 librosa features
2. It then loops through every entry in the database and computes a **similarity score** between your query and each stored file using **cosine similarity** (measures the angle between two vectors — closer to 1 means more similar)
3. The overall score is a weighted blend: **60% CLAP** (semantic understanding) + **40% librosa combined** (acoustic features)
4. Results are sorted highest to lowest and the top K are returned

---

### Step 4 — Reason Generation

This is the interesting part. After computing the overall score, the system goes back and compares each of the 6 feature vectors *individually* between the query and the matched file. If any individual feature has a cosine similarity above **80%**, it gets reported as a reason. So for example:

- Chroma vectors are similar → "🎵 Tune / Melody matched"
- MFCC vectors are similar → "🎤 Tone / Vocals matched"
- Rhythm vectors are similar → "🥁 Rhythm / Tempo matched"

This is how it tells you *why* something matched, not just *that* it matched.

---

### Data Flow Summary

```
Upload audio
     │
     ▼
extract_librosa_features()  ──► 6 named vectors (mfcc, chroma, rhythm, etc.)
extract_clap_embedding()    ──► 1 semantic vector
     │
     ▼
save to embeddings.pkl + metadata.json  (database)

─────────────────────────────────────────────

Query audio
     │
     ▼
Same extraction runs on query
     │
     ▼
For each DB entry:
  cosine(query_clap, db_clap) * 0.6
+ cosine(query_librosa_combined, db_librosa_combined) * 0.4
= overall_score
     │
     ▼
For each DB entry:
  compare each of 6 features individually
  → if score > 80%, add to reasons list
     │
     ▼
Sort by overall_score, return top K with reasons
```

---

### Why This Approach Works Well

The two-layer design is intentional. CLAP alone would give you a similarity score but no explanation. Librosa alone would give you reasons but miss high-level semantic similarity (like two different recordings of the same song). Together, CLAP handles "does this sound like the same kind of audio" and librosa handles "here's specifically what's similar about it."
