"""
app.py
=======
Streamlit UI for the Audio Similarity Search system.
All ML computation is delegated to audio_engine.py.

Run:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

import audio_engine as engine

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Audio Similarity Search",
    page_icon="🎧",
    layout="wide",
)

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]

# ──────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES  (model loaded once per session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading audio model…")
def get_model():
    """Load CLAP (or fall back to librosa-only) once and cache it."""
    return engine.load_clap_model()


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_upload_to_temp(uploaded_file) -> str:
    """Write an UploadedFile to a named temp file and return its path."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def similarity_badge(score: float) -> str:
    pct = score * 100
    if pct >= 75:
        return f"🟢 {pct:.1f}%"
    if pct >= 50:
        return f"🟡 {pct:.1f}%"
    return f"🔴 {pct:.1f}%"


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(model_type: str) -> None:
    _, metadata = engine.load_database()

    badge = "🟢 CLAP  (Semantic + Feature)" if model_type == "clap" else "🟡 Librosa Features Only"
    st.sidebar.markdown(f"**Model:** {badge}")
    st.sidebar.divider()
    st.sidebar.markdown(f"**Database:** {len(metadata)} file(s)")

    if metadata:
        for m in metadata:
            st.sidebar.markdown(f"- {m['label']}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — MANAGE DATABASE
# ──────────────────────────────────────────────────────────────────────────────

def render_database_tab(clap_model, model_type: str) -> None:
    st.header("📂 Manage Database")
    st.info("Upload audio files to build your searchable library. "
            "Supported formats: WAV · MP3 · FLAC · OGG · M4A")

    uploaded_files = st.file_uploader(
        "Choose audio files",
        type=AUDIO_TYPES,
        accept_multiple_files=True,
        key="db_uploader",
    )

    # Label inputs (one per file)
    if uploaded_files:
        st.markdown("**Set a label for each file:**")
        for uf in uploaded_files:
            st.text_input(
                f"Label — {uf.name}",
                value=Path(uf.name).stem,
                key=f"lbl_{uf.name}",
            )

        if st.button("➕ Add to Database", type="primary"):
            for uf in uploaded_files:
                label = st.session_state.get(f"lbl_{uf.name}", Path(uf.name).stem)
                tmp_path = save_upload_to_temp(uf)

                with st.spinner(f"Processing '{uf.name}'…"):
                    ok, msg = engine.add_audio_to_database(
                        tmp_path, label, clap_model, model_type
                    )
                os.unlink(tmp_path)

                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

            st.rerun()

    # Show current contents
    _, metadata = engine.load_database()
    if metadata:
        st.divider()
        st.subheader("Current Database")

        for i, m in enumerate(metadata):
            col_name, col_path = st.columns([3, 4])
            col_name.markdown(f"**{i + 1}. {m['label']}**")
            col_path.caption(Path(m["path"]).name)

        st.divider()
        if st.button("🗑️ Clear Entire Database", type="secondary"):
            engine.clear_database()
            st.success("Database cleared.")
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def render_search_tab(clap_model, model_type: str) -> None:
    st.header("🔍 Search Similar Audio")

    _, metadata = engine.load_database()

    if not metadata:
        st.warning("Your database is empty. Go to **Manage Database** and add some audio files first.")
        return

    query_file = st.file_uploader(
        "Upload your query audio",
        type=AUDIO_TYPES,
        key="search_uploader",
    )

    col1, col2 = st.columns(2)
    top_k     = col1.slider("Results to show",       1,    min(10, len(metadata)), min(engine.TOP_K, len(metadata)))
    threshold = col2.slider("Min similarity (%)",     0,    100,                    0,                              5)

    if query_file and st.button("🔍 Search", type="primary"):
        tmp_path = save_upload_to_temp(query_file)

        with st.spinner("Analysing query and searching database…"):
            results = engine.search(
                query_path     = tmp_path,
                clap_model     = clap_model,
                model_type     = model_type,
                top_k          = top_k,
                min_similarity = threshold / 100,
            )
        os.unlink(tmp_path)

        st.divider()
        _render_results(results)


def _render_results(results: list[dict]) -> None:
    if not results:
        st.error("No matches found above the similarity threshold.")
        return

    st.subheader(f"Found {len(results)} match(es)")

    for r in results:
        badge   = similarity_badge(r["similarity"])
        header  = f"#{r['rank']}  {badge}  —  **{r['label']}**"

        with st.expander(header, expanded=(r["rank"] == 1)):
            # Overall score bar
            st.progress(
                r["similarity"],
                text=f"Overall similarity: {r['similarity']*100:.1f}%",
            )

            # Per-feature reasons
            if r["reasons"]:
                st.markdown("**Why it matched:**")
                for feat_label, feat_desc, feat_sim in r["reasons"]:
                    st.markdown(
                        f"- {feat_label} &nbsp;&nbsp; `{feat_sim*100:.0f}%` — {feat_desc}"
                    )
            else:
                st.caption(
                    "Match is based on the overall audio profile; "
                    "no single feature strongly dominates."
                )

            # Playback (only if file is still accessible on disk)
            if Path(r["path"]).exists():
                st.audio(r["path"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ──────────────────────────────────────────────────────────────────────────────

def render_info_tab() -> None:
    st.header("ℹ️ How It Works")

    st.markdown("""
    ### Architecture

    The system uses **two complementary embedding layers**:

    **1. CLAP** *(optional — install `msclap`)*  
    LAION's Contrastive Language–Audio Pretraining model encodes the full audio
    holistically — vocals, instruments, ambient sounds — into a single semantic vector.
    Contributes **60 %** of the final similarity score.

    **2. Librosa feature vectors**  
    Six named, interpretable feature vectors are extracted and compared individually.
    Contributes **40 %** of the final score and powers the **reason** explanations.

    ---

    ### Feature → Reason Mapping

    | Feature | What it captures | Shown as |
    |---|---|---|
    | Chroma (CQT) | Musical notes & harmony | 🎵 Tune / Melody |
    | MFCC | Timbre & vocal tone colour | 🎤 Tone / Vocals |
    | Beat / Tempo | Rhythm speed & regularity | 🥁 Rhythm / Tempo |
    | Spectral Contrast | Foreground vs background | 🌊 Texture / Background |
    | Zero-Crossing Rate | Vocal vs non-vocal content | 🗣️ Vocal Presence |
    | RMS Energy | Loudness dynamics | 🔊 Loudness Pattern |

    A feature is reported as a reason only when its per-feature cosine similarity
    exceeds **80 %**.

    ---

    ### Tips for Best Results
    - Keep audio clips consistent in length (10–60 s works well).
    - Silence is automatically trimmed before embedding.
    - Use descriptive labels so results are easy to interpret.
    - Install CLAP for significantly better results on music and speech.

    ```bash
    pip install msclap   # downloads ~400 MB model on first use, then offline
    ```
    """)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎧 Audio Similarity Search")
    st.caption("Build a database of audio files, then search for matches with explanations.")

    clap_model, model_type = get_model()
    render_sidebar(model_type)

    tab_db, tab_search, tab_info = st.tabs(
        ["📂 Manage Database", "🔍 Search", "ℹ️ How It Works"]
    )

    with tab_db:
        render_database_tab(clap_model, model_type)

    with tab_search:
        render_search_tab(clap_model, model_type)

    with tab_info:
        render_info_tab()


if __name__ == "__main__":
    main()
