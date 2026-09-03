import os
import tempfile
import streamlit as st

# ============================================================
#  SECURE SECRETS LOADING (Cloud‑First)
#  This must run BEFORE any module that reads os.getenv().
# ============================================================
try:
    os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]
    os.environ["WHISPER_MODEL"]   = st.secrets["WHISPER_MODEL"]
    os.environ["SARVAM_API_KEY"]  = st.secrets["SARVAM_API_KEY"]
    os.environ["SARVAM_STT_MODE"] = st.secrets["SARVAM_STT_MODE"]
except KeyError as e:
    st.error(f"❌ Missing secret: {e}. Please set it in Streamlit Cloud secrets.")
    st.stop()
# ============================================================

# Now import the rest – they will find the env vars via os.getenv()
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ----------------------------------------------------------------------
#  Page configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------
#  Session state initialisation
# ----------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_file" not in st.session_state:
    st.session_state.temp_file = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

def cleanup_temp_file():
    if st.session_state.temp_file and os.path.exists(st.session_state.temp_file):
        try:
            os.remove(st.session_state.temp_file)
        except OSError:
            pass
    st.session_state.temp_file = None
    st.session_state.uploaded_name = None

# ----------------------------------------------------------------------
#  Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 AI Video Assistant")
    st.caption("Transcribe → Summarize → Extract → Chat")

    source_mode = st.radio("Input source", ["YouTube URL", "Upload file", "Local path"])
    source = None

    if source_mode == "YouTube URL":
        url = st.text_input("URL", placeholder="https://www.youtube.com/watch?v=...")
        if url.strip() and url.strip().startswith(("http://", "https://")):
            source = url.strip()
    elif source_mode == "Upload file":
        uploaded = st.file_uploader(
            "Audio / video file",
            type=["mp4", "mkv", "webm", "mov", "mp3", "wav", "m4a", "flac", "ogg"],
        )
        if uploaded is not None:
            if st.session_state.uploaded_name != uploaded.name:
                cleanup_temp_file()
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                st.session_state.temp_file = tmp.name
                st.session_state.uploaded_name = uploaded.name
            source = st.session_state.temp_file
    else:  # Local path
        path = st.text_input("File path", placeholder="D:/meetings/standup.mp4")
        if path.strip() and os.path.exists(path.strip()):
            source = path.strip()

    input_language = st.selectbox(
        "Input language (for transcription)",
        ["english", "hinglish", "nepali", "hindi", "spanish", "french", "german", "chinese", "japanese", "korean", "russian"],
        index=0
    )
    output_language = st.selectbox(
        "Output language (for summary & chat)",
        ["english", "nepali", "hindi"],
        index=0
    )

    process_clicked = st.button("🚀 Run pipeline", type="primary", use_container_width=True, disabled=(source is None))

    if st.session_state.results is not None:
        st.divider()
        if st.button("🗑️ New session", use_container_width=True):
            cleanup_temp_file()
            st.session_state.results = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("✅ Secrets are loaded securely from Streamlit Cloud (no .env needed).")

# ----------------------------------------------------------------------
#  Pipeline runner
# ----------------------------------------------------------------------
def run_pipeline(source: str, input_lang: str, output_lang: str) -> dict:
    with st.status("Running AI pipeline…", expanded=True) as status:
        st.write("✂️ Splitting media into chunks…")
        chunks = process_input(source)
        st.write("🎙️ Transcribing ({input_lang})…")
        transcript = transcribe_all(chunks, input_lang)
        st.write("🏷️ Generating title…")
        title = generate_title(transcript, output_lang)
        st.write("📝 Summarizing…")
        summary = summarize(transcript, output_lang)
        st.write("✅ Extracting action items…")
        action_items = extract_action_items(transcript, output_lang)
        st.write("🔑 Extracting key decisions…")
        key_decisions = extract_key_decisions(transcript, output_lang)
        st.write("❓ Extracting open questions…")
        open_questions = extract_questions(transcript, output_lang)
        st.write("🧠 Building RAG index…")
        rag_chain = build_rag_chain(transcript, output_lang)
        status.update(label="Pipeline complete ✅", state="complete", expanded=False)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }

# ----------------------------------------------------------------------
#  Main area
# ----------------------------------------------------------------------
st.title("🎬 AI Video Assistant")
st.caption("Turn any video or meeting recording into a summary, insights, and a searchable chat.")

if process_clicked and source is not None:
    try:
        st.session_state.results = run_pipeline(source, input_language, output_language)
        st.session_state.messages = []
        st.toast("Pipeline finished 🎉")
    except Exception as e:
        st.session_state.results = None
        st.error(f"❌ Pipeline failed: {e}")

results = st.session_state.results
if results is None:
    st.info("👈 Choose a source in the sidebar and click **Run pipeline** to get started.")
    st.stop()

# ----------------------------------------------------------------------
#  Display results
# ----------------------------------------------------------------------
st.header(results["title"])
m1, m2, m3 = st.columns(3)
m1.metric("Words in transcript", f"{len(results['transcript'].split()):,}")
m2.metric("Characters", f"{len(results['transcript']):,}")
m3.metric("Chat messages", len(st.session_state.messages))

d1, d2 = st.columns(2)
d1.download_button("⬇️ Transcript (.txt)", results["transcript"], file_name="transcript.txt")
d2.download_button(
    "⬇️ Summary (.md)",
    f"# {results['title']}\n\n{results['summary']}",
    file_name="summary.md",
)

tabs = st.tabs(["📋 Summary", "✅ Action Items", "🔑 Key Decisions", "❓ Open Questions", "📄 Transcript"])
with tabs[0]:
    st.markdown(results["summary"])
with tabs[1]:
    st.markdown(results["action_items"])
with tabs[2]:
    st.markdown(results["key_decisions"])
with tabs[3]:
    st.markdown(results["open_questions"])
with tabs[4]:
    st.text(results["transcript"])

# ----------------------------------------------------------------------
#  Chat interface
# ----------------------------------------------------------------------
st.divider()
st.subheader("💬 Chat with your meeting")
pending = st.session_state.pop("pending_question", None)
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if not st.session_state.messages and pending is None:
    suggestions = ["What are the action items?", "What decisions were made?", "What questions are still open?"]
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

prompt = st.chat_input("Ask anything about the meeting…") or pending
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        placeholder.markdown("Thinking…")
        try:
            answer = ask_question(results["rag_chain"], prompt)
        except Exception as e:
            answer = f"⚠️ Error: {e}"
        placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
