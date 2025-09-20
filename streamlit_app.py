import os
import time
import tempfile
from pathlib import Path
import streamlit as st
from typing import Optional, List

# Local imports
from retrieval import studybot_query_with_meta
from embeddings import build_course_index
from config import NOTES_DIR, INDEX_DIR

# ----------------------
# UI CONFIG
# ----------------------
st.set_page_config(page_title="Notes Q&A (Glass UI)", page_icon="📚", layout="centered")

# Custom CSS for glassmorphism, shadows, and subtle floating animation
GLASS_CSS = """
<style>
/* Background gradient */
.stApp {
  background: linear-gradient(135deg, rgba(24,26,32,1) 0%, rgba(33,37,41,1) 50%, rgba(48,52,58,1) 100%);
}


/* Glass card */
.glass-card {
  background: rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  backdrop-filter: blur(12px) saturate(1.2);
  -webkit-backdrop-filter: blur(12px) saturate(1.2);
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 20px 22px;
  animation: floaty 6s ease-in-out infinite;
}

/* Floating animation */
@keyframes floaty {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-4px); }
  100% { transform: translateY(0px); }
}

/* Headings */
.glass-title {
  color: #f2f4f8;
  font-weight: 800;
  letter-spacing: 0.2px;
}

/* Subtle text */
.subtle { color: #c9d1d9; opacity: 0.9; }

/* Monospace answer block */
.answer {
  white-space: pre-wrap;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  color: #ecf0f3;
}

/* Pill badges */
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(255,255,255,0.10); color: #e5e7eb; margin-right: 6px; font-size: 12px; }

/* Buttons */
.stButton>button {
  background: linear-gradient(135deg, #4c82ff 0%, #7c5cff 100%);
  color: white;
  border: 0;
  border-radius: 10px;
  box-shadow: 0 8px 20px rgba(76,130,255,0.35);
}

/* Inputs */
.stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div>div {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  color: #f3f4f6;
}

/* Divider */
hr.glass-divider {border: 0; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.3), rgba(255,255,255,0));}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.title("⚙️ Settings")

# Courses: list subfolders in notes directory
courses: List[str] = []
notes_root = NOTES_DIR
if os.path.exists(notes_root):
    for name in sorted(os.listdir(notes_root)):
        p = os.path.join(notes_root, name)
        if os.path.isdir(p):
            courses.append(name)
if "DSA" not in courses:
    courses.insert(0, "DSA")

# Source selector: Local notes vs Uploaded
source_mode = st.sidebar.radio("Source", ["Local notes", "Uploaded (session)"])

# Upload panel: stores to a session temp dir and builds index for course 'UPLOADED'
uploaded_course = "UPLOADED"
session_tmp_root = Path(tempfile.gettempdir()) / "studybot_uploads"
session_tmp_root.mkdir(parents=True, exist_ok=True)
session_upload_dir = session_tmp_root / uploaded_course

if source_mode == "Uploaded (session)":
    st.sidebar.markdown("---")
    files = st.sidebar.file_uploader(
        "Upload notes (PDF, DOCX, PPTX, TXT, MD)",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        help="Files are stored in a temp folder on this machine while the app runs."
    )
    if files:
        # Write files to temp dir structure: <tmp>/UPLOADED/
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            out_path = session_upload_dir / f.name
            with open(out_path, "wb") as out:
                out.write(f.read())
            saved.append(out_path)
        with st.spinner("Indexing uploaded notes..."):
            # Build a fresh index for uploaded course
            index_path = build_course_index(uploaded_course, str(session_upload_dir))
        st.sidebar.success(f"Indexed {len(saved)} file(s). Course: {uploaded_course}")
        st.session_state["uploaded_ready"] = True

course = st.sidebar.selectbox(
    "Course",
    options=([uploaded_course] if source_mode == "Uploaded (session)" and st.session_state.get("uploaded_ready") else []) + courses,
    index=0
)
lang = st.sidebar.selectbox("Language", options=["English", "Tamil", "Sinhala"], index=0)
lang_map = {"English": "en", "Tamil": "ta", "Sinhala": "si"}
show_score = st.sidebar.toggle("Show top score", value=True)

protect_list = st.sidebar.text_input("Protected terms (comma-separated)", value="")

st.sidebar.markdown("""
<div class='badge'>Hybrid Retrieval</div>
<div class='badge'>Glass UI</div>
<div class='badge'>Offline-friendly</div>
""", unsafe_allow_html=True)

# ----------------------
# Header
# ----------------------
st.markdown("""
<h1 class='glass-title'>📚 Notes Q&A — Glass</h1>
<p class='subtle'>Ask questions about your notes. The bot answers strictly from your content with citations.</p>
<hr class='glass-divider'/>
""", unsafe_allow_html=True)

# ----------------------
# Chat input card
# ----------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    question = st.text_input("Ask your question", key="question_input", placeholder="e.g., What Can PHP Do")
    ask = st.button("Ask Question", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Answer card
# ----------------------
if ask and question.strip():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    # Build extra protected list
    extra_terms = [x.strip() for x in protect_list.split(',') if x.strip()]

    # Map lang
    target_lang = lang_map.get(lang, "en")

    # Query
    start = time.time()
    # If source is uploaded but not indexed yet, block
    if source_mode == "Uploaded (session)" and not st.session_state.get("uploaded_ready"):
        st.warning("Please upload notes and wait for indexing before asking a question.")
    else:
        formatted, top_score = studybot_query_with_meta(course, question.strip(), target_lang)
    dur = time.time() - start

    # Display
    st.markdown("<h3 class='glass-title'>Answer</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='answer'>{formatted}</div>", unsafe_allow_html=True)

    # Footer badges
    badges = [f"{dur*1000:.0f} ms"]
    if show_score:
        badges.append(f"Top score: {top_score:.2f}")
    st.markdown(" ".join([f"<span class='badge'>{b}</span>" for b in badges]), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Helpful info block
with st.expander("About this app"):
    st.write("""
- Retrieval is hybrid (embeddings + lexical BM25) with heading-aware selection.
- Answers are composed strictly from your notes with citations.
- Labels and answer text localize to English/Tamil/Sinhala; citations stay unchanged.
- Proper Noun Lock preserves code tokens, laws (e.g., Data Protection Act 1998), orgs, versions, and URLs.
- Display cleanup fixes OCR run-ons and spacing.
""")
