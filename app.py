import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from generation.pipeline  import DocuMindPipeline
from embeddings.index_docs import index_documents
from embeddings.vectorstore import VectorStore
from ui.components import render_message, render_status_badge

# ─── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: #0d1117;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] * {
    color: #c9d1d9;
}

/* ── Main header ── */
.documind-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 28px;
}

.documind-logo {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.documind-title {
    font-size: 26px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.5px;
}

.documind-subtitle {
    font-size: 13px;
    color: #8b949e;
    margin-top: 2px;
}

/* ── Chat messages ── */
.message {
    margin: 12px 0;
    display: flex;
    gap: 12px;
}

.user-message {
    justify-content: flex-end;
}

.assistant-message {
    justify-content: flex-start;
}

.message-content {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.7;
}

.user-message .message-content {
    background: #1f6feb;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}

.assistant-message .message-content {
    background: #161b22;
    color: #c9d1d9;
    border: 1px solid #21262d;
    border-bottom-left-radius: 4px;
}

/* ── Citation cards ── */
.sources-block {
    margin: 8px 0 16px 0;
    padding-left: 4px;
}

.sources-label {
    font-size: 11px;
    font-weight: 500;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.citation-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #1f6feb;
    border-radius: 6px;
    margin-bottom: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

.citation-num {
    color: #388bfd;
    font-weight: 500;
    min-width: 28px;
}

.citation-text {
    color: #8b949e;
}

/* ── Input area ── */
.stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #c9d1d9 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 18px !important;
}

.stTextInput input:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.15) !important;
}

/* ── Buttons ── */
.stButton button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}

.stButton button:hover {
    background: #388bfd !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 12px;
    padding: 12px;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px 16px;
}

[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 12px !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 24px !important;
}

/* ── Divider ── */
hr {
    border-color: #21262d !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}

/* ── Status badges ── */
.status-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #8b949e;
}

.empty-state h3 {
    font-size: 18px;
    color: #c9d1d9;
    margin-bottom: 8px;
}

.empty-state p {
    font-size: 14px;
    line-height: 1.6;
}

/* ── Doc chip ── */
.doc-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #c9d1d9;
}

.doc-chip-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #388bfd;
}
</style>
""", unsafe_allow_html=True)


# ─── Load pipeline once ──────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return DocuMindPipeline()


@st.cache_resource
def load_store():
    return VectorStore()


# ─── Session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False


# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocuMind")
    st.markdown("---")

    # Document upload
    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            dest = os.path.join("docs", uploaded_file.name)
            if not os.path.exists(dest):
                with open(dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved: {uploaded_file.name}")

        if st.button("Index New Documents", use_container_width=True):
            with st.spinner("Indexing documents..."):
                index_documents()
            st.success("Indexing complete!")
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")

    # Loaded documents
    st.markdown("#### Loaded Documents")
    store = load_store()
    docs  = store.get_all_documents()

    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="doc-chip">
                    <span class="doc-chip-name">📄 {doc}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{doc}",
                             help=f"Remove {doc}"):
                    store.delete_document(doc)
                    try:
                        os.remove(os.path.join("docs", doc))
                    except:
                        pass
                    st.cache_resource.clear()
                    st.rerun()
    else:
        st.info("No documents loaded yet.")

    st.markdown("---")

    # Stats
    st.markdown("#### Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", len(docs))
    with col2:
        st.metric("Chunks", store.count())

    st.markdown("---")

    # Clear chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="font-size:11px;color:#8b949e;margin-top:16px;
    text-align:center">
    Powered by Mistral + ChromaDB<br>Running 100% locally
    </div>
    """, unsafe_allow_html=True)


# ─── Main area ───────────────────────────────────────────────
st.markdown("""
<div class="documind-header">
    <div class="documind-logo">🧠</div>
    <div>
        <div class="documind-title">DocuMind</div>
        <div class="documind-subtitle">
            Ask anything about your documents
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status badges
store_count = store.count()
doc_count   = len(docs)
status_html = '<div class="status-row">'
status_html += render_status_badge(
    "Model", "Mistral 7B", "blue")
status_html += render_status_badge(
    "Docs", str(doc_count), "green" if doc_count > 0 else "gray")
status_html += render_status_badge(
    "Chunks", str(store_count),
    "green" if store_count > 0 else "gray")
status_html += render_status_badge(
    "Status", "Ready" if store_count > 0 else "No docs", "green"
    if store_count > 0 else "amber")
status_html += '</div>'
st.markdown(status_html, unsafe_allow_html=True)

# Chat history
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <h3>Welcome to DocuMind</h3>
        <p>Upload documents using the sidebar,<br>
        then ask anything about them below.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(
            msg["role"],
            msg["content"],
            msg.get("sources", [])
        )

# ─── Chat input ──────────────────────────────────────────────
st.markdown("<div style='height:20px'></div>",
            unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="What does this document say about...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button(
            "Ask", use_container_width=True)

if submitted and user_input.strip():
    # Add user message
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input
    })

    # Check documents loaded
    if store.count() == 0:
        response = {
            "answer":  "Please upload and index some documents "
                       "first using the sidebar.",
            "sources": [],
            "has_sources": False
        }
    else:
        # Run pipeline
        with st.spinner("Thinking..."):
            pipeline = load_pipeline()
            response = pipeline.ask(user_input)

    # Add assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response["answer"],
        "sources": response.get("sources", [])
    })

    st.rerun()