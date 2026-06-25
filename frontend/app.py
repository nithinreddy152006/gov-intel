"""
Gov-Intel | Regulatory Intelligence Engine
Premium Streamlit Frontend — Three operational modes:
  Mode A: Semantic Similarity Search (no LLM, deterministic)
  Mode B: RAG Chatbot (retrieve → rerank → Gemini synthesis)
  Mode C: Draft Policy Comparison (conflict/overlap detection)
"""
import os
import time
import requests
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gov-Intel | Regulatory Intelligence Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Gov-Intel v1.0 — Ministry of Education Regulatory Intelligence Engine"
    },
)

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:      #0a0e1a;
    --bg-card:      #111627;
    --bg-card2:     #161d30;
    --border:       #1e2a45;
    --accent:       #3b82f6;
    --accent-glow:  #3b82f620;
    --accent2:      #06b6d4;
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --text-primary: #e2e8f0;
    --text-muted:   #64748b;
    --text-dim:     #475569;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* ── Hide default Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #0a0e1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem !important;
    max-width: 1200px !important;
}

/* ── Hero header ── */
.govintel-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #0d1b2e 50%, #091420 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.govintel-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent));
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { opacity: 0.7; }
    50% { opacity: 1; }
}
.govintel-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.govintel-header p {
    font-size: 0.92rem !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
}
.badge {
    display: inline-block;
    background: var(--accent-glow);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Mode cards ── */
.mode-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.mode-card:hover { border-color: var(--accent); }
.mode-card h3 {
    color: var(--text-primary) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    margin: 0 0 6px 0 !important;
}
.mode-card p {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}

/* ── Result cards ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.15s;
}
.result-card:hover {
    border-left-color: var(--accent2);
    transform: translateX(2px);
}
.result-card .source-meta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--accent2) !important;
    margin-bottom: 8px !important;
}
.result-card .score-bar-container {
    background: #1a2438;
    border-radius: 4px;
    height: 6px;
    margin: 8px 0 12px 0;
    overflow: hidden;
}
.result-card .score-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.5s ease;
}
.result-card .text-content {
    font-size: 0.88rem !important;
    color: var(--text-primary) !important;
    line-height: 1.6 !important;
    white-space: pre-wrap;
}

/* ── Chat bubbles ── */
.chat-user {
    background: var(--accent-glow);
    border: 1px solid var(--accent);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    font-size: 0.9rem;
    color: var(--text-primary);
}
.chat-assistant {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    font-size: 0.9rem;
    color: var(--text-primary);
}

/* ── Citation pills ── */
.citation-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #1a2438;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent2);
    margin: 2px;
}

/* ── Compare ── */
.compare-meter {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.overlap-duplicate { border-left: 4px solid var(--danger) !important; }
.overlap-conflict   { border-left: 4px solid var(--warning) !important; }
.overlap-related    { border-left: 4px solid var(--success) !important; }
.overlap-badge-duplicate { color: var(--danger); background: #ef444415; border-color: var(--danger); }
.overlap-badge-conflict  { color: var(--warning); background: #f59e0b15; border-color: var(--warning); }
.overlap-badge-related   { color: var(--success); background: #10b98115; border-color: var(--success); }

/* ── Stats bar ── */
.stat-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 20px;
    margin: 4px;
    min-width: 100px;
}
.stat-pill .stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-pill .stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* ── Selectbox / radio ── */
.stRadio > label, .stSelectbox > label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Alert boxes ── */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* ── Sidebar mode buttons ── */
.sidebar-mode-btn {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    color: var(--text-muted);
    cursor: pointer;
    margin-bottom: 6px;
    transition: all 0.15s;
    font-size: 0.88rem;
}
.sidebar-mode-btn.active {
    background: var(--accent-glow);
    border-color: var(--accent);
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)


# ─── Helper functions ────────────────────────────────────────────────────────

def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is it running?"
    except Exception as e:
        return None, str(e)


def api_post(endpoint: str, json_data=None, files=None, timeout=600):
    try:
        if files:
            r = requests.post(f"{API_URL}{endpoint}", files=files, timeout=timeout)
        else:
            r = requests.post(f"{API_URL}{endpoint}", json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is it running on port 8000?"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The model may still be processing."
    except Exception as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail


def score_color(score: float) -> str:
    if score >= 0.8:   return "#10b981"
    if score >= 0.55:  return "#3b82f6"
    if score >= 0.35:  return "#f59e0b"
    return "#ef4444"


def render_source_card(idx: int, result: dict):
    score = result.get("score", 0.0)
    bar_pct = min(int(score * 100), 100)
    color = score_color(score)
    file_name = result.get("file_name", "Unknown")
    page = result.get("page_number", "?")
    text = result.get("text", "")
    url = result.get("url", "")

    source_link = f'<a href="{url}" target="_blank" style="color:#06b6d4">{file_name}</a>' if url else file_name

    st.markdown(f"""
    <div class="result-card">
        <div class="source-meta">
            [{idx}] {source_link} &nbsp;·&nbsp; Page {page} &nbsp;·&nbsp;
            <span style="color:{color}; font-weight:600">Score: {score:.4f}</span>
        </div>
        <div class="score-bar-container">
            <div class="score-bar" style="width:{bar_pct}%; background:linear-gradient(90deg, {color}, {color}88)"></div>
        </div>
        <div class="text-content">{text[:600]}{"…" if len(text) > 600 else ""}</div>
    </div>
    """, unsafe_allow_html=True)


def backend_status():
    """Get health info from backend."""
    data, err = api_get("/")
    if err:
        return None
    return data


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-size:2.4rem">🏛️</div>
        <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0">Gov-Intel</div>
        <div style="font-size:0.78rem; color:#475569; margin-top:2px">Regulatory Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "OPERATIONAL MODE",
        options=[
            "🔍  Mode A — Similarity Search",
            "💬  Mode B — RAG Chatbot",
            "📋  Mode C — Policy Comparison",
            "⚙️   Admin Panel",
        ],
        label_visibility="visible",
    )

    st.markdown("---")

    # Backend health
    st.markdown('<p style="font-size:0.75rem;color:#475569;text-transform:uppercase;letter-spacing:0.5px">System Status</p>', unsafe_allow_html=True)
    health = backend_status()
    if health:
        total = health.get("total_chunks", 0)
        st.markdown(f"""
        <div style="background:#0d1321;border:1px solid #1e2a45;border-radius:8px;padding:12px 14px;font-size:0.82rem">
            <div style="color:#10b981;margin-bottom:4px">● Backend Online</div>
            <div style="color:#64748b">Qdrant: {"✓ Connected" if health.get("qdrant_connected") else "✗ Offline"}</div>
            <div style="color:#64748b">Chunks: <span style="color:#3b82f6;font-weight:600">{total:,}</span></div>
            <div style="color:#64748b">Embed: <span style="color:#e2e8f0">{health.get("embedding_mode","")}</span></div>
            <div style="color:#64748b">LLM: <span style="color:#e2e8f0">{health.get("llm_mode","")[:30]}</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#1a0d0d;border:1px solid #ef444435;border-radius:8px;padding:12px 14px;font-size:0.82rem;color:#ef4444">
            ✗ Backend Offline<br>
            <span style="color:#64748b;font-size:0.78rem">Run: uvicorn backend.api.main:app</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#334155;text-align:center">
        Gov-Intel v1.0 · MoE India<br>
        BAAI/bge-m3 · Qdrant · Gemini
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ─────────────────────────────────────────────────────────────

mode_labels = {
    "🔍  Mode A — Similarity Search": ("🔍 Semantic Similarity Search", "Deterministic hybrid retrieval — no LLM hallucinations", "MODE A", "No LLM"),
    "💬  Mode B — RAG Chatbot": ("💬 RAG Policy Chatbot", "Retrieve → Rerank → Gemini Synthesis with citations", "MODE B", "Gemini"),
    "📋  Mode C — Policy Comparison": ("📋 Draft Policy Comparison", "Upload a draft — detect conflicts, duplicates & overlaps", "MODE C", "AI Analysis"),
    "⚙️   Admin Panel": ("⚙️ Administration Panel", "Manage ingestion, scrapers, and system health", "ADMIN", "System"),
}

title, subtitle, mode_badge, tech_badge = mode_labels[mode]

st.markdown(f"""
<div class="govintel-header">
    <div style="margin-bottom:10px">
        <span class="badge">{mode_badge}</span>
        <span class="badge" style="color:#06b6d4;border-color:#06b6d4;background:#06b6d415">{tech_badge}</span>
    </div>
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE A — Similarity Search
# ═══════════════════════════════════════════════════════════════════════════════

if "Mode A" in mode:
    col_main, col_opts = st.columns([3, 1])

    with col_opts:
        top_k = st.number_input("Results (top-k)", min_value=1, max_value=20, value=5)
        st.markdown('<p style="font-size:0.78rem;color:#475569;margin-top:8px">Hybrid dense+sparse search with BGE-Reranker cross-encoder.</p>', unsafe_allow_html=True)

    with col_main:
        query = st.text_input(
            "Query",
            placeholder="e.g. UGC recruitment rules for Assistant Professor eligibility...",
            label_visibility="collapsed",
        )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        search_clicked = st.button("Search Knowledge Base", type="primary", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Embedding query → Hybrid search → BGE-Reranker..."):
                data, err = api_post("/api/search", json_data={"query": query, "top_k": top_k})

            if err:
                st.error(f"**Error:** {err}")
            else:
                results = data.get("results", [])
                total = data.get("total_found", 0)

                if not results:
                    st.info("No relevant documents found. Have you ingested any PDFs?")
                else:
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap">
                        <div class="stat-pill"><span class="stat-val">{total}</span><span class="stat-label">Results</span></div>
                        <div class="stat-pill"><span class="stat-val">{results[0].get('score',0):.3f}</span><span class="stat-label">Top Score</span></div>
                        <div class="stat-pill"><span class="stat-val">{results[-1].get('score',0):.3f}</span><span class="stat-label">Min Score</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"### Retrieved Chunks — `{query[:60]}{'…' if len(query)>60 else ''}`")
                    for idx, result in enumerate(results, 1):
                        render_source_card(idx, result)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE B — RAG Chatbot
# ═══════════════════════════════════════════════════════════════════════════════

elif "Mode B" in mode:
    # Session state init
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = {}
    if "chat_temp_context" not in st.session_state:
        st.session_state.chat_temp_context = None
    if "chat_temp_filename" not in st.session_state:
        st.session_state.chat_temp_filename = None

    # Controls
    col_clear, col_llm, col_info = st.columns([1, 2, 4])
    with col_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.chat_sources = {}
            st.session_state.chat_temp_context = None
            st.session_state.chat_temp_filename = None
            st.rerun()

    with col_llm:
        llm_choice = st.radio("LLM Engine", ["Cloud (Gemini)", "Local (Llama 3 8B)"], horizontal=True, label_visibility="collapsed")

    with col_info:
        temp_upload = st.file_uploader("Upload PDF to permanent knowledge base", type=["pdf"], key="chat_temp_upload", label_visibility="collapsed")
        
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = set()
            
        if temp_upload and temp_upload.name not in st.session_state.ingested_files:
            import time
            with st.status(f"Ingesting {temp_upload.name} into Knowledge Base...", expanded=True) as status:
                files = {"file": (temp_upload.name, temp_upload.getvalue(), "application/pdf")}
                data, err = api_post("/api/ingest", files=files, timeout=30)
                if err:
                    status.update(label=f"Error uploading: {err}", state="error")
                else:
                    job_id = data.get("job_id")
                    status.write("Processing document in background...")
                    for _ in range(60):
                        time.sleep(2)
                        status_data, _ = api_get(f"/api/ingest/status/{job_id}")
                        if status_data:
                            s = status_data.get("status")
                            chunks = status_data.get("chunks_indexed", 0)
                            if s == "done":
                                status.update(label=f"✓ Permanent Ingestion Complete. {chunks} chunks added to database.", state="complete")
                                st.session_state.ingested_files.add(temp_upload.name)
                                break
                            elif s == "failed":
                                status.update(label=f"✗ Failed to ingest: {status_data.get('error', 'unknown')}", state="error")
                                break

    # Render history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#334155">
                <div style="font-size:2.5rem;margin-bottom:12px">💬</div>
                <div style="font-size:1rem;color:#475569">Ask a question about MoE policies, UGC regulations, or AICTE circulars.</div>
                <div style="font-size:0.82rem;color:#334155;margin-top:6px">The system retrieves relevant documents and synthesises a cited answer.</div>
            </div>
            """, unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show sources if assistant message
                if msg["role"] == "assistant" and i in st.session_state.chat_sources:
                    sources = st.session_state.chat_sources[i]
                    if sources:
                        with st.expander(f"📎 {len(sources)} Source(s) Retrieved"):
                            for j, src in enumerate(sources, 1):
                                render_source_card(j, {
                                    "file_name": src.get("file_name"),
                                    "page_number": src.get("page_number"),
                                    "score": src.get("score", 0),
                                    "text": src.get("text", ""),
                                    "url": src.get("url", ""),
                                })

    # Input
    if prompt := st.chat_input("Ask about policy eligibility, scheme criteria, section references…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            history = [
                m for m in st.session_state.chat_messages[:-1]
            ]
            payload = {
                "query": prompt, 
                "conversation_history": history,
                "temporary_context": st.session_state.chat_temp_context,
                "use_local_llm": llm_choice == "Local (Llama 3 8B)"
            }
            
            import json
            status = st.status("Thinking...", expanded=True)
            answer_placeholder = st.empty()
            
            full_answer = ""
            sources = []
            model = "none"

            try:
                response = requests.post(
                    f"{API_URL}/api/chat_stream",
                    json=payload,
                    stream=True,
                    timeout=180,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        obj = json.loads(line)
                        if obj["type"] == "status":
                            status.update(label=obj["message"], state="running")
                        elif obj["type"] == "sources":
                            sources = obj["data"]
                            status.write(f"Retrieved {len(sources)} context chunks.")
                        elif obj["type"] == "chunk":
                            full_answer += obj["data"]
                            answer_placeholder.markdown(full_answer + "▌")
                        elif obj["type"] == "done":
                            model = obj.get("model", "none")
                            
                status.update(label="Detailed Synthesis Complete", state="complete", expanded=False)
                answer_placeholder.markdown(full_answer)
                st.caption(f"_Model: {model}_")

                msg_idx = len(st.session_state.chat_messages)
                st.session_state.chat_messages.append({"role": "assistant", "content": full_answer})
                st.session_state.chat_sources[msg_idx] = sources

                if sources:
                    with st.expander(f"📎 {len(sources)} Source(s) Retrieved"):
                        for j, src in enumerate(sources, 1):
                            render_source_card(j, {
                                "file_name": src.get("file_name"),
                                "page_number": src.get("page_number"),
                                "score": src.get("score", 0),
                                "text": src.get("text", ""),
                                "url": src.get("url", ""),
                            })

            except Exception as e:
                status.update(label="Failed", state="error")
                answer = f"⚠️ **Error:** {e}"
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════════════════════════
# MODE C — Policy Comparison
# ═══════════════════════════════════════════════════════════════════════════════

elif "Mode C" in mode:
    st.markdown("""
    <div class="mode-card">
        <h3>📋 Draft Policy Conflict Detector</h3>
        <p>Upload a draft regulation PDF. The system will score it against the full knowledge base
        using deterministic hybrid search — no LLM bias. Matches are classified as
        <span style="color:#ef4444">Duplicate</span>,
        <span style="color:#f59e0b">Conflict</span>, or
        <span style="color:#10b981">Related</span>.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Draft Policy PDF",
        type=["pdf"],
        help="Upload the draft regulation you want to compare against existing policies.",
    )

    if uploaded:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.markdown(f"""
            <div style="background:#0d1321;border:1px solid #1e2a45;border-radius:8px;padding:10px 14px;font-size:0.85rem">
                📄 <strong style="color:#e2e8f0">{uploaded.name}</strong>
                <span style="color:#475569;margin-left:8px">{uploaded.size / 1024:.1f} KB</span>
            </div>
            """, unsafe_allow_html=True)
        with col_btn:
            compare_clicked = st.button("Analyse Draft", type="primary", use_container_width=True)

        if compare_clicked:
            with st.spinner("Parsing draft → Searching knowledge base → Classifying matches…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                data, err = api_post("/api/compare", files=files, timeout=180)

            if err:
                st.error(f"**Error:** {err}")
            else:
                max_sim = data.get("max_similarity", 0)
                matches = data.get("matches", [])
                summary = data.get("summary", "")
                draft_name = data.get("draft_name", uploaded.name)

                # ── Summary Meter ──
                meter_color = "#ef4444" if max_sim >= 0.85 else "#f59e0b" if max_sim >= 0.65 else "#10b981"
                verdict = "⚠️ High Overlap Detected" if max_sim >= 0.85 else "🟡 Moderate Overlap" if max_sim >= 0.65 else "✅ Low Overlap"

                st.markdown(f"""
                <div class="compare-meter">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <span style="font-size:1rem;font-weight:600;color:#e2e8f0">{draft_name}</span>
                        <span style="font-size:1rem;font-weight:700;color:{meter_color}">{verdict}</span>
                    </div>
                    <div style="background:#1a2438;border-radius:6px;height:10px;overflow:hidden;margin-bottom:8px">
                        <div style="width:{int(max_sim*100)}%;height:100%;background:linear-gradient(90deg,{meter_color},{meter_color}99);border-radius:6px;transition:width 1s ease"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between">
                        <span style="font-size:0.8rem;color:#475569">Maximum Similarity Score</span>
                        <span style="font-size:0.92rem;font-weight:700;color:{meter_color}">{max_sim:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Stats ──
                dupes = sum(1 for m in matches if m.get("overlap_type") == "duplicate")
                conflicts = sum(1 for m in matches if m.get("overlap_type") == "conflict")
                related = sum(1 for m in matches if m.get("overlap_type") == "related")

                st.markdown(f"""
                <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap">
                    <div class="stat-pill"><span class="stat-val" style="color:#ef4444">{dupes}</span><span class="stat-label">Duplicates</span></div>
                    <div class="stat-pill"><span class="stat-val" style="color:#f59e0b">{conflicts}</span><span class="stat-label">Conflicts</span></div>
                    <div class="stat-pill"><span class="stat-val" style="color:#10b981">{related}</span><span class="stat-label">Related</span></div>
                    <div class="stat-pill"><span class="stat-val">{len(matches)}</span><span class="stat-label">Total Matches</span></div>
                </div>
                """, unsafe_allow_html=True)

                # ── AI Summary ──
                if summary:
                    with st.expander("🤖 AI Analysis Summary", expanded=True):
                        st.markdown(summary)

                # ── Match cards ──
                st.markdown("### Similar Regulations Found")
                for idx, match in enumerate(matches, 1):
                    otype = match.get("overlap_type", "related")
                    score = match.get("similarity_score", 0)
                    badge_class = f"overlap-badge-{otype}"
                    card_class = f"overlap-{otype}"
                    color = {"duplicate": "#ef4444", "conflict": "#f59e0b", "related": "#10b981"}.get(otype, "#3b82f6")

                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div class="source-meta" style="display:flex;justify-content:space-between">
                            <span>[{idx}] {match.get("file_name","?")} · Page {match.get("page_number","?")}</span>
                            <span>
                                <span class="badge {badge_class}">{otype.upper()}</span>
                                <span style="color:{color};font-weight:700;margin-left:8px">{score:.1%}</span>
                            </span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{int(score*100)}%;background:linear-gradient(90deg,{color},{color}88)"></div>
                        </div>
                        <div class="text-content">{match.get("excerpt","")}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

elif "Admin" in mode:
    tab1, tab2, tab3 = st.tabs(["📥 Ingest Documents", "🕷️ Scrapers", "📊 System Info"])

    with tab1:
        st.markdown("### Upload Document(s) to Knowledge Base")

        col_upload, col_dir = st.columns(2)

        with col_upload:
            st.markdown("**Single PDF Upload**")
            upload_file = st.file_uploader("Choose PDF", type=["pdf"], key="admin_upload")
            if upload_file and st.button("Ingest PDF", type="primary"):
                with st.spinner(f"Ingesting {upload_file.name}..."):
                    files = {"file": (upload_file.name, upload_file.getvalue(), "application/pdf")}
                    data, err = api_post("/api/ingest", files=files, timeout=30)
                if err:
                    st.error(f"Error: {err}")
                else:
                    job_id = data.get("job_id")
                    st.success(f"Ingestion started. Job ID: `{job_id}`")
                    st.caption("Poll status below:")
                    # Poll status
                    for _ in range(30):
                        time.sleep(3)
                        status_data, _ = api_get(f"/api/ingest/status/{job_id}")
                        if status_data:
                            s = status_data.get("status")
                            chunks = status_data.get("chunks_indexed", 0)
                            if s == "done":
                                st.success(f"✓ Done — {chunks} chunks indexed")
                                break
                            elif s == "failed":
                                st.error(f"✗ Failed: {status_data.get('error', 'unknown')}")
                                break
                            else:
                                st.info(f"Status: {s} | Chunks so far: {chunks}")

        with col_dir:
            st.markdown("**Bulk: Ingest Policies Directory**")
            st.caption("Ingests all PDFs in `/data/policies`")
            if st.button("Ingest All PDFs in /data/policies/"):
                data, err = api_post("/api/ingest/directory")
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success(f"Started {len(data)} ingestion job(s).")
                    for j in data:
                        st.markdown(f"- `{j['file_name']}` → Job `{j['job_id']}`")

    with tab2:
        st.markdown("### Scraper Control")

        status_data, _ = api_get("/api/scrape/status")
        if status_data:
            last = status_data.get("last_run", "Never")
            discovered = status_data.get("files_discovered", 0)
            indexed = status_data.get("files_indexed", 0)
            next_run = status_data.get("next_scheduled_run", "N/A")

            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap">
                <div class="stat-pill"><span class="stat-val">{discovered}</span><span class="stat-label">Discovered</span></div>
                <div class="stat-pill"><span class="stat-val">{indexed}</span><span class="stat-label">Indexed</span></div>
                <div class="stat-pill"><span class="stat-val" style="font-size:0.9rem">{str(last)[:19] if last else "Never"}</span><span class="stat-label">Last Run</span></div>
                <div class="stat-pill"><span class="stat-val" style="font-size:0.9rem">{str(next_run)[:19] if next_run else "N/A"}</span><span class="stat-label">Next Run</span></div>
            </div>
            """, unsafe_allow_html=True)

            errors = status_data.get("errors", [])
            if errors:
                st.warning("Errors from last run:")
                for e in errors:
                    st.caption(f"• {e}")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("▶ Trigger Scraper Now", type="primary"):
                data, err = api_post("/api/scrape/trigger")
                if err:
                    st.error(f"Error: {err}")
                else:
                    st.success("Scraper started in background. Refresh status in a few minutes.")

        st.markdown("**Portals being scraped:**")
        for portal in [
            "MoE Schemes (education.gov.in)", 
            "UGC Regulations (ugc.gov.in)", 
            "MoE OpenData (data.gov.in)", 
            "NEP 2020 (education.gov.in)"
        ]:
            st.markdown(f"- {portal}")

    with tab3:
        st.markdown("### System Health")
        health, err = api_get("/")
        if err:
            st.error(f"Backend unreachable: {err}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Backend Version:** `{health.get("version","")}`
                **Qdrant:** {"✅ Connected" if health.get("qdrant_connected") else "❌ Offline"}
                **Collection:** {"✅ Ready" if health.get("collection_exists") else "❌ Not found"}
                **Chunks in DB:** `{health.get("total_chunks", 0):,}`
                """)
            with col2:
                st.markdown(f"""
                **Embedding Mode:** `{health.get("embedding_mode","")}`
                **LLM Mode:** `{health.get("llm_mode","")}`
                """)
            st.json(health)

# Fix variable used in admin panel
settings_dir = os.environ.get("POLICIES_DIR", "./data/policies")
