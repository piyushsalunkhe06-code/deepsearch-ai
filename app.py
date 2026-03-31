"""
DeepSearch AI — v3.0  ██████████████████████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All v2.0 features retained
🆕 1. FAISS Persistence  — save/load index.faiss per video
🆕 2. Multi-Video SQLite — video library with frame DB
🆕 3. Timeline View      — visual ⚠️ marker strip for matches
🆕 4. Heatmap Strip      — density bar across video duration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import cv2
import numpy as np
import torch
import sqlite3
import faiss
import hashlib
import json
import time
import tempfile
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="DeepSearch AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# CSS — Design System (unchanged from v2.0)
# ══════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono&display=swap');

[data-testid="stAppViewContainer"] { background:#07090d; color:#dde6f0; }
[data-testid="stSidebar"]          { background:#0b0f14; border-right:1px solid #1a2535; }

.stButton > button {
    background:linear-gradient(135deg,#00c8f8,#0090c0);
    color:#000; font-weight:800; border:none; border-radius:8px;
    padding:0.55rem 1.4rem; letter-spacing:.4px; transition:all .2s;
}
.stButton > button:hover {
    background:linear-gradient(135deg,#26d8ff,#00aadd);
    box-shadow:0 0 22px rgba(0,200,248,.4); transform:translateY(-1px);
}
.stButton > button:disabled { opacity:.35; transform:none !important; box-shadow:none !important; }

.ds-card {
    background:#0e1520; border:1px solid #1a2535; border-radius:12px;
    padding:16px; margin-bottom:14px; transition:border-color .2s, box-shadow .2s;
}
.ds-card:hover { border-color:#00c8f8; box-shadow:0 4px 24px rgba(0,200,248,.1); }

.rank-badge { display:inline-block; border-radius:6px; padding:3px 10px;
    font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }
.rank-1 { background:rgba(255,215,0,.15);  color:#ffd700; border:1px solid rgba(255,215,0,.4); }
.rank-2 { background:rgba(192,192,192,.12);color:#c0c0c0; border:1px solid rgba(192,192,192,.3); }
.rank-3 { background:rgba(205,127,50,.12); color:#cd7f32; border:1px solid rgba(205,127,50,.3); }
.rank-n { background:rgba(0,200,248,.07);  color:#5a7090; border:1px solid #1a2535; }

.alert-critical {
    background:rgba(255,55,75,.1); border:1px solid rgba(255,55,75,.5);
    border-radius:10px; padding:14px 18px; color:#ff5070;
    font-family:'Space Mono',monospace; font-size:13px; animation:flash 1.2s infinite;
}
.alert-warn {
    background:rgba(255,175,0,.08); border:1px solid rgba(255,175,0,.4);
    border-radius:10px; padding:14px 18px; color:#ffaf00;
    font-family:'Space Mono',monospace; font-size:13px;
}
.alert-ok {
    background:rgba(0,196,98,.08); border:1px solid rgba(0,196,98,.3);
    border-radius:10px; padding:14px 18px; color:#00c462;
    font-family:'Space Mono',monospace; font-size:13px;
}
@keyframes flash { 0%,100%{opacity:1} 50%{opacity:.5} }

.q-chip {
    display:inline-block; background:rgba(0,200,248,.1);
    border:1px solid rgba(0,200,248,.35); border-radius:100px;
    padding:3px 12px; margin:2px; font-family:'Space Mono',monospace;
    font-size:11px; color:#00c8f8;
}
.score-outer { height:6px; background:#1a2535; border-radius:100px; overflow:hidden; margin-top:4px; }
.score-inner { height:100%; border-radius:100px; transition:width .4s; }

.ds-title {
    font-family:'Syne',sans-serif; font-size:40px; font-weight:800;
    background:linear-gradient(120deg,#00c8f8 0%,#ffffff 52%,#ff3c6e 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1.1; margin-bottom:4px;
}
.ds-sub {
    font-family:'Space Mono',monospace; font-size:10px;
    color:#1e3040; letter-spacing:3px; text-transform:uppercase;
}

/* ── NEW v3.0: Timeline & DB styles ── */
.timeline-wrap {
    background:#0b0f14; border:1px solid #1a2535; border-radius:10px;
    padding:18px 20px; margin:18px 0;
}
.tl-label {
    font-family:'Space Mono',monospace; font-size:9px;
    color:#1e3040; letter-spacing:2px; text-transform:uppercase;
    margin-bottom:10px;
}
.tl-track {
    position:relative; height:28px; background:#0e1520;
    border-radius:6px; overflow:visible; margin-bottom:6px;
}
.tl-fill {
    position:absolute; left:0; top:0; height:100%;
    border-radius:6px; pointer-events:none;
}
.tl-marker {
    position:absolute; top:-4px;
    width:3px; height:36px; border-radius:2px;
    cursor:pointer; transition:opacity .15s;
}
.tl-marker:hover { opacity:.7; }
.tl-tick-row {
    display:flex; justify-content:space-between;
    font-family:'Space Mono',monospace; font-size:8px; color:#1e3040;
    padding:0 2px;
}
.heat-row {
    display:flex; height:10px; border-radius:4px; overflow:hidden; margin:4px 0 10px;
}
.heat-cell { flex:1; transition:background .3s; }

.vid-lib-row {
    background:#0e1520; border:1px solid #1a2535; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; display:flex;
    align-items:center; gap:12px;
    transition:border-color .2s;
}
.vid-lib-row:hover { border-color:#00c8f8; }
.vid-lib-name {
    font-family:'Syne',sans-serif; font-size:14px; font-weight:700;
    color:#dde6f0; flex:1; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
}
.vid-lib-meta {
    font-family:'Space Mono',monospace; font-size:9px; color:#1e3040;
}
.badge-indexed {
    background:rgba(0,196,98,.12); color:#00c462;
    border:1px solid rgba(0,196,98,.3); border-radius:4px;
    padding:2px 8px; font-family:'Space Mono',monospace; font-size:9px;
}
.badge-new {
    background:rgba(0,200,248,.1); color:#00c8f8;
    border:1px solid rgba(0,200,248,.3); border-radius:4px;
    padding:2px 8px; font-family:'Space Mono',monospace; font-size:9px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# CONSTANTS & PATHS
# ══════════════════════════════════════════════

DB_DIR   = Path("deepsearch_db")
DB_DIR.mkdir(exist_ok=True)
DB_PATH  = DB_DIR / "videos.db"
IDX_DIR  = DB_DIR / "faiss_indexes"
IDX_DIR.mkdir(exist_ok=True)
VID_DIR  = DB_DIR / "videos"
VID_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════
# 🆕 UPGRADE 2: SQLite Video Database
# ══════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id    TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            path        TEXT NOT NULL,
            duration    REAL,
            fps         REAL,
            frame_count INTEGER,
            indexed_at  TEXT,
            faiss_path  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id    TEXT NOT NULL,
            frame_idx   INTEGER NOT NULL,
            timestamp   REAL NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(video_id)
        )
    """)
    conn.commit()
    return conn


def video_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes[:65536]).hexdigest()[:16]


def db_list_videos(conn) -> list:
    cur = conn.execute("""
        SELECT video_id, name, duration, frame_count, indexed_at
        FROM videos ORDER BY indexed_at DESC
    """)
    return cur.fetchall()


def db_get_video(conn, video_id: str) -> dict | None:
    cur = conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,))
    row = cur.fetchone()
    if not row:
        return None
    keys = ["video_id","name","path","duration","fps","frame_count","indexed_at","faiss_path"]
    return dict(zip(keys, row))


def db_get_timestamps(conn, video_id: str) -> list:
    cur = conn.execute(
        "SELECT timestamp FROM frames WHERE video_id=? ORDER BY frame_idx",
        (video_id,)
    )
    return [r[0] for r in cur.fetchall()]


def db_save_video(conn, video_id, name, path, duration, fps,
                  frame_count, faiss_path, timestamps):
    conn.execute("""
        INSERT OR REPLACE INTO videos
            (video_id,name,path,duration,fps,frame_count,indexed_at,faiss_path)
        VALUES (?,?,?,?,?,?,?,?)
    """, (video_id, name, str(path), duration, fps, frame_count,
          datetime.now().isoformat(timespec="seconds"), str(faiss_path)))
    conn.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
    conn.executemany(
        "INSERT INTO frames (video_id,frame_idx,timestamp) VALUES (?,?,?)",
        [(video_id, i, ts) for i, ts in enumerate(timestamps)]
    )
    conn.commit()


# ══════════════════════════════════════════════
# 🆕 UPGRADE 1: FAISS Persistence helpers
# ══════════════════════════════════════════════

def faiss_path_for(video_id: str) -> Path:
    return IDX_DIR / f"{video_id}.faiss"


def save_faiss(index: faiss.Index, video_id: str) -> Path:
    p = faiss_path_for(video_id)
    faiss.write_index(index, str(p))
    return p


def load_faiss(video_id: str) -> faiss.Index | None:
    p = faiss_path_for(video_id)
    if p.exists():
        return faiss.read_index(str(p))
    return None


def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    dim   = embeddings.shape[1]          # 512 for CLIP ViT-B/32
    index = faiss.IndexFlatIP(dim)       # inner-product == cosine on L2-normed vecs
    index.add(embeddings)
    return index


# ══════════════════════════════════════════════
# HUGGING FACE CLIP  (cached singleton)
# ══════════════════════════════════════════════

@st.cache_resource(show_spinner="🤖 Loading CLIP from Hugging Face (first run only)...")
def load_model():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


# ══════════════════════════════════════════════
# PIPELINE FUNCTIONS
# ══════════════════════════════════════════════

def extract_frames(video_path: str, interval_sec: float = 3.0):
    """OpenCV: 1 frame every interval_sec seconds."""
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps * interval_sec))
    data, idx = [], 0
    prog = st.progress(0, "⏳ Extracting frames (OpenCV)...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            data.append((idx / fps, frame.copy()))
            prog.progress(min(int(idx / max(total, 1) * 100), 99),
                          f"⏳ {len(data)} frames sampled...")
        idx += 1
    duration = total / fps if fps else 0
    cap.release()
    prog.progress(100, f"✅ {len(data)} frames extracted")
    time.sleep(0.3); prog.empty()
    return data, fps, duration


def embed_frames(frames_data, model, processor, device, batch=32):
    """HuggingFace CLIP: frames → L2-norm embeddings (N, 512)."""
    all_emb = []
    prog = st.progress(0, "🧠 Encoding with CLIP (Hugging Face)...")
    for i in range(0, len(frames_data), batch):
        chunk    = frames_data[i:i+batch]
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for _, f in chunk]
        inp      = processor(images=pil_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            raw = model.get_image_features(**inp)
            emb = raw if isinstance(raw, torch.Tensor) else raw.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.cpu().numpy().astype(np.float32))
        prog.progress(min(int((i+len(chunk))/len(frames_data)*100), 99),
                      f"🧠 Batch {i//batch+1} done...")
    prog.progress(100, "✅ CLIP encoding complete!"); time.sleep(0.3); prog.empty()
    return np.vstack(all_emb)


def embed_text(query: str, model, processor, device):
    """HuggingFace CLIP: text → L2-norm embedding (1, 512)."""
    inp = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        raw = model.get_text_features(**inp)
        emb = raw if isinstance(raw, torch.Tensor) else raw.pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def cosine_similarity_numpy(text_emb: np.ndarray, frame_embs: np.ndarray) -> np.ndarray:
    return np.dot(frame_embs, text_emb.T).squeeze()


# ─────────────────────────────────────────────
# 🔥 Multi-Query Search (FAISS-accelerated)
# ─────────────────────────────────────────────

def multi_query_search(raw_query, faiss_index, frames_data,
                        model, processor, device, top_k=10, combine="average"):
    sub_qs = [q.strip() for q in raw_query.split("+") if q.strip()]
    if not sub_qs:
        return [], []

    N = len(frames_data)
    score_matrix = np.zeros((len(sub_qs), N), dtype=np.float32)

    for qi, sq in enumerate(sub_qs):
        text_emb = embed_text(sq, model, processor, device)   # (1, 512)
        # FAISS search: returns (distances, indices) for top-N
        D, I = faiss_index.search(text_emb, N)                # D shape (1, N)
        # Scatter back into full-size score array
        for rank_pos in range(N):
            score_matrix[qi, I[0, rank_pos]] = float(D[0, rank_pos])

    combined = {"average": np.mean, "min": np.min, "max": np.max}[combine](score_matrix, axis=0)
    top_idxs = np.argsort(combined)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idxs, 1):
        ts, frame_bgr = frames_data[idx]
        results.append({
            "rank":          rank,
            "timestamp_sec": float(ts),
            "timestamp_fmt": _fmt_ts(float(ts)),
            "score":         float(combined[idx]),
            "per_query":     {sq: float(score_matrix[qi, idx]) for qi, sq in enumerate(sub_qs)},
            "frame_bgr":     frame_bgr,
        })
    return results, sub_qs


# ─────────────────────────────────────────────
# 🔥 Highlight Reel
# ─────────────────────────────────────────────

def extract_highlight_clip(video_path, timestamp_sec, video_fps, duration=6.0) -> str:
    cap  = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS) or video_fps
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec - duration / 2) * 1000)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(duration * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release(); out.release()
    return tmp.name


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _fmt_ts(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}.{ms:03d}"

def _score_color(v: float) -> str:
    if v > 0.30: return "#00e676"
    if v > 0.24: return "#00c8f8"
    if v > 0.18: return "#ffb400"
    return "#5a7090"

def _rank_badge(rank: int) -> str:
    icons = {1:"🥇", 2:"🥈", 3:"🥉"}
    cls   = {1:"rank-1", 2:"rank-2", 3:"rank-3"}.get(rank, "rank-n")
    icon  = icons.get(rank, f"#{rank}")
    return f"<span class='rank-badge {cls}'>{icon} Rank {rank}</span>"

def _score_bar(score: float, max_s: float) -> str:
    pct = max(0, min(100, score / max(max_s, 0.01) * 100))
    col = _score_color(score)
    return (f"<div class='score-outer'>"
            f"<div class='score-inner' style='width:{pct:.0f}%;background:{col}'></div>"
            f"</div>")


# ══════════════════════════════════════════════
# 🆕 UPGRADE 3: Timeline View renderer
# ══════════════════════════════════════════════

def render_timeline(results: list, duration: float, all_timestamps: list,
                    alert_threshold: float):
    """
    Renders two visual strips:
      1. Heatmap — density of all indexed frames across the video
      2. Match timeline — ⚠️ markers for result timestamps, colour-coded by score
    """
    if duration <= 0:
        return

    st.markdown("<div class='timeline-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='tl-label'>📊 Video Timeline</div>", unsafe_allow_html=True)

    # ── Heatmap strip (frame density) ──────────────────────────────
    HEAT_CELLS = 80
    counts = np.zeros(HEAT_CELLS, dtype=float)
    for ts in all_timestamps:
        cell = min(int(ts / duration * HEAT_CELLS), HEAT_CELLS - 1)
        counts[cell] += 1
    mx = max(counts.max(), 1)
    heat_cells_html = ""
    for c in counts:
        intensity = int(c / mx * 180)
        heat_cells_html += (
            f"<div class='heat-cell' "
            f"style='background:rgba(0,200,248,{intensity/255:.2f})'></div>"
        )
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:8px;"
        "color:#1e3040;margin-bottom:2px'>FRAME DENSITY</div>"
        f"<div class='heat-row'>{heat_cells_html}</div>",
        unsafe_allow_html=True,
    )

    # ── Match marker strip ──────────────────────────────────────────
    TRACK_PX = 640
    markers_html = ""
    for r in results:
        pct  = r["timestamp_sec"] / duration
        left = pct * 100          # percent from left
        col  = _score_color(r["score"])
        tip  = f"{r['timestamp_fmt']} · {r['score']*100:.1f}%"
        markers_html += (
            f"<div class='tl-marker' title='{tip}' "
            f"style='left:{left:.2f}%;background:{col};opacity:.9'></div>"
        )
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:8px;"
        "color:#1e3040;margin-bottom:4px'>MATCH LOCATIONS</div>"
        f"<div class='tl-track'>{markers_html}</div>",
        unsafe_allow_html=True,
    )

    # ── Tick labels ─────────────────────────────────────────────────
    n_ticks = 6
    ticks_html = "".join(
        f"<span>{_fmt_ts(duration * i / (n_ticks-1))}</span>"
        for i in range(n_ticks)
    )
    st.markdown(f"<div class='tl-tick-row'>{ticks_html}</div>", unsafe_allow_html=True)

    # ── Legend ──────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:8px;color:#1e3040;"
        "margin-top:8px;display:flex;gap:16px'>"
        "<span><span style='color:#00e676'>■</span> High (&gt;30%)</span>"
        "<span><span style='color:#00c8f8'>■</span> Medium (&gt;24%)</span>"
        "<span><span style='color:#ffb400'>■</span> Low (&gt;18%)</span>"
        "<span><span style='color:#5a7090'>■</span> Weak</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════

_defaults = dict(
    faiss_index=None, frames_data=None,
    active_video_id=None, active_video_info=None,
    results=[], sub_queries=[], alert_log=[],
    highlight_cache={}, jump_ts=None, highlight_req=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

conn = get_db()


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:14px 0 6px'>"
        "<span style='font-family:Syne,sans-serif;font-size:21px;font-weight:800;"
        "background:linear-gradient(120deg,#00c8f8,#fff,#ff3c6e);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent'>"
        "🔍 DeepSearch AI</span><br>"
        "<span style='font-size:9px;color:#1e3040;letter-spacing:2px'>V3.0 · ZERO-SHOT · MULTIMODAL</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**📥 Input Source**")
    input_mode = st.radio("mode", ["📁 Upload Video File", "📡 Live RTSP / IP Camera"],
                          label_visibility="collapsed")
    st.divider()

    uploaded_file = rtsp_url = None
    combine_key = "average"
    top_k = 8
    clip_duration = 6
    alert_threshold = 0.27
    index_btn = live_btn = False
    sample_interval = 3.0

    if input_mode == "📁 Upload Video File":
        uploaded_file   = st.file_uploader("Upload MP4 / AVI / MOV",
                                           type=["mp4","avi","mov","mkv","webm"])
        sample_interval = st.slider("⏱ Sample every N seconds", 0.5, 10.0, 3.0, 0.5)
        combine_mode    = st.selectbox("🔗 Multi-query mode",
                                       ["average", "min (strict AND)", "max (broad OR)"])
        combine_key     = combine_mode.split()[0]
        top_k           = st.slider("📊 Max results", 3, 20, 8)
        clip_duration   = st.slider("🎬 Highlight clip (sec)", 3, 15, 6)
        alert_threshold = st.slider("🚨 Alert threshold", 0.15, 0.45, 0.27, 0.01)
        st.divider()
        index_btn = st.button("⚡ Index This Video", use_container_width=True,
                              disabled=(uploaded_file is None))
    else:
        rtsp_url       = st.text_input("Camera URL",
                                       placeholder="rtsp://192.168.x.x:554/stream")
        live_query_val = st.text_input("🔎 Live watch query",
                                       placeholder="A person running...", key="live_q")
        alert_threshold = st.slider("🚨 Alert threshold", 0.15, 0.45, 0.27, 0.01)
        st.divider()
        live_btn = st.button("📡 Grab & Analyze Frame", use_container_width=True,
                             disabled=not rtsp_url)

    device_lbl = "CUDA 🟢" if torch.cuda.is_available() else "CPU 🟡"
    st.markdown(
        f"<div style='font-size:9px;color:#1e3040;line-height:2;font-family:Space Mono,monospace'>"
        f"Model: CLIP ViT-B/32 (HuggingFace)<br>"
        f"Index: FAISS IndexFlatIP<br>"
        f"Storage: SQLite + .faiss files<br>"
        f"Device: {device_lbl}</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════

st.markdown(
    "<div class='ds-title'>DeepSearch AI</div>"
    "<div class='ds-sub'>Natural Language Video Search · CLIP (Hugging Face) · FAISS · SQLite · OpenCV</div>",
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════
# MODE A — UPLOAD & SEARCH
# ══════════════════════════════════════════════

if input_mode == "📁 Upload Video File":

    # ══════════════════════════════════════════
    # 🆕 UPGRADE 2: Video Library panel
    # ══════════════════════════════════════════

    library = db_list_videos(conn)
    if library:
        with st.expander(f"📚 Video Library — {len(library)} indexed video(s)", expanded=False):
            for vid_id, name, duration, frame_count, indexed_at in library:
                dur_str = _fmt_ts(duration) if duration else "?"
                is_active = (vid_id == st.session_state.active_video_id)
                bcol1, bcol2 = st.columns([5, 1])
                with bcol1:
                    _badge_html = "<span class='badge-indexed'>● ACTIVE</span>" if is_active else "<span class='badge-new'>indexed</span>"
                    st.markdown(
                        f"<div class='vid-lib-row'>"
                        f"<span class='vid-lib-name'>🎥 {name}</span>"
                        f"<span class='vid-lib-meta'>{frame_count} frames · {dur_str} · {indexed_at[:10]}</span>"
                        f"{_badge_html}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with bcol2:
                    if not is_active:
                        if st.button("Load", key=f"lib_{vid_id}", use_container_width=True):
                            info = db_get_video(conn, vid_id)
                            if info and Path(info["path"]).exists():
                                idx = load_faiss(vid_id)
                                if idx:
                                    ts_list = db_get_timestamps(conn, vid_id)
                                    # Reconstruct minimal frames_data (timestamps only, no BGR)
                                    st.session_state.faiss_index      = idx
                                    st.session_state.frames_data      = [(ts, None) for ts in ts_list]
                                    st.session_state.active_video_id  = vid_id
                                    st.session_state.active_video_info = info
                                    st.session_state.results          = []
                                    st.session_state.highlight_cache  = {}
                                    st.session_state.jump_ts          = None
                                    st.session_state.highlight_req    = None
                                    st.success(f"✅ Loaded '{name}' from library (FAISS index restored)")
                                    st.rerun()
                                else:
                                    st.error("FAISS index file missing — re-index this video.")
                            else:
                                st.error("Video file not found on disk. Re-upload to re-index.")

    col_vid, col_search = st.columns([1, 1], gap="large")

    # ── Video column ──────────────────────────────────────────────
    with col_vid:
        st.markdown("### 🎬 Video")
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            vid_id     = video_hash(file_bytes)
            suffix     = Path(uploaded_file.name).suffix

            # Persist video file to DB_DIR so it survives session
            vid_save_path = VID_DIR / f"{vid_id}{suffix}"
            if not vid_save_path.exists():
                vid_save_path.write_bytes(file_bytes)

            # Check if already indexed
            existing = db_get_video(conn, vid_id)

            if st.session_state.active_video_id != vid_id:
                # New video uploaded — auto-load from DB if indexed
                if existing and faiss_path_for(vid_id).exists():
                    idx     = load_faiss(vid_id)
                    ts_list = db_get_timestamps(conn, vid_id)
                    st.session_state.faiss_index      = idx
                    st.session_state.frames_data      = [(ts, None) for ts in ts_list]
                    st.session_state.active_video_id  = vid_id
                    st.session_state.active_video_info = existing
                    st.session_state.results          = []
                    st.session_state.highlight_cache  = {}
                    st.session_state.jump_ts          = None
                    st.session_state.highlight_req    = None
                else:
                    st.session_state.faiss_index     = None
                    st.session_state.frames_data     = None
                    st.session_state.active_video_id = vid_id
                    st.session_state.results         = []

            if existing and faiss_path_for(vid_id).exists():
                st.markdown(
                    f"<span class='badge-indexed'>✅ Already indexed — {existing['frame_count']} frames · {_fmt_ts(existing['duration'] or 0)}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span class='badge-new'>⚡ New video — click Index</span>",
                            unsafe_allow_html=True)

            st.video(uploaded_file)

            # ── Index button handler ─────────────────────────────
            if index_btn:
                model, processor, device = load_model()
                fd, fps, duration = extract_frames(str(vid_save_path), sample_interval)

                with st.spinner("🧠 Building CLIP embeddings..."):
                    embs = embed_frames(fd, model, processor, device)

                # 🆕 Build & SAVE FAISS index
                with st.spinner("💾 Saving FAISS index to disk..."):
                    faiss_idx  = build_faiss(embs)
                    saved_path = save_faiss(faiss_idx, vid_id)

                # 🆕 Save to SQLite
                timestamps = [ts for ts, _ in fd]
                db_save_video(conn, vid_id, uploaded_file.name,
                              str(vid_save_path), duration, fps,
                              len(fd), saved_path, timestamps)

                st.session_state.faiss_index      = faiss_idx
                st.session_state.frames_data      = fd
                st.session_state.active_video_id  = vid_id
                st.session_state.active_video_info = db_get_video(conn, vid_id)
                st.session_state.results          = []
                st.session_state.highlight_cache  = {}
                st.success(
                    f"✅ **{len(fd)} frames** indexed & saved to FAISS + SQLite — "
                    f"next upload of this video loads **instantly**!"
                )
                st.rerun()
        else:
            st.info("👆 Upload a video file in the sidebar to begin.")

    # ── Search column ──────────────────────────────────────────────
    with col_search:
        st.markdown("### 🔎 Search")
        is_ready = st.session_state.faiss_index is not None

        st.markdown(
            "<div style='font-size:11px;color:#1e3040;font-family:Space Mono,monospace;"
            "margin-bottom:6px'>💡 Combine with <b style=color:#00c8f8>+</b> for multi-event search</div>",
            unsafe_allow_html=True,
        )

        examples = [
            "A person in a red jacket",
            "A white car + speeding",
            "Someone running + crowd",
            "Person dropping an object",
            "Fight + two people",
            "Fire + smoke",
        ]
        ex_cols = st.columns(2)
        for i, ex in enumerate(examples):
            with ex_cols[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True, disabled=not is_ready):
                    st.session_state["qprefill"] = ex

        raw_query = st.text_input(
            "Describe what to find:",
            placeholder="e.g.  person + red jacket + running",
            value=st.session_state.get("qprefill", ""),
            key="main_query",
            disabled=not is_ready,
        )

        if raw_query and "+" in raw_query:
            chips = "".join(
                f"<span class='q-chip'>{q.strip()}</span>"
                for q in raw_query.split("+") if q.strip()
            )
            st.markdown(f"<div style='margin:4px 0 10px'>Queries: {chips}</div>",
                        unsafe_allow_html=True)

        search_btn = st.button(
            "🔍 Search Video", use_container_width=True,
            disabled=(not is_ready or not raw_query.strip() if raw_query else True),
            type="primary",
        )

        if search_btn and raw_query and raw_query.strip():
            model, processor, device = load_model()
            t0 = time.time()
            with st.spinner("FAISS + CLIP cosine search..."):
                results, sub_qs = multi_query_search(
                    raw_query, st.session_state.faiss_index,
                    st.session_state.frames_data, model, processor, device,
                    top_k=top_k, combine=combine_key,
                )
            elapsed = (time.time() - t0) * 1000
            st.session_state.results     = results
            st.session_state.sub_queries = sub_qs
            st.success(f"**{len(results)} matches** found in **{elapsed:.0f}ms** (FAISS-accelerated)")

            if results and results[0]["score"] >= alert_threshold:
                st.session_state.alert_log.insert(0, {
                    "time":  datetime.now().strftime("%H:%M:%S"),
                    "query": raw_query[:40],
                    "score": results[0]["score"],
                    "ts":    results[0]["timestamp_fmt"],
                })

    # ══════════════════════════════════════════
    # 🆕 UPGRADE 3: Timeline View
    # ══════════════════════════════════════════

    if st.session_state.results and st.session_state.active_video_info:
        info     = st.session_state.active_video_info
        duration = info.get("duration") or 0
        all_ts   = db_get_timestamps(conn, info["video_id"])
        render_timeline(st.session_state.results, duration, all_ts, alert_threshold)

    # ══════════════════════════════════════════
    # CONFIDENCE RANKING UI (unchanged from v2)
    # ══════════════════════════════════════════

    if st.session_state.results:
        st.divider()
        results  = st.session_state.results
        sub_qs   = st.session_state.sub_queries
        top_sc   = results[0]["score"]
        raw_disp = st.session_state.get("main_query", "")

        h_l, h_r = st.columns([3, 1])
        with h_l:
            if len(sub_qs) > 1:
                chips = "".join(f"<span class='q-chip'>{q}</span>" for q in sub_qs)
                st.markdown(f"### 📍 Multi-Query Results &nbsp; {chips}", unsafe_allow_html=True)
            else:
                st.markdown(f"### 📍 Results for: *\"{raw_disp}\"*")
        with h_r:
            col = _score_color(top_sc)
            st.markdown(
                f"<div style='text-align:right;font-family:Space Mono,monospace;"
                f"font-size:11px;color:#1e3040;margin-top:14px'>"
                f"{len(results)} matches · top "
                f"<span style='color:{col}'>{top_sc*100:.1f}%</span></div>",
                unsafe_allow_html=True,
            )

        if top_sc >= 0.30:
            st.markdown(
                f"<div class='alert-critical'>🚨 HIGH-CONFIDENCE MATCH — "
                f"{top_sc*100:.1f}% at {results[0]['timestamp_fmt']} · {raw_disp[:50]}</div>",
                unsafe_allow_html=True,
            )
        elif top_sc >= 0.24:
            st.markdown(
                f"<div class='alert-warn'>⚠️ POSSIBLE MATCH — "
                f"{top_sc*100:.1f}% at {results[0]['timestamp_fmt']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='alert-ok'>✅ Search complete — no high-confidence events</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Result grid — only show frames when we have BGR data (freshly indexed)
        cols3 = st.columns(3)
        for i, r in enumerate(results):
            with cols3[i % 3]:
                if r["frame_bgr"] is not None:
                    frame_rgb = cv2.cvtColor(r["frame_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(Image.fromarray(frame_rgb), use_container_width=True)
                else:
                    # Loaded from library — no frame preview (embeddings only)
                    st.markdown(
                        f"<div style='background:#0e1520;border:1px solid #1a2535;"
                        f"border-radius:8px;height:90px;display:flex;align-items:center;"
                        f"justify-content:center;font-family:Space Mono,monospace;"
                        f"font-size:10px;color:#1e3040'>⏱ {r['timestamp_fmt']}</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"{_rank_badge(r['rank'])}"
                    f"<div style='font-family:Space Mono,monospace;font-size:18px;"
                    f"font-weight:700;color:#00c8f8;margin-top:5px'>⏱ {r['timestamp_fmt']}</div>",
                    unsafe_allow_html=True,
                )

                col_v = _score_color(r["score"])
                st.markdown(
                    f"<div style='font-family:Space Mono,monospace;font-size:10px;"
                    f"color:#1e3040;display:flex;justify-content:space-between'>"
                    f"<span>Combined score</span>"
                    f"<span style='color:{col_v}'>{r['score']*100:.1f}%</span></div>"
                    f"{_score_bar(r['score'], top_sc)}",
                    unsafe_allow_html=True,
                )

                if len(sub_qs) > 1:
                    for sq, sc in r["per_query"].items():
                        c = _score_color(sc)
                        st.markdown(
                            f"<div style='font-family:Space Mono,monospace;font-size:9px;"
                            f"color:#1e3040;display:flex;justify-content:space-between;margin-top:2px'>"
                            f"<span style='max-width:70%;overflow:hidden;white-space:nowrap'>"
                            f"{sq[:22]}</span>"
                            f"<span style='color:{c}'>{sc*100:.1f}%</span></div>",
                            unsafe_allow_html=True,
                        )

                # Jump + Highlight reel buttons (only when video path available)
                vid_path = None
                if st.session_state.active_video_info:
                    p = st.session_state.active_video_info.get("path")
                    if p and Path(p).exists():
                        vid_path = p

                if vid_path:
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("▶ Jump", key=f"jump_{i}", use_container_width=True):
                            st.session_state.jump_ts = r["timestamp_sec"]
                            st.rerun()
                    with b2:
                        if st.button("🎬 Clip", key=f"clip_{i}", use_container_width=True):
                            st.session_state.highlight_req = r["timestamp_sec"]
                            st.rerun()

        # ── Full video player ──
        if st.session_state.jump_ts is not None and vid_path:
            st.divider()
            st.markdown(f"### 🎬 Full Video — `{_fmt_ts(st.session_state.jump_ts)}`")
            with open(vid_path, "rb") as vf:
                st.video(vf, start_time=int(st.session_state.jump_ts))

        # ── Highlight Reel ──
        if st.session_state.highlight_req is not None and vid_path:
            ts        = st.session_state.highlight_req
            cache_key = f"{ts:.1f}_{clip_duration}"
            st.divider()
            st.markdown(f"### 🎬 Highlight Reel — {_fmt_ts(ts)} ± {clip_duration//2}s")

            if cache_key not in st.session_state.highlight_cache:
                with st.spinner("✂️ Cutting highlight clip..."):
                    cp = extract_highlight_clip(vid_path, ts, 30.0, float(clip_duration))
                st.session_state.highlight_cache[cache_key] = cp

            with open(st.session_state.highlight_cache[cache_key], "rb") as cf:
                st.video(cf)
            st.caption(f"📍 Centred at {_fmt_ts(ts)} · {clip_duration}s duration")

    # ── Alert Log ──
    if st.session_state.alert_log:
        st.divider()
        st.markdown("### 🚨 Auto Alert Log")
        st.caption("Events where top match exceeded the alert threshold")
        hcols = st.columns([1, 3, 1, 1])
        for h, lbl in zip(hcols, ["Time", "Query", "Score", "Timestamp"]):
            h.markdown(f"**{lbl}**")
        for entry in st.session_state.alert_log[:10]:
            c  = _score_color(entry["score"])
            rc = st.columns([1, 3, 1, 1])
            rc[0].markdown(
                f"<span style='font-family:Space Mono,monospace;font-size:11px;"
                f"color:#1e3040'>{entry['time']}</span>", unsafe_allow_html=True)
            rc[1].markdown(f"<span style='font-size:12px'>{entry['query']}</span>",
                           unsafe_allow_html=True)
            rc[2].markdown(
                f"<span style='font-family:Space Mono,monospace;color:{c}'>"
                f"{entry['score']*100:.1f}%</span>", unsafe_allow_html=True)
            rc[3].markdown(
                f"<span style='font-family:Space Mono,monospace;color:#00c8f8'>"
                f"{entry['ts']}</span>", unsafe_allow_html=True)
        if st.button("🗑 Clear Log"):
            st.session_state.alert_log = []
            st.rerun()


# ══════════════════════════════════════════════
# MODE B — LIVE RTSP / IP CAMERA
# ══════════════════════════════════════════════

else:
    st.markdown("### 📡 Live RTSP / IP Camera — Real-Time CLIP Scoring")
    col_cam, col_info = st.columns([1.2, 1], gap="large")

    with col_cam:
        holder = st.empty()
        if rtsp_url and live_btn:
            model, processor, device = load_model()
            lq = st.session_state.get("live_q", "").strip()
            if not lq:
                st.warning("Enter a live watch query in the sidebar.")
            else:
                with st.spinner(f"Connecting to {rtsp_url}..."):
                    cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame_bgr = cap.read()
                    cap.release()

                if not ok:
                    st.error("❌ Cannot read stream. Check your URL.")
                else:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    holder.image(Image.fromarray(frame_rgb),
                                 caption="📸 Live Frame", use_container_width=True)

                    inp_img = processor(images=[Image.fromarray(frame_rgb)],
                                        return_tensors="pt").to(device)
                    inp_txt = processor(text=[lq], return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        raw_ie = model.get_image_features(**inp_img)
                        raw_te = model.get_text_features(**inp_txt)
                        ie = raw_ie if isinstance(raw_ie, torch.Tensor) else raw_ie.pooler_output
                        te = raw_te if isinstance(raw_te, torch.Tensor) else raw_te.pooler_output
                        ie = ie / ie.norm(dim=-1, keepdim=True)
                        te = te / te.norm(dim=-1, keepdim=True)
                        score = float(np.dot(ie.cpu().numpy(), te.cpu().numpy().T).squeeze())

                    st.session_state["live_score"]    = score
                    st.session_state["live_query_lbl"] = lq

                    if score >= alert_threshold:
                        st.session_state.alert_log.insert(0, {
                            "time":  datetime.now().strftime("%H:%M:%S"),
                            "query": lq[:40],
                            "score": score,
                            "ts":    "LIVE",
                        })
        else:
            holder.info("📡 Enter an RTSP URL and click 'Grab & Analyze Frame'.")

    with col_info:
        if "live_score" in st.session_state:
            score = st.session_state["live_score"]
            col   = _score_color(score)
            pct   = max(0.0, min(1.0, (score - 0.1) / 0.35))

            st.markdown("#### 📊 Live Similarity Score")
            st.markdown(
                f"<div style='font-family:Space Mono,monospace;font-size:54px;"
                f"font-weight:700;color:{col};line-height:1'>{score*100:.1f}%</div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)
            st.markdown(f"**Query:** `{st.session_state.get('live_query_lbl','')}`")
            st.markdown(f"**Threshold:** `{alert_threshold*100:.0f}%` | **Raw:** `{score:.4f}`")
            st.markdown("")

            if score >= alert_threshold:
                st.markdown(
                    "<div class='alert-critical'>🚨 ALERT TRIGGERED — Event detected!</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='alert-ok'>✅ No event — below threshold</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("#### 📱 How to connect your phone")
            st.markdown("""
            | App | URL format |
            |-----|------------|
            | Android IP Webcam | `http://phone-ip:8080/video` |
            | DroidCam | `http://phone-ip:4747/video` |
            | Standard RTSP cam | `rtsp://ip:554/stream` |
            """)

    if st.session_state.alert_log:
        st.divider()
        st.markdown("### 🚨 Alert Log")
        for entry in st.session_state.alert_log[:5]:
            c = _score_color(entry["score"])
            st.markdown(
                f"<div class='ds-card'>"
                f"<span style='color:#1e3040;font-size:10px'>{entry['time']}</span> &nbsp;"
                f"<b>{entry['query']}</b> &nbsp;"
                f"<span style='color:{c};font-family:Space Mono,monospace'>"
                f"{entry['score']*100:.1f}%</span></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════

st.divider()
st.markdown(
    "<div style='text-align:center;font-family:Space Mono,monospace;"
    "font-size:9px;color:#111820;padding:6px'>"
    "DeepSearch AI v3.0 · CLIP ViT-B/32 (Hugging Face) · FAISS IndexFlatIP · "
    "SQLite Video Library · NumPy · OpenCV · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
