"""
app.py — Streamlit interface for the Fungal ITS LCPN Identifier
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import io
from datetime import datetime

# Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="FungiClass — Hierarchical ITS Identifier",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #222;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #aaa !important;
    font-size: 13px;
}

/* Main area */
.main .block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* Header */
.app-header {
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin: 0;
}
.app-subtitle {
    font-size: 13px;
    color: #666;
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Level badge */
.level-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Prediction path card */
.pred-card {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.pred-card.passthrough {
    border-left: 3px solid #6366f1;
    background: #fafaff;
}
.pred-card.stop {
    border-left: 3px solid #f59e0b;
    background: #fffcf0;
    opacity: 0.8;
}
.pred-card.accepted {
    border-left: 3px solid #10b981;
}

/* Confidence bar */
.conf-bar-bg {
    background: #efefef;
    border-radius: 3px;
    height: 6px;
    width: 100px;
    display: inline-block;
    vertical-align: middle;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
}

/* Stats card */
.stat-box {
    background: #0d0d0d;
    color: #fff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1.1;
}
.stat-lbl {
    font-size: 11px;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Result table */
.result-table th {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #666;
    font-weight: 500;
}

/* Info box */
.info-pill {
    display: inline-block;
    background: #1432db;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
    color: #444;
    font-family: 'IBM Plex Mono', monospace;
}

button[kind="primary"] {
    background: #0d0d0d !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.02em !important;
}
</style>
""", unsafe_allow_html=True)

# Constants ───────────────────────────────────────────────────────────────
MODEL_PATH = "models_hier"
LEVELS     = ["phylum", "class", "order", "family", "genus", "species"]
LEVEL_COLORS = {
    "phylum":  "#6366f1",
    "class":   "#8b5cf6",
    "order":   "#0ea5e9",
    "family":  "#10b981",
    "genus":   "#f59e0b",
    "species": "#ef4444",
}

# ── Load models (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classifier models…")
def load_resources():
    try:
        tv = joblib.load(f"{MODEL_PATH}/tfidf_global.pkl")
    except FileNotFoundError:
        st.error("Model files not found. Make sure `models_hier/` is in the working directory.")
        st.stop()

    pt_path = f"{MODEL_PATH}/passthrough_map.json"
    pt_map  = json.load(open(pt_path)) if os.path.exists(pt_path) else {}

    try:
        import variables
        thresholds = variables.THRESHOLDS
        k          = variables.K_MER_SIZE
    except ImportError:
        thresholds = {"phylum":0.76,"class":0.90,"order":0.92,
                      "family":0.88,"genus":0.84,"species":0.92}
        k = 6

    return tv, pt_map, thresholds, k

tv, PASSTHROUGH_MAP, THRESHOLDS, K = load_resources()

MODEL_CACHE: dict = {}

def load_model(level, parent_value=None):
    cache_key = "root" if level == "phylum" else f"{level}_{parent_value}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    if level == "phylum":
        mp, lp = f"{MODEL_PATH}/xgb_root.pkl", f"{MODEL_PATH}/le_root.pkl"
    else:
        if parent_value is None:
            return None, None
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(parent_value))
        mp, lp = f"{MODEL_PATH}/xgb_{level}_{safe}.pkl", f"{MODEL_PATH}/le_{level}_{safe}.pkl"
    if not os.path.exists(mp):
        MODEL_CACHE[cache_key] = (None, None)
        return None, None
    model, le = joblib.load(mp), joblib.load(lp)
    MODEL_CACHE[cache_key] = (model, le)
    return model, le

# K-mer  ───────────────────────────────────────────────────────────
IUPAC = {"A":"A","C":"C","G":"G","T":"T"}

def clean_seq(seq):
    return "".join(IUPAC.get(b, "0") for b in seq.upper())

def seq_to_kmers(seq, k=6):
    seq = clean_seq(seq)
    return " ".join(seq[i:i+k] for i in range(len(seq)-k+1) if "0" not in seq[i:i+k])

# predict ────────────────────────────────────────────────────────────
def predict(sequence: str):
    kmers_str = seq_to_kmers(sequence, K)
    if not kmers_str.strip():
        return []
    x = tv.transform([kmers_str])
    path, current_parent = [], None
    for i, level in enumerate(LEVELS):
        if i > 0:
            pt_val = PASSTHROUGH_MAP.get(f"{LEVELS[i-1]}->{level}", {}).get(current_parent)
            if pt_val is not None:
                path.append({"level": level, "label": pt_val, "conf": 1.0, "status": "PASSTHROUGH"})
                current_parent = pt_val
                continue
        model, le = load_model(level, current_parent)
        if model is None:
            path.append({"level": level, "label": None, "conf": 0.0, "status": "NO_MODEL"})
            break
        probs    = model.predict_proba(x)[0]
        best_idx = int(np.argmax(probs))
        conf     = float(probs[best_idx])
        label    = le.inverse_transform([best_idx])[0]
        if conf < THRESHOLDS[level]:
            path.append({"level": level, "label": label, "conf": conf, "status": "LOW_CONF"})
            break
        path.append({"level": level, "label": label, "conf": conf, "status": "OK"})
        current_parent = label
    return path

def format_label(label):
    return label.replace("_", " ") if label else "—"

def deepest(path):
    for step in reversed(path):
        if step["status"] in ("OK", "PASSTHROUGH"):
            return step["level"], step["label"]
    return None, None

# ── UI helpers ──────────────────────────────────────────────────────────────
def conf_bar_html(conf, color="#10b981"):
    pct = int(conf * 100)
    return (
        f'<div class="conf-bar-bg">'
        f'<div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div>'
        f'</div> <span style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#444">'
        f'{pct}%</span>'
    )

def render_path(path):
    for step in path:
        lvl    = step["level"]
        label  = format_label(step["label"]) if step["label"] else "—"
        conf   = step["conf"]
        status = step["status"]
        color  = LEVEL_COLORS.get(lvl, "#888")

        if status == "PASSTHROUGH":
            card_class = "pred-card passthrough"
            badge_html = '<span style="font-size:11px;color:#6366f1;font-family:\'IBM Plex Mono\',monospace">passthrough</span>'
            conf_html  = '<span style="font-size:12px;color:#6366f1;">deterministic</span>'
        elif status == "LOW_CONF":
            card_class = "pred-card stop"
            badge_html = '<span style="font-size:11px;color:#f59e0b;font-family:\'IBM Plex Mono\',monospace">⚠ below threshold</span>'
            conf_html  = conf_bar_html(conf, "#f59e0b")
        elif status == "NO_MODEL":
            card_class = "pred-card stop"
            badge_html = '<span style="font-size:11px;color:#999;font-family:\'IBM Plex Mono\',monospace">no model</span>'
            conf_html  = ""
        else:
            card_class = "pred-card accepted"
            badge_html = ""
            conf_html  = conf_bar_html(conf, color)

        st.markdown(f"""
        <div class="{card_class}">
          <span class="level-badge" style="background:{color}22;color:{color}">{lvl}</span>
          <span style="font-size:14px;font-weight:500;flex:1">{label}</span>
          {conf_html}
          {badge_html}
        </div>
        """, unsafe_allow_html=True)

# Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:500;letter-spacing:-0.01em">
        🍄 FungiClass
      </div>
      <div style="font-size:11px;color:#666;margin-top:4px;text-transform:uppercase;letter-spacing:0.05em">
        ITS Hierarchical Identifier
      </div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["Single sequence", "Batch FASTA", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#FF4B4B;line-height:1.8">
      <div><span class="info-pill">Model</span> LCPN · XGBoost</div>
      <div style="margin-top:6px"><span class="info-pill">Marker</span> ITS (fungal)</div>
      <div style="margin-top:6px"><span class="info-pill">Training</span> CBS ITS 2025</div>
      <div style="margin-top:6px"><span class="info-pill">Species</span> 9,975 known</div>
      <div style="margin-top:6px"><span class="info-pill">Levels</span> 6 (phylum→species)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:10px;color:#555;line-height:1.7">
      Thresholds are calibrated for <b style="color:#aaa">high precision</b>
      (clinical context). The classifier abstains rather than
      guessing when confidence is insufficient.
    </div>
    """, unsafe_allow_html=True)

# Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1 class="app-title">FungiClass</h1>
  <p class="app-subtitle">Hierarchical taxonomic identification of fungal ITS sequences · CBS ITS 2025</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 - SINGLE SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════
if mode == "Single sequence":

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Input sequence")
        seq_input = st.text_area(
            "Paste your ITS sequence (ATCG…)",
            height=200,
            placeholder="ATCGGGTTAGCTATCGATCGATCGATCG…",
            label_visibility="collapsed"
        )

        run_btn = st.button("▶  Identify", type="primary", use_container_width=True)

        st.markdown("""
        <div style="font-size:12px;color:#888;margin-top:0.8rem;line-height:1.7">
          <b>Tips</b><br>
          · Optimal length: 400–800 bp<br>
          · Shorter sequences (&lt;200 bp) may not reach species level<br>
          · Ambiguous bases (N, R, Y…) are automatically excluded from k-mers
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if run_btn and seq_input.strip():
            seq_clean = re.sub(r"\s+", "", seq_input.strip().upper())
            # Remove FASTA header if pasted with >
            if seq_clean.startswith(">"):
                lines = seq_input.strip().split("\n")
                seq_clean = "".join(l for l in lines if not l.startswith(">"))
                seq_clean = re.sub(r"\s+", "", seq_clean.upper())

            if len(seq_clean) < 50:
                st.warning("Sequence too short (< 50 bp). Please provide a longer ITS sequence.")
            else:
                with st.spinner("Running LCPN identification…"):
                    path = predict(seq_clean)

                if not path:
                    st.error("Classification failed — sequence produced no valid k-mers.")
                else:
                    d_level, d_label = deepest(path)
                    n_accepted = sum(1 for s in path if s["status"] in ("OK","PASSTHROUGH"))

                    # Summary stats
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="stat-box">
                          <div class="stat-val">{n_accepted}</div>
                          <div class="stat-lbl">levels reached</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        deepest_idx = LEVELS.index(d_level) + 1 if d_level else 0
                        st.markdown(f"""
                        <div class="stat-box">
                          <div class="stat-val">{deepest_idx}/6</div>
                          <div class="stat-lbl">depth score</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        avg_conf = np.mean([s["conf"] for s in path if s["status"]=="OK"])
                        st.markdown(f"""
                        <div class="stat-box">
                          <div class="stat-val">{avg_conf*100:.0f}%</div>
                          <div class="stat-lbl">avg confidence</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    st.markdown("#### Identification path")
                    render_path(path)

                    if d_label:
                        st.markdown(f"""
                        <div style="margin-top:1rem;padding:0.8rem 1rem;background:#0d0d0d;
                             border-radius:6px;font-family:'IBM Plex Mono',monospace;">
                          <span style="font-size:11px;color:#666;text-transform:uppercase;
                                letter-spacing:0.05em">Best prediction</span><br>
                          <span style="font-size:1rem;color:#fff">{format_label(d_label)}</span>
                          <span style="font-size:12px;color:#666;margin-left:8px">at {d_level} level</span>
                        </div>
                        """, unsafe_allow_html=True)

        elif run_btn:
            st.info("Please paste a sequence first.")
        else:
            st.markdown("""
            <div style="height:200px;display:flex;align-items:center;justify-content:center;
                 color:#ccc;font-size:13px;border:1px dashed #e0e0e0;border-radius:8px;">
              Identification results will appear here
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH FASTA
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Batch FASTA":

    st.markdown("#### Batch identification")
    st.markdown("""
    <p style="color:#666;font-size:13px;margin-bottom:1rem">
    Upload a FASTA file to classify multiple sequences at once.
    Results are downloadable as CSV.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload FASTA file", type=["fasta","fa","fna","txt"])

    if uploaded is not None:
        # Parse FASTA
        content   = uploaded.read().decode("utf-8", errors="ignore")
        sequences = {}
        current_id = None
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line.upper()

        n_seq = len(sequences)
        if n_seq == 0:
            st.error("No sequences found. Make sure the file is in FASTA format.")
        else:
            st.markdown(f"""
            <div style="padding:0.6rem 1rem;background:#25cc3e;border-radius:6px;
                 font-size:13px;margin-bottom:1rem;">
              Found <b>{n_seq}</b> sequences in <code>{uploaded.name}</code>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  Run batch identification", type="primary"):
                results  = []
                progress = st.progress(0)
                status   = st.empty()

                for i, (seq_id, seq) in enumerate(sequences.items()):
                    status.markdown(f"<span style='font-size:12px;color:#666'>"
                                    f"Classifying {i+1}/{n_seq}: {seq_id[:40]}…</span>",
                                    unsafe_allow_html=True)
                    progress.progress((i + 1) / n_seq)

                    seq_clean = re.sub(r"\s+", "", seq)
                    path      = predict(seq_clean) if len(seq_clean) >= 50 else []
                    d_level, d_label = deepest(path)

                    row = {"sequence_id": seq_id, "length_bp": len(seq_clean)}
                    for level in LEVELS:
                        step = next((s for s in path if s["level"] == level), None)
                        if step and step["status"] in ("OK", "PASSTHROUGH"):
                            row[level]              = format_label(step["label"])
                            row[f"conf_{level}"] = f"{round(step['conf'] * 100, 1)}%"
                            row[f"status_{level}"]  = step["status"]
                        else:
                            row[level]              = ""
                            row[f"conf_{level}"]    = ""
                            row[f"status_{level}"]  = step["status"] if step else "not_reached"

                    row["deepest_level"]      = d_level or "none"
                    row["best_prediction"]    = format_label(d_label) if d_label else "unclassified"
                    results.append(row)

                status.empty()
                progress.empty()

                df_res = pd.DataFrame(results)

                #Summary stats
                st.markdown("#### Results summary")
                n_species = (df_res["deepest_level"] == "species").sum()
                n_genus   = (df_res["deepest_level"] == "genus").sum()
                n_unc     = (df_res["deepest_level"] == "none").sum()
                avg_depth = df_res["deepest_level"].map(
                    {l: i+1 for i, l in enumerate(LEVELS)}
                ).fillna(0).mean()

                c1, c2, c3, c4 = st.columns(4)
                for col, val, lbl in [
                    (c1, n_seq,    "sequences"),
                    (c2, n_species, "to species"),
                    (c3, n_genus,  "to genus"),
                    (c4, n_unc,    "unidentified"),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="stat-box">
                          <div class="stat-val">{val}</div>
                          <div class="stat-lbl">{lbl}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:0.8rem;font-size:13px;color:#666">
                  Average depth reached: <b>{avg_depth:.2f} / 6</b>
                </div>
                """, unsafe_allow_html=True)

                #Distribution chart 
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                depth_counts = df_res["deepest_level"].value_counts()
                ordered = {l: int(depth_counts.get(l, 0)) for l in LEVELS + ["none"]}
                chart_df = pd.DataFrame({
                    "Level":  list(ordered.keys()),
                    "Count":  list(ordered.values()),
                })
                st.bar_chart(chart_df.set_index("Level"), color="#0d0d0d")

                #Table preview 
                st.markdown("#### Preview (first 50 rows)")
                display_cols = ["sequence_id","length_bp","deepest_level","best_prediction"] + [col for l in LEVELS for col in [l, f"conf_{l}"]]
                st.dataframe(
                    df_res[display_cols].head(50),
                    use_container_width=True,
                    hide_index=True,
                )

                #Download
                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="⬇  Download full results (CSV)",
                    data=csv_bytes,
                    file_name=f"LCPN_results_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "About":

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        #### About FungiClass

        **FungiClass** is a hierarchical taxonomic identifier for fungal ITS sequences,
        built on the Local Classifier Per Parent Node (LCPN) architecture.

        Unlike flat classifiers that predict species directly, FungiClass traverses
        a six-level taxonomic hierarchy — from phylum down to species — using a
        dedicated XGBoost model at each node. Identification halts when confidence
        falls below a calibrated threshold, returning the deepest reliable prediction
        rather than a forced guess.

        ---

        #### Architecture

        | Component | Detail |
        |-----------|--------|
        | Feature extraction | 6-mer TF-IDF on cleaned ITS sequences |
        | Classifier | XGBoost (one model per taxonomic node) |
        | Hierarchy | LCPN — Local Classifier Per Parent Node |
        | Passthrough | Deterministic for monophyletic nodes |
        | Thresholds | Calibrated via knee-point on held-out set |
        | Training data | CBS ITS 2025 (~50,000 sequences) |
        | Known taxa | 9,975 species across 6 taxonomic levels |

        ---

        #### Threshold policy

        Thresholds are calibrated for **high-precision** use (clinical/diagnostic context).
        The system abstains rather than returning low-confidence predictions.
        For environmental screening with higher coverage, thresholds can be lowered
        in `variables.py`.

        ---

        #### Citation

        If you use FungiClass in your research, please cite:

        > *[Author], [Year]. FungiClass: a hierarchical LCPN identifier for fungal ITS
        > sequences based on the CBS ITS 2025 database. [Journal/Thesis].*

        """)

    with col2:
        st.markdown("#### Performance (CBS ITS 2025 test set)")

        perf_data = {
            "Level":    ["Phylum","Class","Order","Family","Genus","Species"],
            "Accuracy": [0.9455, 0.9387, 0.9456, 0.9454, 0.9435, 0.9370],
            "Coverage": [2072,   1861,   1765,   1668,   1540,   809],
        }
        df_perf = pd.DataFrame(perf_data)

        for _, row in df_perf.iterrows():
            lvl   = row["Level"].lower()
            color = LEVEL_COLORS.get(lvl, "#888")
            pct   = int(row["Accuracy"] * 100)
            st.markdown(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;
                   font-size:12px;margin-bottom:4px">
                <span style="font-weight:500">{row['Level']}</span>
                <span style="color:#666">{pct}% acc · {row['Coverage']} seqs</span>
              </div>
              <div style="background:#f0f0f0;border-radius:3px;height:8px;overflow:hidden">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:11px;color:#888;margin-top:1rem;line-height:1.7">
          Test set: 10% stratified split (2,175 sequences).<br>
          Coverage decreases at finer levels due to threshold filtering
          and monotypic genera not modelled at species level.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Taxonomic coverage")
        st.markdown("""
        <div style="font-size:12px;color:#444;line-height:2">
          <div>🟣 Ascomycota — 29,449 sequences</div>
          <div>🔵 Basidiomycota — 9,048 sequences</div>
          <div>🟢 Mucoromycota — 1,288 sequences</div>
          <div>⚫ Other phyla — 3,696 sequences</div>
        </div>
        """, unsafe_allow_html=True)
