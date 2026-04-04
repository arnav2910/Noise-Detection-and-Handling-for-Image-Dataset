"""
app.py  –  Streamlit frontend for the Noise Detection & Cleaning Pipeline
Run with:  streamlit run app.py
"""

import io
import os
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import process_image, process_batch

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NoiseGuard – Image Noise Cleaner",
    page_icon="🧹",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #0d0f14; }

h1, h2, h3, .stMarkdown h1 { font-family: 'Space Mono', monospace; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00e5ff, #7c4dff, #ff4081);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.subtitle {
    color: #8892a4;
    font-size: 1.05rem;
    margin-top: -0.5rem;
    margin-bottom: 2rem;
}

.noise-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-clean        { background:#0d2e1a; color:#00e676; border: 1px solid #00e676; }
.badge-gaussian     { background:#1a1a2e; color:#7c4dff; border: 1px solid #7c4dff; }
.badge-salt_pepper  { background:#2e1a00; color:#ff9100; border: 1px solid #ff9100; }
.badge-blur         { background:#002e2e; color:#00e5ff; border: 1px solid #00e5ff; }
.badge-compression  { background:#2e002e; color:#ff4081; border: 1px solid #ff4081; }
.badge-adversarial  { background:#2e0000; color:#ff1744; border: 1px solid #ff1744; }

.report-card {
    background: #131720;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 8px 0;
}

.metric-big {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #00e5ff;
}

.metric-label {
    color: #8892a4;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

div[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e2535;
}

.stProgress > div > div { background: linear-gradient(90deg, #00e5ff, #7c4dff); }

section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1e2535;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

NOISE_COLORS = {
    "clean":       "#00e676",
    "gaussian":    "#7c4dff",
    "salt_pepper": "#ff9100",
    "blur":        "#00e5ff",
    "compression": "#ff4081",
    "adversarial": "#ff1744",
}

NOISE_LABELS = {
    "clean":       "✅ Clean",
    "gaussian":    "〰️ Gaussian Noise",
    "salt_pepper": "⬛ Salt & Pepper",
    "blur":        "🌫️ Blur",
    "compression": "📦 Compression Artifacts",
    "adversarial": "⚠️ Adversarial Perturbation",
}


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    if cv_img.ndim == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def noise_badge(noise_type: str) -> str:
    return f'<span class="noise-badge badge-{noise_type}">{NOISE_LABELS.get(noise_type, noise_type)}</span>'


def images_to_zip(image_pairs: list[tuple[str, Image.Image]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, pil_img in image_pairs:
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            zf.writestr(f"cleaned/{fname}", img_buf.getvalue())
    return buf.getvalue()


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def load_images_from_upload(uploaded_files) -> list[tuple[str, np.ndarray]]:
    images = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix.lower()
        if suffix not in SUPPORTED:
            continue
        pil = Image.open(f)
        images.append((f.name, pil_to_cv(pil)))
    return images


def load_images_from_zip(zip_file) -> list[tuple[str, np.ndarray]]:
    images = []
    with zipfile.ZipFile(zip_file, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() not in SUPPORTED:
                continue
            with zf.open(name) as img_f:
                pil = Image.open(img_f).copy()
                images.append((Path(name).name, pil_to_cv(pil)))
    return images


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    strength = st.select_slider(
        "Cleaning Strength",
        options=["light", "medium", "strong"],
        value="medium",
        help="Stronger cleaning removes more noise but may affect image quality slightly.",
    )
    st.markdown("---")
    st.markdown("### 🎨 Noise Types Detected")
    for nt, label in NOISE_LABELS.items():
        color = NOISE_COLORS[nt]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color}"></div>'
            f'<span style="color:#cdd6e0;font-size:0.88rem">{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        '<div style="color:#4a5568;font-size:0.78rem">NoiseGuard v1.0<br>'
        'Detect · Clean · Report</div>',
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🧹 NoiseGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload images or a ZIP dataset — we detect noise, clean it, and give you a full report.</div>',
    unsafe_allow_html=True,
)

# ── Upload Mode ────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["🖼️ Single / Multiple Images", "🗜️ ZIP Dataset"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Individual images
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        "Drop images here",
        type=list(SUPPORTED),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        images = load_images_from_upload(uploaded)

        if not images:
            st.warning("No supported images found.")
        else:
            with st.spinner("🔍 Analysing and cleaning…"):
                results, report = process_batch(images, strength=strength)

            # ── Summary row ──────────────────────────────────────────────────
            st.markdown("### 📊 Batch Report")
            cols = st.columns(len(NOISE_LABELS))
            for col, (nt, label) in zip(cols, NOISE_LABELS.items()):
                count = report["counts"].get(nt, 0)
                pct   = report["percentages"].get(nt, 0)
                color = NOISE_COLORS[nt]
                col.markdown(
                    f'<div class="report-card" style="border-color:{color}33">'
                    f'<div class="metric-big" style="color:{color}">{count}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'<div style="color:#4a5568;font-size:0.8rem;margin-top:4px">{pct}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Per-image results ────────────────────────────────────────────
            st.markdown("### 🖼️ Results")
            cleaned_pils = []

            for fname, res in results:
                with st.expander(f"**{fname}** — {NOISE_LABELS.get(res.noise_type, res.noise_type)}", expanded=True):
                    c1, c2, c3 = st.columns([1, 1, 1])

                    # find original PIL
                    orig_pil = None
                    for uf in uploaded:
                        if uf.name == fname:
                            orig_pil = Image.open(uf)
                            break

                    with c1:
                        st.markdown("**Original**")
                        if orig_pil:
                            st.image(orig_pil, use_container_width=True)

                    with c2:
                        st.markdown("**Cleaned**")
                        clean_pil = cv_to_pil(res.cleaned_image)
                        st.image(clean_pil, use_container_width=True)
                        cleaned_pils.append((fname, clean_pil))

                    with c3:
                        st.markdown("**Analysis**")
                        st.markdown(noise_badge(res.noise_type), unsafe_allow_html=True)
                        st.markdown(f"Confidence: **{res.confidence:.0%}**")
                        st.markdown("**Raw Features:**")
                        for k, v in res.details.items():
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;'
                                f'color:#8892a4;font-size:0.82rem;padding:2px 0">'
                                f'<span>{k.replace("_"," ").title()}</span>'
                                f'<span style="color:#cdd6e0;font-family:monospace">{v}</span></div>',
                                unsafe_allow_html=True,
                            )

            # ── Download button ──────────────────────────────────────────────
            if cleaned_pils:
                zip_bytes = images_to_zip(cleaned_pils)
                st.download_button(
                    label="⬇️ Download All Cleaned Images (.zip)",
                    data=zip_bytes,
                    file_name="noiseguard_cleaned.zip",
                    mime="application/zip",
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – ZIP upload
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    zip_upload = st.file_uploader(
        "Drop your ZIP dataset here",
        type=["zip"],
        label_visibility="collapsed",
    )

    if zip_upload:
        with st.spinner("📦 Extracting ZIP…"):
            images = load_images_from_zip(zip_upload)

        if not images:
            st.warning("No supported images found inside the ZIP.")
        else:
            st.info(f"Found **{len(images)}** image(s) in the archive.")

            with st.spinner(f"🔍 Running pipeline on {len(images)} images…"):
                results, report = process_batch(images, strength=strength)

            # ── Summary ──────────────────────────────────────────────────────
            st.markdown("### 📊 Dataset Report")

            total = report["total_images"]
            st.markdown(
                f'<div class="report-card">'
                f'<div class="metric-big">{total}</div>'
                f'<div class="metric-label">Total Images Processed</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            items = list(NOISE_LABELS.items())
            for i, (nt, label) in enumerate(items):
                count = report["counts"].get(nt, 0)
                pct   = report["percentages"].get(nt, 0)
                color = NOISE_COLORS[nt]
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="report-card" style="border-color:{color}55">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<div>'
                        f'<div class="metric-big" style="color:{color}">{count}</div>'
                        f'<div class="metric-label">{label}</div>'
                        f'</div>'
                        f'<div style="font-size:1.4rem;color:{color};font-family:monospace">{pct}%</div>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Progress bar visual ───────────────────────────────────────────
            st.markdown("### 🎯 Noise Distribution")
            for nt, label in NOISE_LABELS.items():
                count = report["counts"].get(nt, 0)
                pct   = report["percentages"].get(nt, 0)
                color = NOISE_COLORS[nt]
                if count > 0:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:12px;margin:6px 0">'
                        f'<div style="width:120px;color:#8892a4;font-size:0.82rem;text-align:right">{label}</div>'
                        f'<div style="flex:1;background:#1e2535;border-radius:4px;height:18px;overflow:hidden">'
                        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:4px;'
                        f'transition:width 0.8s ease"></div></div>'
                        f'<div style="width:40px;color:{color};font-family:monospace;font-size:0.82rem">{pct}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Sample previews ───────────────────────────────────────────────
            st.markdown("### 🖼️ Sample Previews (first 12)")
            preview_results = results[:12]
            n_cols = 4
            rows = [preview_results[i:i+n_cols] for i in range(0, len(preview_results), n_cols)]

            for row in rows:
                cols = st.columns(n_cols)
                for col, (fname, res) in zip(cols, row):
                    with col:
                        clean_pil = cv_to_pil(res.cleaned_image)
                        st.image(clean_pil, use_container_width=True)
                        color = NOISE_COLORS.get(res.noise_type, "#888")
                        st.markdown(
                            f'<div style="text-align:center;font-size:0.75rem;margin-top:2px">'
                            f'<span style="color:{color}">● {NOISE_LABELS.get(res.noise_type,"?")}</span><br>'
                            f'<span style="color:#4a5568">{fname[:20]}</span></div>',
                            unsafe_allow_html=True,
                        )

            # ── Download ──────────────────────────────────────────────────────
            cleaned_pils = [(fname, cv_to_pil(res.cleaned_image)) for fname, res in results]
            zip_bytes = images_to_zip(cleaned_pils)
            st.download_button(
                label=f"⬇️ Download All {len(results)} Cleaned Images (.zip)",
                data=zip_bytes,
                file_name="noiseguard_cleaned_dataset.zip",
                mime="application/zip",
            )
