# 🧹 NoiseGuard — Image Noise Detection & Cleaning Pipeline

Detect and remove multiple types of image noise automatically, with a beautiful web frontend.

---

## 📁 Project Structure

```
noise_project/
│
├── app.py                          ← Streamlit frontend (run this)
├── pipeline.py                     ← Orchestrator: detect → defend
├── requirements.txt
│
├── detectors/
│   └── noise_detector.py           ← Detects noise type using image analysis
│
└── defenders/
    ├── defend_gaussian.py          ← Removes Gaussian noise (Gaussian blur + NLM)
    ├── defend_salt_pepper.py       ← Removes salt & pepper (Median filter)
    ├── defend_blur.py              ← Sharpens blurry images (Unsharp masking)
    ├── defend_compression.py       ← Fixes JPEG artifacts (Bilateral filter)
    └── defend_adversarial.py       ← Removes adversarial perturbations (Ensemble)
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 🔍 How It Works

### Detection Pipeline (detectors/noise_detector.py)

For each image, five features are extracted:

| Feature | What it measures |
|---|---|
| `pixel_variance` | Overall pixel spread — high = noisy |
| `laplacian_variance` | Edge sharpness — low = blurry |
| `salt_pepper_ratio` | Fraction of extreme (0 or 255) pixels |
| `high_freq_energy` | FFT energy in outer frequency ring |
| `block_artifact_score` | Discontinuity across 8-pixel JPEG boundaries |

These are combined in a decision tree to classify each image as one of:
- `clean` — no significant noise
- `gaussian` — random Gaussian noise
- `salt_pepper` — impulse/spike noise
- `blur` — motion or defocus blur
- `compression` — JPEG/codec block artifacts
- `adversarial` — crafted perturbation (FGSM/PGD-style)

### Defense Modules

Each noise type gets its own dedicated defender:

#### Gaussian → `defend_gaussian.py`
- Gaussian blur to suppress random noise
- Non-Local Means (NLM) for residual cleanup
- Preserves edges well

#### Salt & Pepper → `defend_salt_pepper.py`
- Median filter (best choice for impulse noise)
- Applied per channel to avoid colour bleeding

#### Blur → `defend_blur.py`
- Unsharp Masking (USM) to recover lost edges
- Threshold prevents over-sharpening clean regions

#### Compression → `defend_compression.py`
- Bilateral filter (smooths flat regions, keeps real edges)
- Targets 8×8 block discontinuities from JPEG encoding

#### Adversarial → `defend_adversarial.py`
- **Ensemble of three methods:**
  1. JPEG re-compression — destroys high-frequency adversarial signal
  2. Bit-depth reduction — quantises away subtle pixel changes
  3. Gaussian blur — low-pass filter removes fine perturbations
- Weighted average: 50% JPEG + 25% bit-depth + 25% blur

---

## 🖥️ Frontend Features

- **Single / Multiple Image upload** — side-by-side before/after comparison
- **ZIP Dataset upload** — processes entire datasets at once
- **Noise Distribution Chart** — visual breakdown of noise types
- **Per-image analysis** — shows raw feature values
- **Download cleaned images** — as a ZIP archive
- **Cleaning strength control** — light / medium / strong

---

## 📊 Output Report (per batch)

```json
{
  "total_images": 100,
  "counts": {
    "clean": 40,
    "gaussian": 25,
    "salt_pepper": 15,
    "blur": 10,
    "compression": 7,
    "adversarial": 3
  },
  "percentages": {
    "clean": 40.0,
    "gaussian": 25.0,
    ...
  }
}
```

---

## ⚠️ Limitations

- **Adversarial detection** is heuristic — adaptive attacks can evade it
- **Blur reversal** is approximate; true deconvolution requires the blur kernel
- **Mixed noise** (e.g., Gaussian + compression) is classified as the dominant type
- No GPU required — all methods use CPU-based OpenCV

---

## 🔮 Possible Extensions

- Replace heuristic detector with a trained CNN classifier
- Add autoencoder-based denoising for learned reconstruction
- Add adversarial attack generation (FGSM/PGD) for testing
- Plug in a robustness evaluation module (accuracy before vs after)
- Add PSNR / SSIM metrics comparing original vs cleaned

---

## 🧾 Resume Description

> "Developed a full-stack image noise detection and cleaning system supporting Gaussian, salt-and-pepper, blur, compression, and adversarial perturbation noise types. Built dedicated defense modules per noise class and an interactive Streamlit frontend for dataset upload, automated cleaning, and noise distribution reporting."
