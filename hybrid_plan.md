# Hybrid DnCNN + ESRGAN Training & Evaluation Plan (Ultrasound Enhancement)

This document describes the **complete, correct, medically-safe procedure** for building, training, and evaluating a *hybrid image enhancement system* composed of:

- **Stage 1 — DnCNN (Denoising)**
- **Stage 2 — ESRGAN (Refinement / Enhancement)**
- **Hybrid Output = ESRGAN(DnCNN(original))**

This plan ensures:
- No distribution mismatch
- Stable ESRGAN training for a small medical dataset (400 images)
- Reliable hybrid evaluation (PSNR / SSIM)
- Clean comparison between methods

---

# 1. Project Structure

```
project/
│
├── data/
│   ├── normal/            # original ultrasound images
│   ├── denoised/          # ground‑truth reference images
│   ├── dncnn_out/         # generated later using trained DnCNN
│
├── dncnn_model/           # trained DnCNN weights
├── esrgan_model/          # trained ESRGAN weights (later)
│
├── hybrid_training.py
├── esrgan.py
└── models.py
```

---

# 2. Step-by-Step Plan

## STEP 1 — Train DnCNN (already completed)

Training objective:
```
Input:   normal image (noisy)
Target:  denoised reference
```

This model will serve as the *first stage* of the hybrid pipeline.

No changes needed.

---

# STEP 2 — Generate “DnCNN Output Dataset”

This step **avoids domain mismatch** between training and testing in ESRGAN.

We use the trained DnCNN to process all “normal” images and save the outputs.

### 2.1. Run DnCNN over all images:

```python
for img in data/normal/*:
    y = dncnn(img)   # normalized to [0,1]
    save y to data/dncnn_out/   # same filename
```

### 2.2. Now the ESRGAN dataset becomes:

```
ESRGAN input  = data/dncnn_out/
ESRGAN target = data/denoised/
```

This ensures ESRGAN sees **exactly the type of inputs** it will receive in the hybrid.

---

# STEP 3 — Retrain ESRGAN (with improved stability)

GANs collapse easily on medical images, so the training procedure must be adapted.

### ESRGAN training configuration:

- `scale = 1`
- `nf = 32`
- `nb = 8–12`
- `use_tanh = False`
- losses:
  - L1 loss
  - Perceptual loss (VGG)
  - SSIM loss (very important)
- GAN phase starts only **after 20–30 epochs** of stable optimization
- No cropping in early epochs
- No RGB (grayscale only)

### ESRGAN training goal:
```
Input:   DnCNN output  (dncnn_out/)
Target:  denoised reference  (denoised/)
```

This makes ESRGAN a **refinement network** rather than a deblurring/hallucinating network.

---

# STEP 4 — Build the Hybrid Enhancer

The hybrid enhancer simply chains the networks:

```
Original → DnCNN → ESRGAN → Enhanced Output
```

### Pseudocode:

```python
class HybridEnhancer:
    def __init__(self, dncnn_path, esrgan_path):
        load DnCNN
        load ESRGAN

    def enhance(self, img):
        x = preprocess_to_01(img)
        d = dncnn(x)                  # denoising
        e = esrgan(d)                 # refinement
        return e
```

---

# STEP 5 — Evaluation (Quantitative)

Compare the following to the ground truth (`denoised`):

- **Original**
- **DnCNN**
- **ESRGAN-only** (optional)
- **Hybrid (DnCNN → ESRGAN)**

Using:

- **PSNR**
- **SSIM**
- **Visual inspection**

### Evaluation criteria:

| Method | Expected Result |
|--------|-----------------|
| Original → denoised | low PSNR / low SSIM |
| DnCNN → denoised | moderate PSNR / SSIM |
| ESRGAN-only → denoised | inconsistent results |
| Hybrid → denoised | highest PSNR/SSIM (target) |

The improvement of the hybrid over DnCNN validates the effectiveness of using **both** models.

---

# STEP 6 — Final Report (for publication / research)

Your final research output should include:

1. Description of dataset (400 paired images)
2. Model architecture summary
3. Hybrid pipeline diagram
4. Results table:

```
Method        | PSNR  | SSIM
--------------|-------|------
Original      |   -   |   -
DnCNN         |   -   |   -
ESRGAN        |   -   |   -
Hybrid        |   -   |   -
```

5. Visual examples (side-by-side)
6. Discussion about stability, limits, and future work

---

# 7. Summary

The correct hybrid training workflow:

```
Train DnCNN → Generate dncnn_out/ → Train ESRGAN on (dncnn_out → denoised) → Hybrid Inference → Evaluation
```

This ensures:
- No distribution mismatch
- ESRGAN doesn’t collapse
- Accurate scientific comparison
- Valid and publishable improvement results

---

# 8. Next Actions

You can request:

- **The script to generate dncnn_out/**
- **A rewritten ESRGAN training script tuned for ultrasound**
- **Hybrid inference class**
- **Evaluation script (PSNR/SSIM)**

All fully integrated with your current code.

