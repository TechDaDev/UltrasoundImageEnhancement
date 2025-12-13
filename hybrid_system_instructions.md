# Hybrid Ultrasound Enhancement System  
## DnCNN → ESRGAN (Using Existing DnCNN Script & Model)

This document describes **exactly how to build and use the hybrid system** that combines:

1. **DnCNN** (your already-trained denoiser)
2. **ESRGAN** (to refine DnCNN outputs)
3. **Hybrid pipeline:** `Hybrid(x) = ESRGAN(DnCNN(x))`

All instructions are written to match your current codebase, especially the `denoise_images.py` script and its saved `.h5` model. fileciteturn4file0  

---

## 1. Current DnCNN Setup (What You Already Have)

You already use `denoise_images.py` to denoise all ultrasound images in a folder using a trained DnCNN model.

Key points from that script:

- **Input folder (default):** `./normal/`  
  (Contains original noisy ultrasound images.)

- **Output folder (default):** `./denoise_images/`  
  (Where denoised images will be written.)

- **Model path (example):**  
  `./snapshot/save_DnCNN_sigma10_2025-08-10-03-42-53/model_100.h5`  
  (Your trained DnCNN model in Keras `.h5` format.)

- **Processing:**
  - Images are converted to **grayscale**.
  - Normalized to **[0,1]**.
  - Padded to multiples of 4 (reflection pad), then passed through DnCNN.
  - Output is clipped to `[0,1]`, then saved as `uint8` (`0–255`).

- **Command-line usage example:**

```bash
python denoise_images.py     --input ./normal     --output ./denoise_images     --model ./snapshot/save_DnCNN_sigma10_2025-08-10-03-42-53/model_100.h5     --save_mode denoised
```

Using `--save_mode denoised` ensures you save **only the DnCNN result**, not the side‑by‑side composite.

This DnCNN output is what we will feed into **ESRGAN** for the hybrid system.

---

## 2. Target Directory Structure for the Hybrid System

To keep things clean, use this structure:

```text
project_root/
├── data/
│   ├── normal/          # original noisy ultrasound images
│   ├── denoised_gt/     # ground-truth clean images (if available)
│   ├── dncnn_out/       # DnCNN outputs (will be generated)
│
├── snapshot/
│   └── save_DnCNN_.../  # your existing DnCNN .h5 models
│
├── esrgan_model/        # ESRGAN weights (.pt or .pth)
├── scripts/
│   ├── denoise_images.py
│   ├── train_esrgan.py
│   └── hybrid_inference.py
```

If your **ground truth** is the DnCNN output itself (no extra clean reference), then:

- `normal/` = input
- `dncnn_out/` = “cleaner” reference  
- You can still train ESRGAN to go *one step further* on top of DnCNN outputs.

If you have separate **true clean** images, put them in `denoised_gt/` and use those as ESRGAN targets.

---

## 3. Step 1 — Generate the DnCNN Output Dataset (`dncnn_out/`)

You already have `denoise_images.py`. To generate a consistent DnCNN dataset for ESRGAN training:

### 3.1. Create `dncnn_out/` directory

```bash
mkdir -p ./data/dncnn_out
```

### 3.2. Run DnCNN over all images

Use your existing script but point the output to `dncnn_out` and **save only the denoised image**:

```bash
python denoise_images.py     --input ./data/normal     --output ./data/dncnn_out     --model ./snapshot/save_DnCNN_sigma10_2025-08-10-03-42-53/model_100.h5     --save_mode denoised
```

Result:

- For each `./data/normal/img001.png`, you get `./data/dncnn_out/img001_denoised.png` (or similar name).
- These files are grayscale `uint8` images, denoised by DnCNN.

> **Important:** Use **consistent naming** so ESRGAN can match `dncnn_out` images to their targets (either `normal` or `denoised_gt`). If you prefer, you can rename outputs so they have exactly the same filename (e.g. `img001.png`) before training ESRGAN.

---

## 4. Step 2 — Define ESRGAN Training Pairs

Now we decide what ESRGAN learns to do. There are two sensible options:

### Option A — ESRGAN Refines DnCNN Towards a True Clean Reference

Use this if you have **true clean images** (for example, manually curated “best-quality” frames).

- **Input (LR):** `dncnn_out/`  (DnCNN denoised images)
- **Target (HR):** `denoised_gt/` (true clean images)

This tells ESRGAN:
> “You will receive DnCNN outputs at test time, and you must refine them to match the real clean image.”

### Option B — ESRGAN Refines DnCNN Towards Original (Mild SR / contrast)

Use this if you **do not** have separate clean targets and want ESRGAN mainly as a contrast/texture enhancer.

- **Input (LR):** `dncnn_out/`
- **Target (HR):** `normal/` (or slightly preprocessed)

This tells ESRGAN:
> “You receive denoised images and learn to reintroduce structure/contrast similar to the original.

For a conservative medical setup, **Option A** is preferred when possible.

In both cases, training pairs are defined by filename (e.g. `img001_denoised.png` ↔ `img001.png`).

---

## 5. Step 3 — Train ESRGAN on DnCNN Outputs

### 5.1. ESRGAN training configuration (recommended for ultrasound)

- **Scale:** `1` (no super-resolution, only enhancement)
- **Input channels:** `1` (grayscale)
- **nf (num features):** `32`
- **nb (num residual blocks):** `8–12`
- **Losses:**
  - L1 loss (pixel-wise)
  - Perceptual (VGG-based) loss (low weight)
  - SSIM loss (stronger weight for structural similarity)
- **GAN strategy:**
  - Pre-train generator with (L1 + SSIM + perceptual) for 20–30 epochs.
  - Optionally add adversarial loss later; keep it very small to avoid hallucinations.

### 5.2. Dataset paths for ESRGAN

In your `esrgan.py` (or future `train_esrgan.py`), set:

```python
args.in_dir  = "./data/dncnn_out"      # input to ESRGAN (from DnCNN)
args.gt_dir  = "./data/denoised_gt"   # or "./data/normal" depending on chosen option
args.out_dir = "./esrgan_model"
args.scale   = 1
args.nf      = 32
args.nb      = 8
```

After training, you should obtain something like:

```text
./esrgan_model/esrgan_final.pth
```

This is the ESRGAN generator used in the hybrid system.

---

## 6. Step 4 — Build the Hybrid Inference Pipeline

Once DnCNN and ESRGAN are both trained, the hybrid system inference looks like this:

```text
Original image (noisy)
   │
   ▼
DnCNN (Keras .h5 model)
   │          (denoised, [0,1])
   ▼
ESRGAN (PyTorch .pth model)
   │
   ▼
Hybrid enhanced image
```

### 6.1. Loading DnCNN (Keras)

```python
from keras.models import load_model
import numpy as np
from PIL import Image

dncnn = load_model("./snapshot/save_DnCNN_sigma10_2025-08-10-03-42-53/model_100.h5", compile=False)

def run_dncnn(img_path):
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape

    # same padding as denoise_images.py
    pad_h = (4 - (h % 4)) % 4
    pad_w = (4 - (w % 4)) % 4
    arr_pad = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')

    x = arr_pad.reshape(1, arr_pad.shape[0], arr_pad.shape[1], 1)
    y = dncnn.predict(x, verbose=0)
    out_full = y.reshape(arr_pad.shape)
    out = out_full[:h, :w]
    out = np.clip(out, 0.0, 1.0)
    return out    # float32 [0,1]
```

### 6.2. Loading ESRGAN (PyTorch)

```python
import torch

from esrgan import ESRGAN_G  # your ESRGAN generator class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esrgan_G = ESRGAN_G(scale=1, nf=32, nb=8, in_nc=1, out_nc=1)
esrgan_G.load_state_dict(torch.load("./esrgan_model/esrgan_final.pth", map_location=device))
esrgan_G.to(device)
esrgan_G.eval()
```

### 6.3. ESRGAN refinement function

Assuming ESRGAN is trained on **inputs normalized to [-1,1]**:

```python
@torch.no_grad()
def run_esrgan(dncnn_out_01: np.ndarray) -> np.ndarray:
    # dncnn_out_01: numpy array in [0,1], shape (H,W)
    x = dncnn_out_01.astype(np.float32)
    x = (x - 0.5) / 0.5          # [0,1] -> [-1,1]

    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    y = esrgan_G(t)
    y = y.squeeze(0).squeeze(0).cpu().numpy()  # [-1,1]

    y = (y + 1.0) / 2.0          # [-1,1] -> [0,1]
    y = np.clip(y, 0.0, 1.0)
    return y
```

### 6.4. Full hybrid call

```python
def hybrid_enhance(img_path: str) -> np.ndarray:
    x_dn = run_dncnn(img_path)       # [0,1]
    x_hybrid = run_esrgan(x_dn)      # [0,1]
    return x_dn, x_hybrid
```

The returned arrays can be converted to `uint8` for saving or displayed directly in Streamlit.

---

## 7. Step 5 — Evaluation: DnCNN vs ESRGAN vs Hybrid

For each available test image (and its target):

1. **Compute DnCNN only:**
   - `dncnn_out`

2. **Compute Hybrid:**
   - `hybrid_out = ESRGAN(dncnn_out)`

3. **Metrics vs ground truth** (`denoised_gt` or best-available image):
   - PSNR
   - SSIM

4. **Visual comparison:**
   - Original
   - DnCNN
   - Hybrid

A simple summary table:

```text
Image   | PSNR(DnCNN) | SSIM(DnCNN) | PSNR(Hybrid) | SSIM(Hybrid)
--------|-------------|-------------|--------------|-------------
img001  |      ..     |     ..      |      ..      |     ..
img002  |      ..     |     ..      |      ..      |     ..
...
```

If the hybrid system is working as intended, you should see **equal or better** PSNR/SSIM and clearer boundaries in the hybrid output vs DnCNN alone, without hallucinating false structures.

---

## 8. Summary

1. **DnCNN is already trained and used via `denoise_images.py`.**
2. Use that same script to generate a consistent dataset: `data/dncnn_out/`.
3. Train **ESRGAN using `dncnn_out` as input** and your chosen target images.
4. Build a hybrid inference pipeline that does:  
   `original → DnCNN (.h5) → ESRGAN (.pth) → enhanced image`.
5. Evaluate with PSNR/SSIM and side‑by‑side visualization.

These steps give you a **proper, publishable hybrid DnCNN + ESRGAN system** based exactly on your current code and model formats.
