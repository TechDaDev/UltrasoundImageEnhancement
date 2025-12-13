# ESRGAN Black Output – Fix Instructions (Hybrid DnCNN + ESRGAN)

This document describes **precise code changes** needed to fix the issue where the **ESRGAN output is pure black** in the hybrid DnCNN + ESRGAN ultrasound enhancement pipeline.

The root cause is a **mismatch between training and inference** for ESRGAN:

1. **Training** uses images normalized to **[-1, 1]** and a **`tanh` output** in the generator.
2. **Inference** currently feeds **[0, 1]** tensors to ESRGAN and disables `tanh`, then clips the raw logits to `[0,1]` → the result collapses to black.

Below are step–by–step fixes you can apply to the existing codebase.

---

## 1. Background – How ESRGAN Is Trained

In `esrgan.py` (training side), the dataset applies:

```python
transforms.ToTensor()              # converts image to [0,1]
transforms.Normalize([0.5],[0.5])  # maps to [-1,1]
```

So **both input and target are in [-1,1] during training**.

The ESRGAN generator is created with `use_tanh=True` during the GAN phase, so it learns to output values in the **[-1,1]** range as well.

---

## 2. Problem – How ESRGAN Is Used at Inference

In `hybrid_training.py`, inside your `HybridDenoiseEnhancer` class:

1. **Generator instantiation**:

```python
self.esrgan_G = ESRGAN_G(
    scale=1,
    nf=32,
    nb=12,
    use_tanh=False   # <-- PROBLEM
).to(self.torch_device)
```

2. **Inference**:

```python
t = torch.from_numpy(img_np).float().to(self.torch_device)
# no normalization here; img_np is in [0,1]
out_es = self.esrgan_G(t)
out_es = out_es.squeeze(0).cpu().numpy()
out_es = np.clip(out_es, 0.0, 1.0)  # logits are mostly negative, so everything becomes 0
```

So at test time we feed `[0,1]` instead of `[-1,1]`, and we remove the `tanh` that was used during training. The generator outputs values that don't match the distribution it learned, and after clipping they become almost entirely zeros → the black panel in your Streamlit app.

---

## 3. Fix 1 – Use `use_tanh=True` for ESRGAN at Inference

### File: `hybrid_training.py`  
### Class: `HybridDenoiseEnhancer.__init__`

**Change this:**

```python
self.esrgan_G = ESRGAN_G(
    scale=1,
    nf=32,
    nb=12,
    use_tanh=False
).to(self.torch_device)
```

**To this:**

```python
self.esrgan_G = ESRGAN_G(
    scale=1,
    nf=32,
    nb=12,
    use_tanh=True  # MUST match training configuration
).to(self.torch_device)
```

This re–enables the `tanh` activation on the ESRGAN output, so it produces values in **[-1,1]**, which is what the training loss expects.

---

## 4. Fix 2 – Match Input Normalization at Inference

During training, ESRGAN sees images normalized as:

```python
x_norm = (x - 0.5) / 0.5  # [0,1] -> [-1,1]
```

We must apply the **same transformation** to the DnCNN output before passing it to ESRGAN.

### File: `hybrid_training.py`  
### Method: `HybridDenoiseEnhancer.enhance` (ESRGAN part)

Instead of sending `img_np` directly to the model, **normalize to [-1,1]** and afterwards convert back to `[0,1]`.

Below is a **drop–in replacement** for the ESRGAN inference method.

#### New ESRGAN Inference Method

```python
@torch.no_grad()
def enhance_with_esrgan(self, img_np: np.ndarray) -> np.ndarray:
    """
    ESRGAN refinement step.

    Parameters
    ----------
    img_np : np.ndarray
        DnCNN output in [0,1], shape (H, W) or (H, W, 1).

    Returns
    -------
    np.ndarray
        ESRGAN–enhanced image in [0,1], shape (H, W).
    """
    # Ensure float32
    x = img_np.astype(np.float32)

    # 1) Normalize [0,1] -> [-1,1] to match training
    x = (x - 0.5) / 0.5

    # 2) Convert to Torch tensor [1,1,H,W]
    t = torch.from_numpy(x).to(self.torch_device)
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    elif t.ndim == 3:
        # assume H,W,C with C=1
        t = t.permute(2, 0, 1).unsqueeze(0)      # [1,C,H,W]

    # 3) Forward through ESRGAN (output in [-1,1] because of tanh)
    out = self.esrgan_G(t)

    # 4) Back to numpy [H,W] in [0,1]
    out = out.squeeze(0).cpu().numpy()          # [C,H,W] or [1,H,W]
    if out.shape[0] == 1:
        out = out[0]                            # [H,W] grayscale
    else:
        out = out.transpose(1, 2, 0)            # [H,W,C], for completeness

    # [-1,1] -> [0,1]
    out = (out + 1.0) / 2.0
    out = np.clip(out, 0.0, 1.0)
    return out
```

Then in the rest of your code, replace any direct ESRGAN calls on `img_np` with:

```python
esrgan_img = self.enhance_with_esrgan(dncnn_output)
```

Where `dncnn_output` is the DnCNN result in `[0,1]` (as you already have).

---

## 5. Optional – Wrap the Whole Hybrid Step

If you want a single method that does **DnCNN + ESRGAN** together (useful for your Streamlit app), you can add:

```python
def enhance_full(self, img_np: np.ndarray) -> np.ndarray:
    """
    Full hybrid enhancement: DnCNN -> ESRGAN.

    img_np is the original noisy ultrasound image in [0,1].
    """
    # 1) DnCNN denoising
    inp_dn = img_np.astype(np.float32)[None, ..., None]  # [1,H,W,1]
    out_dn = self.dncnn.predict(inp_dn, verbose=0)[0, ..., 0]
    out_dn = np.clip(out_dn, 0.0, 1.0)

    # 2) ESRGAN refinement
    out_es = self.enhance_with_esrgan(out_dn)
    return out_es
```

Your front–end can then call `enhancer.enhance_full(original_img_normalized)` and display:

- Original image  
- DnCNN-only (`out_dn`)  
- ESRGAN-enhanced (`out_es`)

---

## 6. Sanity Checks After Applying the Fixes

After implementing the changes:

1. **DnCNN output** should look the same as before.  
2. **ESRGAN output** should no longer be black.  
   - It should resemble the DnCNN output with slightly different contrast/texture.  
3. For debugging, you can print value ranges:

```python
print("DnCNN range", out_dn.min(), out_dn.max())
print("ESRGAN range", out_es.min(), out_es.max())
```

4. If you still get near–black outputs, confirm that:
   - `use_tanh=True` when constructing ESRGAN at inference.
   - You are loading the *correct* weights (e.g. `esrgan_final.pt` from the final training checkpoint).
   - The training code indeed used normalized inputs in `[-1,1]` (as in this project).

---

## 7. Summary of Required Code Changes

1. **Enable tanh in ESRGAN generator at inference**  
   - Change `use_tanh=False` → `use_tanh=True` in `HybridDenoiseEnhancer.__init__`.

2. **Normalize input to ESRGAN exactly as in training**  
   - Before feeding ESRGAN: `(img - 0.5) / 0.5` → tensor in `[-1,1]`.  
   - After ESRGAN (with tanh): `(out + 1) / 2` → image in `[0,1]`.

3. **Use the new `enhance_with_esrgan` (and optional `enhance_full`) methods**  
   - Ensures any UI (Streamlit, etc.) sees consistent, correctly scaled ESRGAN results.
