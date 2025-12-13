"""Hybrid DnCNN + ESRGAN Image Enhancement Pipeline.

Refactored based on `hybrid_dncnn_esrgan_plan.md`.
Implements a strict two-stage pipeline:
1. DnCNN (Supervised Denoising)
2. ESRGAN (Refinement/Super-resolution)

Uses paired training data (Noisy -> Clean).
"""

import os
import time
import math
import shutil
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional

# Keras / TensorFlow
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model

# PyTorch
import torch

# Local modules
from models import DnCNN
import esrgan
from gpu_utils import configure_gpu, force_cpu


# -----------------------------------------------------------------------------
# 1. Dataset & Utils
# -----------------------------------------------------------------------------

def list_paired_files(noisy_dir: Path, clean_dir: Path, exts={".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
    """List paired image files from noisy and clean directories."""
    noisy_files = []
    clean_files = []

    if not noisy_dir.exists():
        print(f"⚠️ Noisy directory not found: {noisy_dir}")
        return [], []
    
    if not clean_dir.exists():
        print(f"⚠️ Clean directory not found: {clean_dir}")
        return [], []

    # Get all clean files first (targets)
    all_clean = {p.name: p for p in clean_dir.iterdir() if p.suffix.lower() in exts}

    # Match noisy files
    for p in sorted(noisy_dir.iterdir()):
        if p.name in all_clean and p.suffix.lower() in exts:
            noisy_files.append(p)
            clean_files.append(all_clean[p.name])
    
    return noisy_files, clean_files


def paired_sequence(noisy_files, clean_files, batch_size=4, target_size=(128, 128), shuffle=True):
    """Generator yielding (x_batch, y_batch) for DnCNN training."""
    if len(noisy_files) == 0:
        return
        
    idx = np.arange(len(noisy_files))
    
    while True:
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            
            # Handle last batch padding to ensure fixed size if needed
            if len(batch_idx) < batch_size:
                pad = batch_size - len(batch_idx)
                batch_idx = np.concatenate([batch_idx, idx[:pad]])

            xs, ys = [], []
            for i in batch_idx:
                try:
                    # Load as grayscale for DnCNN
                    noisy = Image.open(noisy_files[i]).convert("L")
                    clean = Image.open(clean_files[i]).convert("L")

                    if target_size is not None:
                        # Resize to patch size
                        noisy = noisy.resize(target_size, Image.BICUBIC)
                        clean = clean.resize(target_size, Image.BICUBIC)

                    n_arr = np.asarray(noisy, dtype=np.float32) / 255.0
                    c_arr = np.asarray(clean, dtype=np.float32) / 255.0

                    xs.append(n_arr[..., None])  # [H, W, 1]
                    ys.append(c_arr[..., None])  # [H, W, 1]
                except Exception as e:
                    print(f"Error loading pair {noisy_files[i]}: {e}")
                    # Skip or fill with zeros? Filling with zeros to match batch
                    xs.append(np.zeros((target_size[0], target_size[1], 1), dtype=np.float32))
                    ys.append(np.zeros((target_size[0], target_size[1], 1), dtype=np.float32))

            x_batch = np.stack(xs, axis=0)
            y_batch = np.stack(ys, axis=0)
            yield x_batch, y_batch


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to image. img in [0,1], sigma in [0,255]."""
    sig = sigma / 255.0
    n = np.random.normal(0.0, sig, img.shape).astype(np.float32)
    return np.clip(img + n, 0.0, 1.0)


def save_image(arr: np.ndarray, path: Path):
    """Save float [0,1] image (H,W or H,W,C)."""
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    img.save(path)


# -----------------------------------------------------------------------------
# 2. Hybrid Inference Class
# -----------------------------------------------------------------------------

class HybridDenoiseEnhancer:
    """Uses DnCNN (TF) + ESRGAN (Torch) for inference."""
    
    def __init__(self, dncnn_weights: Path, esrgan_weights: Path, force_cpu_flag=False, verbose=True):
        self.verbose = verbose
        self.torch_device = torch.device("cpu")
        
        # 1. Setup TF / GPU
        if force_cpu_flag:
            force_cpu(quiet=not verbose)
        else:
            configure_gpu(memory_growth=True, quiet=not verbose)
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")

        # 2. Load DnCNN
        if verbose: print(f"Loading DnCNN: {dncnn_weights}")
        if str(dncnn_weights).endswith(".keras"):
            self.dncnn = load_model(str(dncnn_weights), compile=False)
        else:
            # Fallback if just weights
            self.dncnn = DnCNN()
            self.dncnn.load_weights(str(dncnn_weights))

        # 3. Load ESRGAN
        if verbose: print(f"Loading ESRGAN: {esrgan_weights}")
        # Note: Plan says nf=32, nb=12. Ensure these match training!
        from esrgan import ESRGAN_G
        # We try to infer params from checkpoint if standard esrgan.py storage, 
        # but defaulting to plan specs: 32/8/16. If failed, user might need to adjust.
        self.esrgan_G = ESRGAN_G(
            scale=1, nf=32, nb=8, gc=16, grad_ckpt=False, use_tanh=False
        ).to(self.torch_device)  # MUST match training configuration
        
        try:
            ckpt = torch.load(esrgan_weights, map_location=self.torch_device)
            # Handle different checkpoint formats
            if "G" in ckpt:
                state = ckpt["G"]
            else:
                state = ckpt
            self.esrgan_G.load_state_dict(state, strict=False)
        except Exception as e:
            if verbose: print(f"Warning: Standard load failed ({e}), trying full dict...")
            self.esrgan_G.load_state_dict(torch.load(esrgan_weights, map_location=self.torch_device))
        
        self.esrgan_G.eval()

    def denoise(self, img_np: np.ndarray) -> np.ndarray:
        """DnCNN inference. Input: [H,W] or [H,W,1] in [0,1]. Output: Same."""
        if img_np.ndim == 2:
            img_np = img_np[..., None]
        
        x_in = img_np[None, ...].astype(np.float32) # [1, H, W, 1]
        out_dn = self.dncnn.predict(x_in, verbose=0)[0]
        return np.clip(out_dn.squeeze(), 0.0, 1.0)

    @torch.no_grad()
    def enhance(self, img_np: np.ndarray) -> np.ndarray:
        """ESRGAN inference with proper normalization (Sigmoid).
        
        Input: [H,W] in [0,1] (from denoise). Output: [H,W] in [0,1].
        """
        # Ensure float32
        x = img_np.astype(np.float32)
        
        # Ensure HxW
        if x.ndim == 3:
            x = x[..., 0]
            
        # To tensor [1,1,H,W]
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.torch_device)
        
        # Forward through ESRGAN (unconstrained output)
        y = self.esrgan_G(t)
        
        # Squash with Sigmoid to [0,1]
        y = torch.sigmoid(y)
        
        # Back to numpy [H,W] in [0,1]
        y = y.squeeze(0).squeeze(0).cpu().numpy()
        y = np.clip(y, 0.0, 1.0)
        return y


# -----------------------------------------------------------------------------
# 3. Training Stages
# -----------------------------------------------------------------------------

def stage1_train_dncnn(
    noisy_dir: Path, 
    clean_dir: Path, 
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 4
):
    """Stage 1: Train DnCNN using paired images."""
    print("\n" + "="*50)
    print("STAGE 1: Training DnCNN")
    print("="*50)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "dncnn_final.keras"
    if model_path.exists():
        print(f"✅ Found existing DnCNN model at {model_path}, skipping training.")
        return model_path

    # Prepare data
    noisy_files, clean_files = list_paired_files(noisy_dir, clean_dir)
    print(f"Found {len(noisy_files)} paired images for DnCNN training.")
    
    if len(noisy_files) == 0:
        raise ValueError("No paired training data found for DnCNN.")

    dncnn = DnCNN()
    dncnn.compile(optimizer=Adam(1e-3), loss="mae")
    
    steps = math.ceil(len(noisy_files) / batch_size)
    gen = paired_sequence(noisy_files, clean_files, batch_size=batch_size)

    print("Starting training...")
    dncnn.fit(
        gen,
        steps_per_epoch=steps,
        epochs=epochs,
        verbose=1
    )
    
    dncnn.save(model_path)
    print(f"✅ DnCNN saved to {model_path}")
    return model_path


def stage2_generate_intermediates(
    dncnn_model_path: Path,
    noisy_dir: Path,
    intermediate_dir: Path
):
    """Stage 2: Generate Denoised images to feed into ESRGAN."""
    print("\n" + "="*50)
    print("STAGE 2: Generating DnCNN Outputs")
    print("="*50)
    
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already generated
    files_in_out = list(intermediate_dir.glob("*.png")) + list(intermediate_dir.glob("*.jpg"))
    if len(files_in_out) > 0:
         print(f"✅ Found {len(files_in_out)} images in {intermediate_dir}, skipping generation.")
         return

    print("Loading DnCNN...")
    model = load_model(str(dncnn_model_path), compile=False)
    
    images = sorted([p for p in noisy_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    print(f"Processing {len(images)} images...")
    
    for i, p in enumerate(images):
        # Process as Grayscale
        img = Image.open(p).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        
        # Predict
        inp = arr[None, ..., None]
        out = model.predict(inp, verbose=0)[0, ..., 0]
        
        save_image(out, intermediate_dir / p.name)
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)}")
            
    print(f"✅ Generated output images in {intermediate_dir}")


def stage3_train_esrgan(
    input_dir: Path,  # dncnn outputs
    clean_dir: Path,  # Ground Truth
    output_dir: Path,
    epochs: int = 40,
    batch_size: int = 4
):
    """Stage 3: Train ESRGAN on DnCNN output -> Clean GT."""
    print("\n" + "="*50)
    print("STAGE 3: Training ESRGAN Refiner")
    print("="*50)
    
    final_model = output_dir / "ckpts" / "esrgan_final.pt"
    if final_model.exists():
        print(f"✅ Found existing ESRGAN model at {final_model}, skipping training.")
        return final_model

    # Configure args for esrgan.train
    # The plan suggests nf=32, nb=12, scale=1, crop=128
    # We split epochs into pretrain and gan for smoother convergence
    pretrain = max(5, epochs // 2)
    gan_epochs = epochs - pretrain
    
    args = argparse.Namespace(
        noisy_dir=str(input_dir),
        clean_dir=str(clean_dir),
        test_dir=None,
        out_dir=str(output_dir),
        scale=1,
        crop=128,
        rand_crop=True, # Random crop for better generalization
        batch=batch_size,
        epochs=gan_epochs,
        pretrain_epochs=pretrain,
        
        # Model Specs (Lighter model)
        nf=32,
        nb=12,
        use_tanh=False, 
        no_tanh_pretrain=True, # Important to avoid saturation
        
        # Losses / Opt
        lr=1e-4,
        lambda_l1=1.0,
        lambda_perc=1.0,
        lambda_gan=0.005,
        lambda_ssim_pre=0.5,
        
        # System
        cpu=not torch.cuda.is_available(),
        amp=torch.cuda.is_available(),
        skip_tf=True, # We already used TF in stage 1, cleaner to skip here
        grad_ckpt=False,
        accum_steps=1,
        auto_batch=True,
        
        # Defaults for validation logic
        save_every=5,
        sample_every=500,
        d_lr=None,
        spectral_norm=False,
        var_guard_thresh=0.01,
        var_guard_patience=2,
        pre_collapse_var=0.0005,
        pre_collapse_patience=2,
        warmup_steps=0,
        smooth_real=1.0,
        fail_on_warn=False,
        dry_run=False,
        quick_diag=False,
        num_workers=0
    )

    args = esrgan.validate_args(args)
    esrgan.train(args)
    return final_model


# -----------------------------------------------------------------------------
# 4. Main Config & Execution
# -----------------------------------------------------------------------------

def main():
    print("🚀 Initializing Hybrid Training Pipeline...")
    
    # Paths (relative to project root)
    DATA_ROOT = Path("dataset")
    # If using your provided structure
    if not DATA_ROOT.exists():
        # Fallback to current dir 'dataset' or extracting
        zip_path = Path("dataset.zip")
        if zip_path.exists():
            print("Extracting dataset.zip...")
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(".")
            DATA_ROOT = Path("dataset")
        
    # Auto-detect Noisy/Clean from folders (similar to old script logic but stricter)
    # We expect 'normal' and 'denoised' (or clean/noisy)
    if DATA_ROOT.exists():
        subs = list(DATA_ROOT.iterdir())
        # Heuristic search
        clean_dir = None
        noisy_dir = None
        
        for p in subs:
            if not p.is_dir(): continue
            name = p.name.lower()
            if any(x in name for x in ["clean", "denoise", "gt"]):
                clean_dir = p
            elif any(x in name for x in ["noisy", "normal", "input"]):
                noisy_dir = p
                
        # Fallback synthesis if only clean exists
        if clean_dir and not noisy_dir:
            print("⚠️ Only Clean directory found. Generating synthetic Noisy data...")
            noisy_dir = DATA_ROOT / "synthetic_noisy"
            noisy_dir.mkdir(exist_ok=True)
            for f in clean_dir.glob("*"):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    img = Image.open(f).convert("L")
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    noisy = add_gaussian_noise(arr, sigma=15.0) # Sigma 15
                    save_image(noisy, noisy_dir / f.name)
            print(f"✅ Generated {len(list(noisy_dir.glob('*')))} synthetic noisy images.")
            
    else:
        # Defaults if autodetection fails (user must ensure these exist)
        clean_dir = Path("data/denoised")
        noisy_dir = Path("data/normal")

    if not clean_dir or not clean_dir.exists():
        print(f"❌ Error: Clean data directory not found. Please ensure dataset is present.")
        return

    print(f"📂 Configuration:")
    print(f"  Noisy Input: {noisy_dir}")
    print(f"  Clean Target: {clean_dir}")
    
    # Output Directories
    RUN_ROOT = Path("runs_hybrid_paired")
    DNCNN_OUT = RUN_ROOT / "dncnn_model"
    INTERMEDIATE = RUN_ROOT / "dncnn_inferred_trainset"
    ESRGAN_OUT = RUN_ROOT / "esrgan_model"
    
    # --- EXECUTE PIPELINE ---
    
    # 1. Train DnCNN
    dncnn_model = stage1_train_dncnn(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        output_dir=DNCNN_OUT,
        epochs=30,
        batch_size=4
    )
    
    # 2. Generate Intermediates
    stage2_generate_intermediates(
        dncnn_model_path=dncnn_model,
        noisy_dir=noisy_dir,
        intermediate_dir=INTERMEDIATE
    )
    
    # 3. Train ESRGAN (Refinement)
    esrgan_model = stage3_train_esrgan(
        input_dir=INTERMEDIATE,
        clean_dir=clean_dir,
        output_dir=ESRGAN_OUT,
        epochs=40, # 20 pre + 20 gan roughly
        batch_size=4
    )
    
    print("\n✅ Hybrid Training Complete!")
    print(f"DnCNN: {dncnn_model}")
    print(f"ESRGAN: {esrgan_model}")
    
    # Save a handy inference test script or usage example?
    # The App uses this file, so updating the HybridDenoiseEnhancer class in-place (above) is key.


if __name__ == "__main__":
    main()
