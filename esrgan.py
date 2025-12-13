# esrgan_phase1.py
import os, argparse
from pathlib import Path
import numpy as np
from packaging import version
from PIL import Image

# --- NumPy / PyTorch compatibility probe ---
# Earlier version of this script aborted unconditionally for NumPy>=2. Now we only warn if an
# actual interoperability check fails (e.g., ABI mismatch creating a tensor from a NumPy array).
_np_ver = version.parse(np.__version__)
try:
    import torch as _torch_probe
    _torch_ver = _torch_probe.__version__
    if _np_ver >= version.parse("2.0.0"):
        try:
            _a = np.array([1,2,3], dtype=np.float32)
            _t = _torch_probe.from_numpy(_a)  # zero-copy view
            _ = _t.numpy()  # round-trip
        except Exception as _compat_err:
            print(
                f"[WARN] Torch {_torch_ver} <-> NumPy {np.__version__} interop test failed: {_compat_err}\n"
                "        If you hit runtime errors, downgrade NumPy: pip install 'numpy<2' --force-reinstall"
            )
        else:
            # Optional concise note to acknowledge modern combo.
            pass
except Exception as _import_err:
    # If torch import itself fails here we'll hit it again later; keep silent to not mask real traceback.
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils as vutils, models
from gpu_utils import configure_gpu, force_cpu
import torch.utils.checkpoint as checkpoint
from torch import amp  # modern AMP API (replaces torch.cuda.amp)

# -------------------------
# Utils & Metrics
# -------------------------
def to_uint8_grid(t):
    # t: tensor in [-1,1]
    t = (t.clamp(-1,1) + 1) * 0.5
    return (t * 255.0).round().byte()

@torch.no_grad()
def psnr_torch(x, y):   # x,y in [-1,1]
    x = (x+1)/2; y = (y+1)/2
    mse = F.mse_loss(x, y, reduction="mean")
    if mse.item() == 0: return torch.tensor(99.0, device=x.device)
    return 20 * torch.log10(torch.tensor(1.0, device=x.device) / torch.sqrt(mse))

def _gauss_kernel(ch, k, sigma=1.5, device="cpu"):
    ax = torch.arange(k, device=device) - k//2
    g = torch.exp(-(ax**2)/(2*sigma*sigma))
    ker = (g[:,None] * g[None,:]); ker /= ker.sum()
    return ker.expand(ch,1,k,k)

@torch.no_grad()
def ssim_torch(x, y, ksize=11, C1=0.01**2, C2=0.03**2):
    # x,y in [-1,1], CHW batch
    # compute in float32 for numeric stability & dtype alignment under AMP
    dtype = torch.float32
    x = (x+1)/2; y = (y+1)/2
    if x.dtype != dtype: x = x.float()
    if y.dtype != dtype: y = y.float()
    ch = x.size(1)
    k = _gauss_kernel(ch, ksize, device=x.device).to(dtype)
    mu_x = F.conv2d(x, k, padding=ksize//2, groups=ch)
    mu_y = F.conv2d(y, k, padding=ksize//2, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x*mu_y
    sx = F.conv2d(x*x, k, padding=ksize//2, groups=ch) - mu_x2
    sy = F.conv2d(y*y, k, padding=ksize//2, groups=ch) - mu_y2
    sxy= F.conv2d(x*y, k, padding=ksize//2, groups=ch) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sxy + C2))/((mu_x2 + mu_y2 + C1)*(sx + sy + C2))
    return ssim_map.mean()

def trophy(psnr_val, ssim_val):  # composite score for quick ranking
    return (psnr_val/50.0) + ssim_val

# -------------------------
# Dataset (paired)
# -------------------------
class PairedUS(Dataset):
    """
    Pairs images by filename: noisy/<name> with clean/<name>.
    - If sizes don't match:
      - For scale==1: HR is resized to LR's size (conservative for denoising-only).
      - For scale>1: LR is resized from HR / scale by bicubic for consistency.
    """
    def __init__(self, noisy_dir, clean_dir, scale=1, crop=None, augment=False, rand_crop=False):
        self.noisy_dir, self.clean_dir = Path(noisy_dir), Path(clean_dir)
        self.scale, self.crop, self.augment = scale, crop, augment
        self.rand_crop = rand_crop

        noisy_files = sorted([p for p in self.noisy_dir.iterdir()
                              if p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")])
        clean_set = {p.name for p in Path(self.clean_dir).iterdir()}
        self.pairs = [(p, self.clean_dir/p.name) for p in noisy_files if p.name in clean_set]

        self.to_tensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),              # [0,1]
            transforms.Normalize([0.5],[0.5])   # [-1,1]
        ])
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.vflip = transforms.RandomVerticalFlip(p=0.5)

    def __len__(self): return len(self.pairs)

    def _center_crop(self, img, size):
        w,h = img.size
        th, tw = size, size
        x1 = max(0, (w - tw)//2); y1 = max(0, (h - th)//2)
        return img.crop((x1,y1,x1+tw,y1+th))

    def _random_crop(self, img, size):
        w,h = img.size
        th, tw = size, size
        if w == tw and h == th:
            return img
        if w < tw or h < th:
            # fallback to center crop if smaller
            return self._center_crop(img, size)
        x1 = np.random.randint(0, w - tw + 1)
        y1 = np.random.randint(0, h - th + 1)
        return img.crop((x1,y1,x1+tw,y1+th))

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        lr = Image.open(noisy_path).convert("L")
        hr = Image.open(clean_path).convert("L")

        if self.crop is not None:
            # For denoising (scale==1), crop BOTH LR and HR to fixed size
            # For super-resolution (scale>1), crop HR then derive LR from HR/scale
            if self.scale > 1:
                # crop HR first (random or center)
                if self.rand_crop:
                    hr = self._random_crop(hr, self.crop)
                else:
                    hr = self._center_crop(hr, self.crop)
                # enforce LR = HR / scale
                wh = (hr.size[0]//self.scale, hr.size[1]//self.scale)
                lr = hr.resize(wh, Image.BICUBIC)
            else:
                # scale == 1: crop LR and HR consistently to fixed patch size
                if self.rand_crop:
                    lr = self._random_crop(lr, self.crop)
                    hr = self._random_crop(hr, self.crop)
                else:
                    lr = self._center_crop(lr, self.crop)
                    hr = self._center_crop(hr, self.crop)
                # If minor size mismatch, resize HR to LR
                if lr.size != hr.size:
                    hr = hr.resize(lr.size, Image.BICUBIC)
        else:
            # no cropping; enforce size consistency
            if self.scale > 1:
                wh = (hr.size[0]//self.scale, hr.size[1]//self.scale)
                if lr.size != wh:
                    lr = hr.resize(wh, Image.BICUBIC)
            else:
                if lr.size != hr.size:
                    hr = hr.resize(lr.size, Image.BICUBIC)

        if self.augment:
            if np.random.rand() < 0.5: lr = self.hflip(lr); hr = self.hflip(hr)
            if np.random.rand() < 0.5: lr = self.vflip(lr); hr = self.vflip(hr)

        return self.to_tensor(lr), self.to_tensor(hr), noisy_path.name

# -------------------------
# ESRGAN: RRDB Generator + Patch Discriminator
# -------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf,     gc, 3,1,1); self.l1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nf+gc,  gc, 3,1,1); self.l2 = nn.LeakyReLU(0.2, True)
        self.conv3 = nn.Conv2d(nf+2*gc,gc, 3,1,1); self.l3 = nn.LeakyReLU(0.2, True)
        self.conv4 = nn.Conv2d(nf+3*gc,gc, 3,1,1); self.l4 = nn.LeakyReLU(0.2, True)
        self.conv5 = nn.Conv2d(nf+4*gc,nf, 3,1,1)

    def forward(self, x):
        x1 = self.l1(self.conv1(x))
        x2 = self.l2(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.l3(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.l4(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x  # residual scaling

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf,gc)
        self.rdb2 = ResidualDenseBlock(nf,gc)
        self.rdb3 = ResidualDenseBlock(nf,gc)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x

class ESRGAN_G(nn.Module):
    def __init__(self, scale=1, nf=64, nb=23, gc=32, grad_ckpt=False, use_tanh=True):
        super().__init__()
        self.scale = scale
        self.grad_ckpt = grad_ckpt
        self.use_tanh = use_tanh
        self.conv_first = nn.Conv2d(1, nf, 3,1,1)
        self.blocks = nn.ModuleList([RRDB(nf,gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3,1,1)

        ups = []
        if scale > 1:
            n_up = int(np.log2(scale))
            for _ in range(n_up):
                ups += [nn.Conv2d(nf, nf*4, 3,1,1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
        self.ups = nn.Sequential(*ups) if ups else nn.Identity()

        self.hr_conv = nn.Conv2d(nf, nf, 3,1,1)
        self.conv_last = nn.Conv2d(nf, 1, 3,1,1)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk_in = fea
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                # Explicit use_reentrant=False (newer non-reentrant version is recommended for performance)
                trunk_in = checkpoint.checkpoint(blk, trunk_in, use_reentrant=False)
            else:
                trunk_in = blk(trunk_in)
        trunk = self.trunk_conv(trunk_in)
        fea = fea + trunk
        fea = self.ups(fea)
        out  = self.conv_last(F.leaky_relu(self.hr_conv(fea), 0.2, True))
        if self.use_tanh:
            return torch.tanh(out)  # [-1,1]
        return out  # raw (pre-activation) during pretrain if disabled

    def set_tanh(self, enabled: bool):
        self.use_tanh = enabled

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=2, spectral=False):
        super().__init__()
        def conv(ic, oc, k=3, s=2, p=1, bn=True):
            layer = nn.Conv2d(ic, oc, k, s, p)
            if spectral: layer = spectral_norm(layer)
            mods = [layer]
            if bn: mods.append(nn.BatchNorm2d(oc))
            mods.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*mods)
        first = nn.Conv2d(in_ch, 64, 3,1,1)
        if spectral: first = spectral_norm(first)
        tail = nn.Conv2d(256,1,3,1,1)
        if spectral: tail = spectral_norm(tail)
        self.net = nn.Sequential(
            first, nn.LeakyReLU(0.2, True),
            conv(64, 64), conv(64,128), conv(128,128),
            conv(128,256), conv(256,256),
            tail
        )

    def forward(self, cond, img):  # cond: LR upsampled to HR, img: HR/SR
        return self.net(torch.cat([cond, img], 1))

# -------------------------
# Perceptual (VGG) Loss
# -------------------------
class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice = nn.Sequential(*list(vgg.children())[:36]).eval()  # up to relu5_4
        for p in self.slice.parameters(): p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x):  # x in [-1,1], 1ch -> 3ch
        x3 = x.repeat(1,3,1,1)
        x3 = (x3+1)/2
        x3 = (x3 - self.mean)/self.std
        return self.slice(x3)

# -------------------------
# Training / Validation
# -------------------------
def save_triplet(lr, sr, hr, path, nrow=4):
    with torch.no_grad():
        Hh, Wh = hr.shape[-2:]
        lr_up = F.interpolate(lr, size=(Hh,Wh), mode="bicubic", align_corners=False)
        grid = torch.cat([lr_up, sr, hr], 0)  # rows: LR↑, SR(ESRGAN), HR
        grid = (grid.clamp(-1,1)+1)*0.5
        vutils.save_image(grid, path, nrow=nrow, padding=2)

def validate(G, loader, device, use_amp=False):
    G.eval()
    ps, ss, ts = [], [], []
    with torch.no_grad():
        for lr, hr, _ in loader:
            lr, hr = lr.to(device), hr.to(device)
            from torch import amp as _amp
            with _amp.autocast('cuda', enabled=use_amp):
                sr = G(lr)
            p = psnr_torch(sr, hr).item()
            s = ssim_torch(sr, hr).item()
            t = (p/50.0) + s
            ps.append(p); ss.append(s); ts.append(t)
    return float(np.mean(ps) if ps else 0), float(np.mean(ss) if ss else 0), float(np.mean(ts) if ts else 0)

def train(args):
    # Optionally configure TensorFlow (may consume VRAM); allow skipping to free memory.
    if not getattr(args, 'skip_tf', False):
        if args.cpu:
            _tf_dev = force_cpu(quiet=False)
        else:
            _tf_dev = configure_gpu(memory_growth=True, quiet=False)
    else:
        _tf_dev = 'SKIPPED'
        print('Skipped TensorFlow GPU configuration (--skip_tf)')

    use_cuda = torch.cuda.is_available() and (not args.cpu)
    use_amp = use_cuda and getattr(args, 'amp', False)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        print(f"PyTorch will use GPU: {name}")
    else:
        print("PyTorch will use CPU")
    scaler_g = amp.GradScaler('cuda', enabled=use_amp)
    scaler_d = amp.GradScaler('cuda', enabled=use_amp)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f"{args.out_dir}/ckpts", exist_ok=True)
    os.makedirs(f"{args.out_dir}/samples", exist_ok=True)

    # Data
    full = PairedUS(args.noisy_dir, args.clean_dir, scale=args.scale, crop=args.crop, augment=True, rand_crop=getattr(args,'rand_crop',False))
    if len(full) == 0:
        raise SystemExit(f"No paired images found between {args.noisy_dir} and {args.clean_dir}")
    n_val = max(1, int(0.1*len(full)))
    val_ds = torch.utils.data.Subset(full, range(0, n_val))
    trn_ds = torch.utils.data.Subset(full, range(n_val, len(full)))

    # Optional auto batch size finder (model memory only heuristic)
    if args.auto_batch:
        # Build a temporary model (smaller maybe same) to probe memory use.
        probe = ESRGAN_G(scale=args.scale, nf=args.nf, nb=args.nb, grad_ckpt=getattr(args,'grad_ckpt',False)).to(device)
        probe.eval()
        with torch.no_grad():
            lr0, hr0, _ = trn_ds[0]
            H, W = lr0.shape[-2:]
        test_bs = args.batch
        while test_bs > 0:
            try:
                torch.cuda.empty_cache()
                with amp.autocast('cuda', enabled=use_amp), torch.no_grad():
                    x = torch.randn(test_bs, 1, H, W, device=device)
                    _ = probe(x)
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    test_bs //= 2
                    continue
                else:
                    raise
        if test_bs < 1:
            raise SystemExit('Auto batch search failed to find a viable batch size >0')
        if test_bs != args.batch:
            print(f"[auto_batch] Reducing batch size from {args.batch} -> {test_bs}")
            args.batch = test_bs
        del probe; torch.cuda.empty_cache()

    # DataLoader tuning: allow overriding worker count and keep workers alive
    nw = getattr(args, 'num_workers', 2)
    train_loader = DataLoader(
        trn_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    # Optional test-only loader (if you drop clean refs there, metrics will compute; otherwise just samples)
    test_loader = None
    if args.test_dir and Path(args.test_dir).exists():
        test_loader = DataLoader(PairedUS(args.test_dir, args.clean_dir, scale=args.scale, crop=None, augment=False),
                                 batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    G = ESRGAN_G(scale=args.scale, nf=args.nf, nb=args.nb, grad_ckpt=getattr(args,'grad_ckpt',False), use_tanh=not getattr(args,'no_tanh_pretrain', False)).to(device)
    D = PatchDiscriminator(in_ch=2, spectral=getattr(args, 'spectral_norm', False)).to(device)
    perc = VGGPerceptual().to(device)

    # Opts & losses
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9,0.999))
    # Use separate discriminator lr only if provided; fallback to generator lr when None
    disc_lr = args.d_lr if getattr(args, 'd_lr', None) is not None else args.lr
    d_opt = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(0.9,0.999))
    base_g_lr = args.lr
    base_d_lr = disc_lr
    g_update_count = 0  # for warmup
    bce = nn.BCEWithLogitsLoss()
    l1  = nn.L1Loss()
    # Quick diagnostics tracking
    qd = {}
    if getattr(args, 'quick_diag', False):
        watch = []
        for n,_p in G.named_parameters():
            if any(k in n for k in ["conv_first","trunk_conv","conv_last"]):
                watch.append(n)
        qd['initial'] = {n: G.state_dict()[n].clone().detach() for n in watch}
        print(f"[QD] Tracking {len(watch)} generator parameter tensors: {watch}")

    # ---- Phase 1: Pretrain G (SRResNet) w/o GAN
    if args.pretrain_epochs > 0:
        print(f"==> Pretraining G for {args.pretrain_epochs} epochs (L1 + SSIM*{args.lambda_ssim_pre})")
        accum = max(1, args.accum_steps)
        # track variance collapse
        pre_low_var_streak = 0
        last_good_G_state = {k: v.cpu().clone() for k,v in G.state_dict().items()}
        for epoch in range(1, args.pretrain_epochs+1):
            G.train()
            g_opt.zero_grad(set_to_none=True)
            pre_loss_sum = 0.0
            pre_steps = 0
            var_accum = 0.0; var_batches = 0
            epoch_start_state = {k: v.cpu().clone() for k,v in G.state_dict().items()}
            import time as _time
            _epoch_t0 = _time.time()
            for step, (lr, hr, _) in enumerate(train_loader, start=1):
                lr, hr = lr.to(device), hr.to(device)
                with amp.autocast('cuda', enabled=use_amp):
                    sr = G(lr)
                    loss = l1(sr, hr)
                    if args.lambda_ssim_pre > 0:
                        loss = loss + (1.0 - ssim_torch(sr, hr)) * args.lambda_ssim_pre
                with torch.no_grad():
                    var_accum += sr.detach().var().item(); var_batches += 1
                loss_scaled = loss / accum
                scaler_g.scale(loss_scaled).backward()
                if step % accum == 0:
                    scaler_g.step(g_opt)
                    scaler_g.update()
                    g_opt.zero_grad(set_to_none=True)
                    # learning rate warmup (generator) if requested
                    if args.warmup_steps > 0 and g_update_count < args.warmup_steps:
                        g_update_count += 1
                        warm_ratio = min(1.0, g_update_count / args.warmup_steps)
                        for pg in g_opt.param_groups:
                            pg['lr'] = base_g_lr * warm_ratio
                pre_loss_sum += loss.item()
                pre_steps += 1
                # Heartbeat: periodic progress update within epoch
                if step == 1:
                    # Save a sample triplet for sanity at epoch start
                    try:
                        save_triplet(lr, sr, hr, f"{args.out_dir}/samples/pre_e{epoch:03d}_step_{step:05d}.png", nrow=min(args.batch,4))
                    except Exception:
                        pass
                if step % max(50, args.batch) == 0:
                    try:
                        if use_cuda:
                            torch.cuda.synchronize()
                        _elapsed = _time.time() - _epoch_t0
                        _spd = step / max(1.0, _elapsed)
                        print(f"[Pre {epoch:03d}] step {step}/{len(train_loader)} | { _spd:.2f } steps/s")
                    except Exception:
                        pass
            if step % accum != 0:
                scaler_g.step(g_opt); scaler_g.update(); g_opt.zero_grad(set_to_none=True)
                if args.warmup_steps > 0 and g_update_count < args.warmup_steps:
                    g_update_count += 1
                    warm_ratio = min(1.0, g_update_count / args.warmup_steps)
                    for pg in g_opt.param_groups:
                        pg['lr'] = base_g_lr * warm_ratio
            ps, ss, tt = validate(G, val_loader, device, use_amp=use_amp)
            avg_pre = pre_loss_sum / max(1, pre_steps)
            mean_var = var_accum / max(1,var_batches)
            collapse_note = ""
            if args.pre_collapse_var is not None and mean_var < args.pre_collapse_var:
                pre_low_var_streak += 1
                collapse_note = f" | VAR:{mean_var:.6f} ⚠ low ({pre_low_var_streak}/{args.pre_collapse_patience})"
            else:
                pre_low_var_streak = 0
            if getattr(args,'quick_diag', False):
                with torch.no_grad():
                    deltas = []
                    for n, init_w in qd['initial'].items():
                        cur = G.state_dict()[n]
                        deltas.append((cur - init_w).float().abs().mean().item())
                    mean_delta = float(np.mean(deltas)) if deltas else 0.0
                print(f"[Pre {epoch:03d}] L(avg):{avg_pre:.4f} | PSNR:{ps:.6f} SSIM:{ss:.6f} VAR:{mean_var:.6f}{collapse_note} Δw_mean:{mean_delta:.6e} TROPHY:{tt:.6f}")
            else:
                print(f"[Pre {epoch:03d}] L(avg):{avg_pre:.4f} | PSNR:{ps:.4f} SSIM:{ss:.5f} VAR:{mean_var:.5f}{collapse_note} TROPHY:{tt:.5f}")
            # early collapse abort
            if args.pre_collapse_var is not None and pre_low_var_streak >= args.pre_collapse_patience:
                print(f"[PRE-GUARD] Early stopping pretrain: variance stayed below {args.pre_collapse_var} for {pre_low_var_streak} epochs. Restoring last good weights.")
                G.load_state_dict(last_good_G_state)
                break
            else:
                # update last good snapshot
                last_good_G_state = {k: v.cpu().clone() for k,v in G.state_dict().items()}
            if epoch % args.save_every == 0:
                torch.save({"G": G.state_dict()}, f"{args.out_dir}/ckpts/esr_pre_e{epoch:03d}.pt")
        # re-enable tanh for GAN phase if disabled
        if getattr(args,'no_tanh_pretrain', False):
            G.set_tanh(True)
            print("[PRE] Re-enabled tanh activation for adversarial phase.")

    # ---- Phase 2: Adversarial fine-tune (ESRGAN)
    print(f"==> Adversarial training for {args.epochs} epochs")
    global_step = 0
    low_var_streak = 0
    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        accum = max(1, args.accum_steps)
        g_opt.zero_grad(set_to_none=True)
        d_opt.zero_grad(set_to_none=True)
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        adv_steps = 0
        var_accum = 0.0
        var_batches = 0
        for step, (lr, hr, _) in enumerate(train_loader, start=1):
            lr, hr = lr.to(device), hr.to(device)
            # ---- Discriminator ----
            with amp.autocast('cuda', enabled=use_amp):
                with torch.no_grad():
                    sr_tmp = G(lr)
                cond = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
                real_logits = D(cond, hr)
                fake_logits = D(cond, sr_tmp.detach())
                real_targets = torch.full_like(real_logits, args.smooth_real)
                fake_targets = torch.zeros_like(fake_logits)
                d_loss = bce(real_logits, real_targets) + bce(fake_logits, fake_targets)
            d_loss_scaled = d_loss / accum
            scaler_d.scale(d_loss_scaled).backward()
            if step % accum == 0:
                scaler_d.step(d_opt)
                scaler_d.update()
                d_opt.zero_grad(set_to_none=True)

            # ---- Generator ----
            with amp.autocast('cuda', enabled=use_amp):
                sr = G(lr)
                fake_logits = D(cond, sr)
                gan_loss = bce(fake_logits, real_targets) * args.lambda_gan
                l1_loss  = l1(sr, hr) * args.lambda_l1
                pf_sr, pf_hr = perc(sr), perc(hr)
                perc_loss = F.l1_loss(pf_sr, pf_hr) * args.lambda_perc
                g_loss = gan_loss + l1_loss + perc_loss
            g_loss_scaled = g_loss / accum
            scaler_g.scale(g_loss_scaled).backward()
            if step % accum == 0:
                scaler_g.step(g_opt)
                scaler_g.update()
                g_opt.zero_grad(set_to_none=True)
                if args.warmup_steps > 0 and g_update_count < args.warmup_steps:
                    g_update_count += 1
                    warm_ratio = min(1.0, g_update_count / args.warmup_steps)
                    for pg in g_opt.param_groups:
                        # only raise towards base_g_lr; do not exceed it
                        pg['lr'] = base_g_lr * warm_ratio

            global_step += 1
            if global_step % args.sample_every == 0:
                save_triplet(lr, sr, hr, f"{args.out_dir}/samples/step_{global_step:07d}.png", nrow=min(args.batch,4))
            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            adv_steps += 1
            # variance monitor (detach to avoid grad graph growth)
            with torch.no_grad():
                var_accum += sr.detach().var().item()
                var_batches += 1

        # flush leftover grads if loop ended mid-accum
        if step % accum != 0:
            scaler_d.step(d_opt); scaler_d.update(); d_opt.zero_grad(set_to_none=True)
            scaler_g.step(g_opt); scaler_g.update(); g_opt.zero_grad(set_to_none=True)
            if args.warmup_steps > 0 and g_update_count < args.warmup_steps:
                g_update_count += 1
                warm_ratio = min(1.0, g_update_count / args.warmup_steps)
                for pg in g_opt.param_groups:
                    pg['lr'] = base_g_lr * warm_ratio

        ps, ss, tt = validate(G, val_loader, device, use_amp=use_amp)
        mean_d = d_loss_sum / max(1, adv_steps)
        mean_g = g_loss_sum / max(1, adv_steps)
        mean_var = var_accum / max(1, var_batches)
        collapse_note = ""
        if mean_var < getattr(args,'var_guard_thresh', 0.01):
            low_var_streak += 1
            collapse_note = f" | VAR:{mean_var:.5f} ⚠ low ({low_var_streak}/{getattr(args,'var_guard_patience',2)})"
            print(f"[WARN] Low generator output variance ({mean_var:.6f}); possible collapse starting.")
            if low_var_streak >= getattr(args,'var_guard_patience',2):
                # adaptive discriminator LR reduction to ease pressure
                for pg in d_opt.param_groups:
                    pg['lr'] *= 0.5
                print(f"[ADAPT] Reduced D lr to {d_opt.param_groups[0]['lr']:.2e} after sustained low variance.")
                low_var_streak = 0  # reset after intervention
        else:
            low_var_streak = 0
        if getattr(args,'quick_diag', False):
            with torch.no_grad():
                deltas = []
                for n, init_w in qd.get('initial', {}).items():
                    cur = G.state_dict()[n]
                    deltas.append((cur - init_w).float().abs().mean().item())
                mean_delta = float(np.mean(deltas)) if deltas else 0.0
            print(f"[GAN {epoch:03d}] G(avg):{mean_g:.4f} D(avg):{mean_d:.4f} VAR:{mean_var:.6f}{collapse_note} Δw_mean:{mean_delta:.6e} | PSNR:{ps:.6f} SSIM:{ss:.6f} TROPHY:{tt:.5f}")
        else:
            print(f"[GAN {epoch:03d}] G(avg):{mean_g:.4f} D(avg):{mean_d:.4f} VAR:{mean_var:.5f}{collapse_note} | PSNR:{ps:.4f} SSIM:{ss:.5f} TROPHY:{tt:.5f}")

        if epoch % args.save_every == 0:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args)},
                       f"{args.out_dir}/ckpts/esrgan_e{epoch:03d}.pt")

    # Final save
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args)},
               f"{args.out_dir}/ckpts/esrgan_final.pt")
    print("✔ Training complete.")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ESRGAN Phase 1 (paired noisy→clean ultrasound)")
    p.add_argument("--noisy_dir",  type=str, default="./noisy", help="input LR/noisy images folder")
    p.add_argument("--clean_dir",  type=str, default="./data/Train400", help="target HR/clean images folder")
    p.add_argument("--test_dir",   type=str, default="./data/Test/Set68", help="optional test folder (paired by name)")
    p.add_argument("--out_dir",    type=str, default="runs_esrgan")
    p.add_argument("--scale",      type=int, default=1, choices=[1,2,4], help="1=enhance/denoise, 2/4=SR")
    p.add_argument("--crop",       type=int, default=256, help="HR crop size (center or random). Set None to disable",)
    p.add_argument("--rand_crop",  action="store_true", help="use random HR cropping instead of center crop")
    p.add_argument("--batch",      type=int, default=8)
    p.add_argument("--pretrain_epochs", type=int, default=10, help="SRResNet pretrain epochs (no GAN)")
    p.add_argument("--no_tanh_pretrain", action="store_true", help="disable final tanh during pretrain to avoid early saturation; re-enabled for GAN phase")
    p.add_argument("--epochs",     type=int, default=50, help="adversarial epochs")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--nf",         type=int, default=64, help="G feature channels")
    p.add_argument("--nb",         type=int, default=23, help="RRDB blocks")
    p.add_argument("--lambda_l1",  type=float, default=1.0)
    p.add_argument("--lambda_perc",type=float, default=0.1)
    p.add_argument("--lambda_gan", type=float, default=0.005)
    p.add_argument("--lambda_ssim_pre", type=float, default=0.5, help="extra SSIM during pretrain (0=off)")
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--save_every",   type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="enable mixed precision training (fp16 autocast)")
    p.add_argument("--skip_tf", action="store_true", help="skip TensorFlow GPU init to save VRAM")
    p.add_argument("--grad_ckpt", action="store_true", help="enable gradient checkpointing over RRDB blocks (less memory, slower)")
    p.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps to simulate larger batch")
    p.add_argument("--auto_batch", action="store_true", help="auto-reduce batch size on OOM (model-only heuristic)")
    p.add_argument("--smooth_real", type=float, default=1.0, help="one-sided label smoothing value for real labels (e.g. 0.9). 1.0 disables")
    p.add_argument("--spectral_norm", action="store_true", help="apply spectral normalization to discriminator conv layers")
    p.add_argument("--d_lr", type=float, default=None, help="optional discriminator-specific learning rate (defaults to --lr if omitted)")
    p.add_argument("--var_guard_thresh", type=float, default=0.01, help="epoch mean SR variance threshold signaling potential collapse")
    p.add_argument("--var_guard_patience", type=int, default=2, help="epochs below variance threshold before adaptive D lr halves")
    p.add_argument("--pre_collapse_var", type=float, default=0.0005, help="pretrain mean SR variance threshold; early stop if persistently below (set None to disable)")
    p.add_argument("--pre_collapse_patience", type=int, default=2, help="epochs below pre_collapse_var before aborting pretrain")
    p.add_argument("--warmup_steps", type=int, default=0, help="linear LR warmup steps for generator (applies across pretrain+GAN)")
    p.add_argument("--fail_on_warn", action="store_true", help="convert validation warnings into errors (strict mode)")
    p.add_argument("--dry_run", action="store_true", help="run a single forward/backward pass then exit (sanity check)")
    p.add_argument("--quick_diag", action="store_true", help="print per-epoch SR variance and mean parameter delta diagnostics")
    return p.parse_args()


def _warn(msg, strict=False):
    prefix = "[CHECK]"
    if strict:
        raise SystemExit(f"{prefix} ERROR: {msg}")
    print(f"{prefix} WARN: {msg}")


def validate_args(args):
    """Lightweight configuration & environment sanity checks.

    This validates argument ranges, dataset presence, and environment compatibility
    before launching a long training run. Non-fatal issues emit warnings unless
    --fail_on_warn is supplied, which upgrades them to errors.
    """
    strict = getattr(args, 'fail_on_warn', False)

    # Directory checks
    if not Path(args.noisy_dir).exists():
        _warn(f"noisy_dir '{args.noisy_dir}' does not exist", strict)
    if not Path(args.clean_dir).exists():
        _warn(f"clean_dir '{args.clean_dir}' does not exist", strict)
    if args.test_dir and (not Path(args.test_dir).exists()):
        _warn(f"test_dir '{args.test_dir}' not found; test metrics will be skipped", strict=False)

    # Numeric ranges
    if args.batch < 1:
        _warn("--batch must be >=1", strict)
    if args.accum_steps < 1:
        _warn("--accum_steps must be >=1", strict)
    if args.pretrain_epochs < 0 or args.epochs < 0:
        _warn("epochs must be non-negative", strict)
    if args.nf <= 0 or args.nb <= 0:
        _warn("nf and nb must be positive", strict)
    if args.lr <= 0:
        _warn("learning rate must be >0", strict)
    if args.save_every < 1:
        _warn("--save_every must be >=1", strict)
    if args.sample_every < 1:
        _warn("--sample_every must be >=1", strict)
    for name in ["lambda_l1","lambda_perc","lambda_gan","lambda_ssim_pre"]:
        if getattr(args, name) < 0:
            _warn(f"{name} should be >=0", strict)
    if args.pre_collapse_var is not None and args.pre_collapse_var < 0:
        _warn("--pre_collapse_var must be >=0 or None", strict)
    if args.pre_collapse_patience < 1:
        _warn("--pre_collapse_patience must be >=1", strict)
    if args.warmup_steps < 0:
        _warn("--warmup_steps must be >=0", strict)
    if not (0.5 <= args.smooth_real <= 1.0):
        _warn("--smooth_real should be within [0.5,1.0] (suggest 0.85-0.95) or 1.0 to disable", strict)
    # ...existing code...

    # Scale / crop consistency
    if args.scale not in (1,2,4):
        _warn("scale should be one of 1,2,4 (architecture assumes power-of-two)", strict)
    if args.crop is not None and args.crop % args.scale != 0:
        _warn(f"crop {args.crop} not divisible by scale {args.scale}; may cause size mismatch", strict=False)
    if args.grad_ckpt and args.accum_steps > 1:
        _warn("Using both gradient checkpointing and accumulation slows training; ensure needed", strict=False)
    if args.amp and args.cpu:
        _warn("--amp has no effect on CPU", strict=False)

    # CUDA availability for GPU-intended options
    if (args.amp or args.grad_ckpt) and (not torch.cuda.is_available() or args.cpu):
        _warn("Requested GPU-specific features but CUDA not available or --cpu set", strict)

    # Quick dataset pairing count (non-fatal)
    try:
        if Path(args.noisy_dir).exists() and Path(args.clean_dir).exists():
            noisy_names = {p.name for p in Path(args.noisy_dir).iterdir() if p.is_file()}
            clean_names = {p.name for p in Path(args.clean_dir).iterdir() if p.is_file()}
            common = noisy_names & clean_names
            if len(common) == 0:
                _warn("No filename overlap between noisy_dir and clean_dir", strict)
            else:
                if len(common) < 10:
                    _warn(f"Very few paired images detected ({len(common)}); training may be unstable", strict=False)
    except Exception as e:
        _warn(f"Dataset pairing check failed: {e}", strict=False)

    # Informational: effective batch size
    eff_batch = args.batch * max(1, args.accum_steps)
    print(f"[CHECK] effective batch (batch * accum_steps) = {args.batch} * {args.accum_steps} = {eff_batch}")

    # Provide suggested allocator tweak if CUDA available
    if torch.cuda.is_available() and not args.cpu:
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
            print("[CHECK] Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 to reduce fragmentation")

    return args

if __name__ == "__main__":
    args = parse_args()
    args = validate_args(args)
    if args.dry_run:
        print("[DRY-RUN] Performing a short configuration validation forward pass then exiting.")
        # Minimal synthetic dry run (on CPU or GPU) to catch shape / OOM early.
        device = torch.device('cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu')
        G_test = ESRGAN_G(scale=args.scale, nf=args.nf, nb=min(3, args.nb), grad_ckpt=False).to(device)
        H = W = args.crop if args.crop else 128
        x = torch.randn(args.batch,1,H//args.scale,W//args.scale, device=device)
        with amp.autocast('cuda', enabled=(args.amp and device.type=='cuda')):
            y = G_test(x)
        print(f"[DRY-RUN] Input {tuple(x.shape)} -> Output {tuple(y.shape)} OK")
        raise SystemExit(0)
    train(args)
