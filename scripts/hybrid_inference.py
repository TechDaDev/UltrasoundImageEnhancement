
import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add parent directory to path to find esrgan.py and gpu_utils.py
sys.path.append(str(Path(__file__).resolve().parent.parent))

print("Script starting...", flush=True)

from esrgan import ESRGAN_G
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract

# Define DnCNN model structure if needed (fallback)
def DnCNN():
    inpt = Input(shape=(None,None,1))
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   
    model = Model(inputs=inpt, outputs=x)
    return model

class HybridSystem:
    def __init__(self, dncnn_path, esrgan_path, force_cpu=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
        print(f"Using Torch device: {self.device}")
        
        # Load DnCNN
        print(f"Loading DnCNN: {dncnn_path}")
        try:
            self.dncnn = load_model(str(dncnn_path), compile=False)
        except Exception as e:
            print(f"Standard load failed: {e}. Trying to build structure and load weights...")
            self.dncnn = DnCNN()
            self.dncnn.load_weights(str(dncnn_path))

        # Load ESRGAN - refinement model trained on DnCNN outputs
        print(f"Loading ESRGAN: {esrgan_path}")
        self.esrgan_G = ESRGAN_G(
            scale=1,
            nf=32,
            nb=8,
            gc=16,
            grad_ckpt=False,
            use_tanh=False,   # VERY IMPORTANT
        ).to(self.device)

        ckpt = torch.load(esrgan_path, map_location=self.device)
        if "G" in ckpt:
            state = ckpt["G"]
        else:
            state = ckpt
        self.esrgan_G.load_state_dict(state, strict=False)
        self.esrgan_G.eval()

        # # Load ESRGAN
        # print(f"Loading ESRGAN: {esrgan_path}")
        # # Assuming scale=1 based on instructions for hybrid ultrasound
        # self.esrgan_G = ESRGAN_G(scale=1, nf=32, nb=12, use_tanh=True).to(self.device)
        
        # ckpt = torch.load(esrgan_path, map_location=self.device)
        # if "G" in ckpt:
        #     state = ckpt["G"]
        # else:
        #     state = ckpt
        # self.esrgan_G.load_state_dict(state, strict=False)
        # self.esrgan_G.eval()

    def run_dncnn(self, img_path_or_array):
        if isinstance(img_path_or_array, (str, Path)):
            img = Image.open(img_path_or_array).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
        else:
            arr = img_path_or_array
            
        h, w = arr.shape
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4
        arr_pad = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')

        x = arr_pad.reshape(1, arr_pad.shape[0], arr_pad.shape[1], 1)
        y = self.dncnn.predict(x, verbose=0)
        out_full = y.reshape(arr_pad.shape)
        out = out_full[:h, :w]
        out = np.clip(out, 0.0, 1.0)
        return out # float32 [0,1]
    
    @torch.no_grad()
    def run_esrgan(self, dncnn_out_01: np.ndarray) -> np.ndarray:
        """
        Refine DnCNN output using ESRGAN.
        dncnn_out_01: numpy array in [0,1], shape (H,W) or (H,W,1)
        Returns: numpy array in [0,1], shape (H,W)
        """
        x = dncnn_out_01.astype(np.float32)

        # Ensure HxW
        if x.ndim == 3:
            x = x[..., 0]

        # To tensor [1,1,H,W], in [0,1]
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)

        # Generator output is unconstrained; we squash with sigmoid to [0,1]
        y = self.esrgan_G(t)
        y = torch.sigmoid(y)

        y = y.squeeze(0).squeeze(0).cpu().numpy()  # [H,W]
        y = np.clip(y, 0.0, 1.0)
        return y


    # @torch.no_grad()
    # def run_esrgan(self, dncnn_out_01: np.ndarray) -> np.ndarray:
    #     # Match training normalization: [0,1] -> [-1,1]
    #     x = dncnn_out_01.astype(np.float32)
    #     x = (x - 0.5) / 0.5 

    #     t = torch.from_numpy(x).to(self.device)
    #     if t.ndim == 2:
    #          t = t.unsqueeze(0).unsqueeze(0) # [1,1,H,W]
    #     elif t.ndim == 3:
    #          t = t.permute(2,0,1).unsqueeze(0)

    #     # Output in [-1,1] (tanh)
    #     y = self.esrgan_G(t)
        
    #     y = y.squeeze(0).cpu().numpy() # [C,H,W] or [1,H,W]
    #     if y.shape[0] == 1:
    #         y = y[0]
    #     else:
    #         y = y.transpose(1, 2, 0)

    #     # Convert back: [-1,1] -> [0,1]
    #     y = (y + 1.0) / 2.0
    #     y = np.clip(y, 0.0, 1.0)
    #     return y

    def process(self, img_path):
        x_dn = self.run_dncnn(img_path)
        x_hybrid = self.run_esrgan(x_dn)
        return x_dn, x_hybrid

def calculate_metrics(original, processed):
    """Calculate PSNR and SSIM metrics"""
    try:
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        mean_orig = np.mean(original)
        mean_proc = np.mean(processed)
        var_orig = np.var(original)
        var_proc = np.var(processed)
        cov = np.mean((original - mean_orig) * (processed - mean_proc))
        
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        
        ssim = ((2 * mean_orig * mean_proc + c1) * (2 * cov + c2)) / \
               ((mean_orig**2 + mean_proc**2 + c1) * (var_orig + var_proc + c2))
        
        return psnr, ssim
    except:
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Hybrid Inference Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--dncnn_path", type=str, default="../runs_hybrid_paired/dncnn_model/DnCNN_S10_B512.h5")
    parser.add_argument("--esrgan_path", type=str, default="../runs_hybrid_paired/esrgan_model/ckpts/esrgan_final.pt")
    parser.add_argument("--no_save", action="store_true", help="Run inference and evaluation but do not save output images")
    
    args = parser.parse_args()
    
    hybrid = HybridSystem(args.dncnn_path, args.esrgan_path)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inp = Path(args.input)
    if inp.is_file():
        files = [inp]
        out_subdir = output_dir
    else:
        files = sorted([f for f in inp.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
        out_subdir = output_dir
        
    print(f"Processing {len(files)} files...")
    
    # Setup MD file
    if inp.is_dir():
        md_path = inp / "comparison_metrics.md"
    else:
        md_path = inp.parent / "comparison_metrics.md"
        
    print(f"Saving metrics to {md_path}")
    
    with open(md_path, "w") as f_md:
        # Print Header
        print(f"\n{'='*95}")
        header = f"| {'Image Name':<30} | {'PSNR (DnCNN)':<12} | {'SSIM (DnCNN)':<12} | {'PSNR (Hybrid)':<12} | {'SSIM (Hybrid)':<12} |"
        sep_line = f"|{'-'*32}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|"
        
        print(header)
        print(sep_line)
        
        f_md.write(f"# Hybrid System Evaluation Metrics\n\n")
        f_md.write(header + "\n")
        f_md.write(sep_line + "\n")
        
        for f in files:
            try:
                # Load original for metrics (assuming input is reference/target for now per instructions Option B)
                # If you have separate GT, logic would need to find it matching f.name
                gt_img = Image.open(f).convert("L")
                gt_arr = np.array(gt_img, dtype=np.float32) / 255.0
                
                dn, es = hybrid.process(f)
                
                # Calculate metrics
                psnr_dn, ssim_dn = calculate_metrics(gt_arr, dn)
                psnr_hy, ssim_hy = calculate_metrics(gt_arr, es)
                
                # Print row
                row_str = f"| {f.name[:30]:<30} | {psnr_dn:<12.2f} | {ssim_dn:<12.4f} | {psnr_hy:<12.2f} | {ssim_hy:<12.4f} |"
                print(row_str)
                f_md.write(row_str + "\n")
                
                if not args.no_save:
                    # Save DnCNN output
                    dn_uint8 = (dn * 255.0).astype(np.uint8)
                    Image.fromarray(dn_uint8).save(out_subdir / (f.stem + "_dncnn.png"))
                    
                    # Save Hybrid output
                    es_uint8 = (es * 255.0).astype(np.uint8)
                    Image.fromarray(es_uint8).save(out_subdir / (f.stem + "_hybrid.png"))
                
            except Exception as e:
                print(f"Error on {f.name}: {e}")
                
        print(f"{'='*95}\n")
        print("Done! Results saved to", output_dir)



if __name__ == "__main__":
    main()
