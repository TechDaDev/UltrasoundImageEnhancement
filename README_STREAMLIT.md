# 🔬 Ultrasound Image Enhancement App

A powerful Streamlit web application for enhancing ultrasound images using a hybrid AI pipeline combining **DnCNN** (denoising) and **ESRGAN** (enhancement).

## 🌟 Features

- **Two-Stage Enhancement Pipeline**:
  1. **DnCNN**: Deep learning-based denoising to remove noise while preserving important details
  2. **ESRGAN**: Enhanced Super-Resolution GAN for image refinement and quality improvement

- **Interactive Web Interface**: Easy-to-use Streamlit interface with drag-and-drop upload
- **Real-time Processing**: View results immediately after processing
- **Downloadable Results**: Download enhanced images with embedded titles
- **Quality Metrics**: View PSNR and SSIM metrics for quality assessment
- **Side-by-Side Comparison**: Compare original, denoised, and enhanced images

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)
- Trained models in `runs_hybrid_paired/` directory:
  - `dncnn_model/dncnn_final.keras`
  - `esrgan_model/ckpts/esrgan_final.pt`

## 🚀 Quick Start

### Option 1: Using the Launcher Script (Recommended)

```bash
./run_app.sh
```

### Option 2: Manual Setup

1. **Create a virtual environment** (if not already created):
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit numpy Pillow torch tensorflow keras
   ```

4. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📖 Usage

1. **Launch the app** using one of the methods above
2. **Open your browser** to the URL shown in the terminal (typically `http://localhost:8501`)
3. **Upload an ultrasound image** using the file uploader
4. **Click "Enhance Image"** to process
5. **View results** in the tabs:
   - **Comparison**: Side-by-side view of all stages
   - **DnCNN Output**: Denoised result
   - **ESRGAN Output**: Final enhanced result
   - **Metrics**: Quality metrics (PSNR, SSIM)
6. **Download results** using the download buttons (images include titles)

## 🎨 Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- TIFF

## 📊 Quality Metrics

The app calculates and displays:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality (higher is better, typically 20-50 dB)
- **SSIM (Structural Similarity Index)**: Measures structural similarity (closer to 1.0 is better)

## 🏗️ Architecture

```
Input Image
    ↓
[DnCNN Denoising]
    ↓
Denoised Image
    ↓
[ESRGAN Enhancement]
    ↓
Enhanced Image
```

### Model Details

- **DnCNN**: 17-layer convolutional neural network trained for image denoising
- **ESRGAN**: Residual-in-Residual Dense Block (RRDB) based generator with:
  - 32 feature channels (nf=32)
  - 12 RRDB blocks (nb=12)
  - Optimized for medical imaging

## 📁 Project Structure

```
.
├── streamlit_app.py              # Main Streamlit application
├── hybrid_training.py            # Training pipeline and HybridDenoiseEnhancer class
├── esrgan.py                     # ESRGAN model and training code
├── models.py                     # DnCNN model definition
├── gpu_utils.py                  # GPU configuration utilities
├── run_app.sh                    # Launcher script
├── requirements_streamlit.txt    # Python dependencies
└── runs_hybrid_paired/           # Trained models directory
    ├── dncnn_model/
    │   └── dncnn_final.keras
    └── esrgan_model/
        └── ckpts/
            └── esrgan_final.pt
```

## 🔧 Configuration

The app automatically detects:
- GPU availability (uses CUDA if available, otherwise CPU)
- Model paths (expects models in `runs_hybrid_paired/`)

## 🐛 Troubleshooting

### Models not found
Ensure the trained models exist at:
- `runs_hybrid_paired/dncnn_model/dncnn_final.keras`
- `runs_hybrid_paired/esrgan_model/ckpts/esrgan_final.pt`

### Out of memory errors
- Try using CPU mode (the app will automatically fall back if GPU is not available)
- Process smaller images
- Close other applications using GPU memory

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements_streamlit.txt
```

## 💡 Tips

- For best results, use grayscale ultrasound images
- Larger images may take longer to process
- GPU acceleration significantly speeds up processing
- Download buttons include titles embedded in the images

## 📝 License

This project uses:
- DnCNN architecture for denoising
- ESRGAN architecture for enhancement
- Streamlit for the web interface

## 🙏 Acknowledgments

- DnCNN: "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
- ESRGAN: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
- Streamlit: Open-source app framework for Machine Learning and Data Science

## 📧 Support

For issues or questions, please check the troubleshooting section above or review the code comments in `streamlit_app.py`.

---

**Powered by DnCNN + ESRGAN Hybrid Pipeline | Built with ❤️ using Streamlit**
