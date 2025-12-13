"""
Streamlit App for Ultrasound Image Enhancement
Uses DnCNN + ESRGAN Hybrid Pipeline
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import torch
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hybrid_training import HybridDenoiseEnhancer


# Page Configuration
st.set_page_config(
    page_title="Ultrasound Image Enhancement",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS matching the mockup
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
        padding: 2rem 1rem;
    }
    
    /* Primary button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Upload box styling */
    .upload-box {
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    /* Info box with gradient border */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Card styling for results */
    .result-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.4);
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(102, 126, 234, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #888;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 2px solid #667eea;
        color: #667eea;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    .image-container:hover {
        border-color: #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        border-radius: 10px;
        border-left-width: 5px;
    }
    
    /* Section headers */
    h3 {
        color: #667eea;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.2);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load the hybrid model (cached for performance)"""
    try:
        dncnn_path = Path("./runs_hybrid_paired/dncnn_model/DnCNN_S10_B512.h5")
        esrgan_path = Path("./scripts/esrgan_hybrid_model/esrgan_hybrid_best.pth")
        
        if not dncnn_path.exists():
            st.error(f"❌ DnCNN model not found at {dncnn_path}")
            return None
        
        if not esrgan_path.exists():
            st.error(f"❌ ESRGAN model not found at {esrgan_path}")
            return None
        
        with st.spinner("🔄 Loading AI models... This may take a moment."):
            enhancer = HybridDenoiseEnhancer(
                dncnn_weights=dncnn_path,
                esrgan_weights=esrgan_path,
                force_cpu_flag=not torch.cuda.is_available(),
                verbose=False
            )
        return enhancer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None


def add_title_to_image(img_array, title, font_size=30):
    """Add a title banner to the top of an image"""
    if isinstance(img_array, np.ndarray):
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        if img_array.ndim == 2:
            img = Image.fromarray(img_array, mode='L')
        else:
            img = Image.fromarray(img_array)
    else:
        img = img_array
    
    title_height = font_size + 20
    new_height = img.height + title_height
    new_img = Image.new('RGB' if img.mode == 'RGB' else 'L', 
                        (img.width, new_height), 
                        color=(20, 20, 30) if img.mode == 'RGB' else 20)
    
    if img.mode == 'L':
        img_rgb = Image.new('RGB', img.size)
        img_rgb.paste(img)
        new_img = Image.new('RGB', (img.width, new_height), color=(20, 20, 30))
        new_img.paste(img_rgb, (0, title_height))
    else:
        new_img.paste(img, (0, title_height))
    
    draw = ImageDraw.Draw(new_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (img.width - text_width) // 2
    text_y = (title_height - font_size) // 2
    
    shadow_offset = 2
    draw.text((text_x + shadow_offset, text_y + shadow_offset), title, 
              fill=(0, 0, 0), font=font)
    draw.text((text_x, text_y), title, fill=(102, 126, 234), font=font)
    
    return new_img


def process_image(enhancer, uploaded_image):
    """Process uploaded image through the hybrid pipeline"""
    try:
        img = Image.open(uploaded_image).convert('L')
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        original = img_array.copy()
        
        with st.spinner("🔬 Stage 1: Denoising with DnCNN..."):
            denoised = enhancer.denoise(img_array)
        
        with st.spinner("✨ Stage 2: Enhancing with ESRGAN..."):
            enhanced = enhancer.enhance(denoised)
        
        return original, denoised, enhanced
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None, None


def image_to_bytes(img_array, title=None, format='PNG'):
    """Convert numpy array to downloadable bytes with optional title"""
    if title:
        img_with_title = add_title_to_image(img_array, title)
    else:
        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        if img_array.ndim == 2:
            img_with_title = Image.fromarray(img_array, mode='L')
        else:
            img_with_title = Image.fromarray(img_array)
    
    buf = io.BytesIO()
    img_with_title.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def apply_filters(image_np, contrast=1.0, brightness=1.0, sharpness=1.0):
    """Apply filters to numpy image [0,1]"""
    # Convert to PIL
    if image_np.dtype != np.uint8:
        img_pil = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    else:
        img_pil = Image.fromarray(image_np)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast)
    
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness)
        
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(sharpness)
        
    # Convert back to np [0,1]
    return np.array(img_pil, dtype=np.float32) / 255.0



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
        return None, None


# Main App
def main():
    # Header with enhanced styling
    st.markdown("<h1>🔬 Ultrasound Image Enhancement</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div class='info-box'>
            <p style='margin: 0; font-size: 1.2rem; line-height: 1.6;'>
                <strong style='color: #667eea; font-size: 1.3rem;'>Advanced AI-Powered Enhancement Pipeline</strong><br>
                <span style='color: #aaa;'>Combines DnCNN denoising with ESRGAN refinement for superior ultrasound image quality</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### 📋 About This App")
        st.info("""
        **🔬 Two-Stage Enhancement Pipeline:**
        
        **Stage 1: DnCNN Denoising**
        - Deep learning-based noise removal
        - Preserves important image details
        - Trained on paired ultrasound data
        
        **Stage 2: ESRGAN Refinement**
        - Enhanced super-resolution GAN
        - Improves clarity and sharpness
        - Optimized for medical imaging
        
        **📁 Supported Formats:**
        PNG, JPG, JPEG, BMP, TIFF
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ System Information")
        device = "🎮 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.success(f"**Processing Device:**\n{device}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.info(f"**GPU Model:**\n{gpu_name}")
        
        st.markdown("---")
        st.markdown("### 📊 Pipeline Steps")
        st.markdown("""
        1. Upload ultrasound image
        2. Click 'Enhance Image'
        3. View results in tabs
        4. Download enhanced images
        """)
    
    # Load models
    enhancer = load_models()
    
    if enhancer is None:
        st.error("❌ Failed to load models. Please ensure the trained models are available in the correct directory.")
        st.stop()
    
    st.success("✅ AI Models loaded and ready!")
    
    # File uploader section
    st.markdown("### 📤 Upload Ultrasound Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a grayscale ultrasound image for AI-powered enhancement",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("processed_file_id") != file_id and "has_results" in st.session_state:
            del st.session_state.has_results

        # Display original image in a card
        st.markdown("""
            <div class='result-card'>
                <h4 style='color: #667eea; margin-bottom: 1rem;'>📥 Original Uploaded Image</h4>
            </div>
        """, unsafe_allow_html=True)
        
        original_img = Image.open(uploaded_file)
        st.image(original_img, width='stretch', caption="Original Ultrasound Image")
        
        # Enhanced process button
        st.markdown("<br>", unsafe_allow_html=True)
        clicked = st.button("🚀 Enhance Image with AI", type="primary", width='stretch')
        
        if clicked:
            # Process the image
            with st.spinner("⚡ Processing your image through the AI pipeline..."):
                original, denoised, enhanced = process_image(enhancer, uploaded_file)
            
            if original is not None and denoised is not None and enhanced is not None:
                st.session_state.original = original
                st.session_state.denoised = denoised
                st.session_state.enhanced = enhanced
                st.session_state.processed_file_id = file_id
                st.session_state.has_results = True
                st.balloons()
                st.success("✅ Enhancement complete! View results below.")
                
        if True:
            if st.session_state.get("processed_file_id") == file_id and st.session_state.get("has_results"):
                original = st.session_state.original
                denoised = st.session_state.denoised
                enhanced = st.session_state.enhanced
                
                # Display results in enhanced tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Side-by-Side Comparison", 
                    "🔬 DnCNN Denoised", 
                    "✨ ESRGAN Enhanced",
                    "📈 Quality Metrics",
                    "🎨 Interactive Studio"
                ])
                
                with tab1:
                    st.markdown("### 🔍 Compare All Stages")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        st.markdown("""
                            <div class='result-card'>
                                <h4 style='text-align: center; color: #888;'>📥 Original</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(original, width='stretch', clamp=True)
                    
                    with comp_col2:
                        st.markdown("""
                            <div class='result-card'>
                                <h4 style='text-align: center; color: #667eea;'>🔬 DnCNN Denoised</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(denoised, width='stretch', clamp=True)
                    
                    with comp_col3:
                        st.markdown("""
                            <div class='result-card'>
                                <h4 style='text-align: center; color: #764ba2;'>✨ ESRGAN Enhanced</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.image(enhanced, width='stretch', clamp=True)
                
                with tab2:
                    st.markdown("### 🔬 DnCNN Denoising Result")
                    st.markdown("""
                        <div class='info-box'>
                            <p style='margin: 0;'>
                                <strong>Stage 1 Complete:</strong> Noise removed while preserving structural details
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(denoised, width='stretch', clamp=True, caption="DnCNN Denoised Output")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    dncnn_bytes = image_to_bytes(denoised, title="DnCNN Denoised")
                    st.download_button(
                        label="⬇️ Download DnCNN Output (with title)",
                        data=dncnn_bytes,
                        file_name="dncnn_denoised.png",
                        mime="image/png",
                        width='stretch'
                    )
                
                with tab3:
                    st.markdown("### ✨ ESRGAN Enhancement Result")
                    st.markdown("""
                        <div class='info-box'>
                            <p style='margin: 0;'>
                                <strong>Stage 2 Complete:</strong> Image refined and enhanced for maximum clarity
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.image(enhanced, width='stretch', clamp=True, caption="ESRGAN Enhanced Output")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    esrgan_bytes = image_to_bytes(enhanced, title="ESRGAN Enhanced")
                    st.download_button(
                        label="⬇️ Download ESRGAN Output (with title)",
                        data=esrgan_bytes,
                        file_name="esrgan_enhanced.png",
                        mime="image/png",
                        width='stretch'
                    )
                
                with tab4:
                    st.markdown("### 📈 Image Quality Metrics")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Calculate metrics
                    psnr_dn, ssim_dn = calculate_metrics(original, denoised)
                    psnr_es, ssim_es = calculate_metrics(original, enhanced)
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.markdown("""
                            <div class='metric-card'>
                                <h3 style='color: #667eea; margin-bottom: 1.5rem;'>🔬 DnCNN Metrics</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if psnr_dn is not None:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("PSNR", f"{psnr_dn:.2f} dB", delta="Quality Score")
                            with col_b:
                                st.metric("SSIM", f"{ssim_dn:.4f}", delta="Similarity")
                    
                    with metric_col2:
                        st.markdown("""
                            <div class='metric-card'>
                                <h3 style='color: #764ba2; margin-bottom: 1.5rem;'>✨ ESRGAN Metrics</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if psnr_es is not None:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("PSNR", f"{psnr_es:.2f} dB", delta="Quality Score")
                            with col_b:
                                st.metric("SSIM", f"{ssim_es:.4f}", delta="Similarity")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("""
                    **📊 Understanding the Metrics:**
                    
                    - **PSNR (Peak Signal-to-Noise Ratio):** Measures image quality in decibels (dB)
                      - Higher values indicate better quality
                      - Typical range: 20-50 dB
                      - Values above 30 dB are considered good
                    
                    - **SSIM (Structural Similarity Index):** Measures structural similarity
                      - Range: 0 to 1 (1 = perfect similarity)
                      - Values above 0.9 indicate excellent preservation of structure
                    """)

                with tab5:
                    st.markdown("### 🎨 Interactive Post-Processing")
                    st.markdown("Adjust filters on the ESRGAN output to potentially improve quality.")
                    
                    filter_col1, filter_col2 = st.columns([1, 2])
                    
                    with filter_col1:
                        st.markdown("#### 🎛️ Controls")
                        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
                        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
                        
                        st.markdown("---")
                        st.markdown("#### 📊 Real-time Metrics")
                        
                        # Apply filters
                        filtered_img = apply_filters(enhanced, contrast, brightness, sharpness)
                        
                        # Calculate New Metrics
                        psnr_filt, ssim_filt = calculate_metrics(original, filtered_img)
                        
                        # Calculate Delta
                        psnr_orig, ssim_orig = calculate_metrics(original, enhanced) # Re-calc or reuse
                        # To be safe, re-calc or use vars if available. They are in local scope from Tab4 if computed there? 
                        # No, scope issues might exist if Tab4 wasn't rendered. Better re-calculate.
                        
                        psnr_delta = psnr_filt - (psnr_orig if psnr_orig else 0)
                        ssim_delta = ssim_filt - (ssim_orig if ssim_orig else 0)
                        
                        st.metric("New PSNR", f"{psnr_filt:.2f} dB", delta=f"{psnr_delta:+.2f}")
                        st.metric("New SSIM", f"{ssim_filt:.4f}", delta=f"{ssim_delta:+.4f}")
                        
                    with filter_col2:
                        st.markdown("#### 🖼️ Filtered Result")
                        st.image(filtered_img, width='stretch', clamp=True, caption="Filtered Output")
                        
                        filt_bytes = image_to_bytes(filtered_img, title="Filtered Enhanced")
                        st.download_button(
                            label="⬇️ Download Filtered Image",
                            data=filt_bytes,
                            file_name="filtered_enhanced.png",
                            mime="image/png",
                            width='stretch'
                        )
                
                
                # Download all results section
                st.markdown("---")
                st.markdown("### 📦 Download All Results")
                st.markdown("<br>", unsafe_allow_html=True)
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                
                with dl_col1:
                    st.markdown("""
                        <div class='result-card' style='text-align: center; padding: 1rem;'>
                            <h4 style='color: #888;'>📥 Original Image</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    orig_bytes = image_to_bytes(original, title="Original")
                    st.download_button(
                        label="⬇️ Download Original",
                        data=orig_bytes,
                        file_name="original.png",
                        mime="image/png",
                        width='stretch'
                    )
                
                with dl_col2:
                    st.markdown("""
                        <div class='result-card' style='text-align: center; padding: 1rem;'>
                            <h4 style='color: #667eea;'>🔬 DnCNN Output</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    dncnn_bytes = image_to_bytes(denoised, title="DnCNN Denoised")
                    st.download_button(
                        label="⬇️ Download DnCNN",
                        data=dncnn_bytes,
                        file_name="dncnn_output.png",
                        mime="image/png",
                        width='stretch'
                    )
                
                with dl_col3:
                    st.markdown("""
                        <div class='result-card' style='text-align: center; padding: 1rem;'>
                            <h4 style='color: #764ba2;'>✨ ESRGAN Output</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    esrgan_bytes = image_to_bytes(enhanced, title="ESRGAN Enhanced")
                    st.download_button(
                        label="⬇️ Download ESRGAN",
                        data=esrgan_bytes,
                        file_name="esrgan_output.png",
                        mime="image/png",
                        width='stretch'
                    )
    
    else:
        # Enhanced upload prompt
        st.markdown("""
            <div class='upload-box'>
                <h2 style='color: #667eea; margin-bottom: 1rem;'>👆 Upload an Ultrasound Image to Begin</h2>
                <p style='font-size: 1.1rem; color: #888;'>
                    Drag and drop your image here, or click to browse<br>
                    <span style='font-size: 0.9rem;'>Supported formats: PNG, JPG, JPEG, BMP, TIFF</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show example workflow
        st.markdown("---")
        st.markdown("### 🎯 How It Works")
        
        workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
        
        with workflow_col1:
            st.markdown("""
                <div class='result-card' style='text-align: center;'>
                    <h3 style='color: #667eea;'>1️⃣ Upload</h3>
                    <p style='color: #aaa;'>Select your ultrasound image</p>
                </div>
            """, unsafe_allow_html=True)
        
        with workflow_col2:
            st.markdown("""
                <div class='result-card' style='text-align: center;'>
                    <h3 style='color: #667eea;'>2️⃣ Process</h3>
                    <p style='color: #aaa;'>AI enhances your image</p>
                </div>
            """, unsafe_allow_html=True)
        
        with workflow_col3:
            st.markdown("""
                <div class='result-card' style='text-align: center;'>
                    <h3 style='color: #667eea;'>3️⃣ Download</h3>
                    <p style='color: #aaa;'>Get enhanced results</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <p style='color: #667eea; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;'>
                Powered by DnCNN + ESRGAN Hybrid AI Pipeline
            </p>
            <p style='color: #888; font-size: 0.9rem;'>
                Built with ❤️ using Streamlit | Advanced Medical Image Enhancement
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
