"""
Streamlit App for Ultrasound Image Enhancement
Uses DnCNN + ESRGAN Hybrid Pipeline
Enhanced Premium UI
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

# Premium Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Root variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --dark-gradient: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --neon-blue: #00d4ff;
        --neon-purple: #b24bf3;
        --neon-pink: #f72585;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.5);
    }
    
    /* Global font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main background with animated gradient */
    .main {
        background: var(--dark-gradient);
        padding: 2rem 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0d1b2a 100%);
    }
    
    /* Animated gradient orbs background */
    .main::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(102, 126, 234, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(118, 75, 162, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 212, 255, 0.08) 0%, transparent 40%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(2%, 2%) rotate(1deg); }
        50% { transform: translate(-1%, 3%) rotate(-1deg); }
        75% { transform: translate(3%, -2%) rotate(1deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.6); }
    }
    
    /* Primary button styling with glow effect */
    .stButton>button {
        width: 100%;
        background: var(--primary-gradient);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        padding: 1rem 2rem;
        border-radius: 16px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 15px rgba(102, 126, 234, 0.4),
            0 0 30px rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 
            0 12px 35px rgba(102, 126, 234, 0.5),
            0 0 50px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(1.01);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-8px);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(102, 126, 234, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Upload box styling with animated border */
    .upload-box {
        border: 2px dashed transparent;
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        background: 
            linear-gradient(rgba(10, 10, 15, 0.9), rgba(10, 10, 15, 0.9)) padding-box,
            linear-gradient(135deg, #667eea, #764ba2, #f093fb, #667eea) border-box;
        background-size: 100% 100%, 300% 300%;
        animation: borderGradient 4s ease infinite;
        margin: 2rem 0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes borderGradient {
        0%, 100% { background-position: 0% 50%, 0% 50%; }
        50% { background-position: 0% 50%, 100% 50%; }
    }
    
    .upload-box::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at center, rgba(102, 126, 234, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .upload-box:hover::before {
        opacity: 1;
    }
    
    .upload-box:hover {
        transform: scale(1.01);
        box-shadow: 0 0 60px rgba(102, 126, 234, 0.2);
    }
    
    /* Title styling with gradient text and glow */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #00d4ff 50%, #764ba2 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        animation: shimmer 3s linear infinite;
        text-shadow: 0 0 60px rgba(102, 126, 234, 0.5);
    }
    
    .subtitle {
        color: var(--text-secondary);
        text-align: center;
        font-size: 1.3rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* Info box with gradient border */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border-left: 4px solid;
        border-image: var(--primary-gradient) 1;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Result cards with neon hover */
    .result-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .result-card:hover::before {
        opacity: 1;
    }
    
    .result-card:hover {
        transform: translateY(-6px);
        box-shadow: 
            0 20px 50px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Stage indicator pills */
    .stage-pill {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    
    .stage-pill.stage-1 {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(102, 126, 234, 0.1) 100%);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .stage-pill.stage-2 {
        background: linear-gradient(135deg, rgba(118, 75, 162, 0.2) 0%, rgba(118, 75, 162, 0.1) 100%);
        color: #b24bf3;
        border: 1px solid rgba(118, 75, 162, 0.3);
    }
    
    /* Metric card styling with glow */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.1) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    /* Tab styling - Premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 0.75rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 12px;
        color: var(--text-muted);
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button styling */
    .stDownloadButton>button {
        background: rgba(102, 126, 234, 0.1);
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: #667eea;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stDownloadButton>button:hover {
        background: var(--primary-gradient);
        border-color: transparent;
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Image container with glow */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(102, 126, 234, 0.15);
        transition: all 0.4s ease;
    }
    
    .image-container:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.5),
            0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 26, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Success/Info boxes */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        border-left: 4px solid #10b981 !important;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        border-left: 4px solid #667eea !important;
    }
    
    /* Section headers */
    h1 {
        color: var(--text-primary);
    }
    
    h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h4 {
        color: var(--text-secondary);
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 2.5rem 0;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: var(--primary-gradient) !important;
    }
    
    .stSlider > div > div > div > div {
        background: white !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .status-badge.ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-badge.processing {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Feature card */
    .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.3);
        background: rgba(102, 126, 234, 0.05);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Workflow step styling */
    .workflow-step {
        position: relative;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        transition: all 0.4s ease;
    }
    
    .workflow-step:hover {
        background: rgba(102, 126, 234, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-6px);
    }
    
    .workflow-number {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: var(--primary-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Comparison slider container */
    .comparison-container {
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 3rem 1rem;
        margin-top: 2rem;
    }
    
    .footer-brand {
        font-size: 1.2rem;
        font-weight: 600;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .footer-text {
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    
    /* Spinner override */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        padding: 1rem;
        border: 1px dashed rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: var(--primary-gradient) !important;
    }
    
    /* Expander styling - Fixed */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        overflow: hidden;
    }
    
    [data-testid="stExpander"] summary {
        padding: 1rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stExpander"] summary:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0 1rem 1rem 1rem !important;
    }
    
    /* Sidebar info box */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .sidebar-title {
        color: #667eea;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sidebar-content {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    .sidebar-content strong {
        color: rgba(255, 255, 255, 0.9);
    }
    
    .sidebar-list {
        margin: 0;
        padding-left: 1.2rem;
        color: rgba(255, 255, 255, 0.6);
    }
    
    .sidebar-list li {
        margin: 0.3rem 0;
    }
    
    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 16px;
        margin-bottom: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .sidebar-header-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-header-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
    }
    
    .sidebar-header-subtitle {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }
    
    /* GPU Badge in sidebar */
    .gpu-badge {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .gpu-badge-icon {
        font-size: 1.3rem;
    }
    
    .gpu-badge-text {
        color: #10b981;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Quick guide styling */
    .quick-guide {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .quick-guide-title {
        color: #667eea;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    
    .quick-guide ol {
        margin: 0;
        padding-left: 1.5rem;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
    }
    
    .quick-guide li {
        margin: 0.4rem 0;
    }
    
    .quick-guide li strong {
        color: #667eea;
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
        
        with st.spinner("Loading AI models..."):
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
        
        progress_bar = st.progress(0, text="Initializing pipeline...")
        
        progress_bar.progress(10, text="🔬 Stage 1: Applying DnCNN denoising...")
        denoised = enhancer.denoise(img_array)
        
        progress_bar.progress(50, text="✨ Stage 2: Applying ESRGAN enhancement...")
        enhanced = enhancer.enhance(denoised)
        
        progress_bar.progress(100, text="✅ Enhancement complete!")
        
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
    # Header with premium styling
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem;'>
            <h1 class='main-title'>🔬 Ultrasound Enhancement</h1>
            <p class='subtitle'>AI-Powered Medical Image Enhancement Pipeline</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Hero info section
    st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <p style='margin: 0; font-size: 1.15rem; line-height: 1.8; color: rgba(255,255,255,0.8);'>
                <strong style='color: #00d4ff;'>Advanced AI Pipeline</strong> combining 
                <span style='color: #667eea; font-weight: 600;'>DnCNN</span> denoising with 
                <span style='color: #b24bf3; font-weight: 600;'>ESRGAN</span> enhancement
                for superior ultrasound image quality
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
            <div class='sidebar-header'>
                <div class='sidebar-header-icon'>🔬</div>
                <div class='sidebar-header-title'>Ultrasound Enhancement</div>
                <div class='sidebar-header-subtitle'>AI-Powered Pipeline</div>
            </div>
        """, unsafe_allow_html=True)
        
        # About Section
        st.markdown("""
            <div class='sidebar-section'>
                <div class='sidebar-title'>📋 About This App</div>
                <div class='sidebar-content'>
                    <strong>Two-Stage AI Enhancement:</strong>
                    <br><br>
                    <strong style='color: #667eea;'>🔹 Stage 1: DnCNN</strong>
                    <ul class='sidebar-list'>
                        <li>Deep learning denoising</li>
                        <li>Preserves image details</li>
                        <li>Trained on paired data</li>
                    </ul>
                    <br>
                    <strong style='color: #b24bf3;'>🔹 Stage 2: ESRGAN</strong>
                    <ul class='sidebar-list'>
                        <li>Super-resolution GAN</li>
                        <li>Improves clarity</li>
                        <li>Medical imaging optimized</li>
                    </ul>
                    <br>
                    <strong>Supported Formats:</strong><br>
                    PNG, JPG, JPEG, BMP, TIFF
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # System Status Section
        st.markdown("""
            <div class='sidebar-section'>
                <div class='sidebar-title'>⚙️ System Status</div>
            </div>
        """, unsafe_allow_html=True)
        
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        device_icon = "🎮" if torch.cuda.is_available() else "💻"
        
        st.markdown(f"""
            <div class='gpu-badge'>
                <span class='gpu-badge-icon'>{device_icon}</span>
                <span class='gpu-badge-text'>{device}</span>
            </div>
        """, unsafe_allow_html=True)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.markdown(f"""
                <div style='color: rgba(255, 255, 255, 0.5); font-size: 0.8rem; margin-left: 0.5rem;'>
                    {gpu_name}
                </div>
            """, unsafe_allow_html=True)
        
        # Quick Guide Section
        st.markdown("""
            <div class='quick-guide'>
                <div class='quick-guide-title'>🎯 Quick Guide</div>
                <ol>
                    <li>Upload ultrasound image</li>
                    <li>Click <strong>Enhance</strong></li>
                    <li>Compare results</li>
                    <li>Download outputs</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

        # Quality Metrics Guide Section (New)
        st.markdown("""
            <div class='sidebar-section'>
                <div class='sidebar-title'>📈 Metrics Reference</div>
                <div class='sidebar-content'>
                    <div style='margin-bottom: 0.8rem;'>
                        <strong style='color: #667eea;'>PSNR</strong> (Quality)
                        <br>
                        Higher dB means less noise and better fidelity to the clean reference.
                    </div>
                    <div>
                        <strong style='color: #b24bf3;'>SSIM</strong> (Structure)
                        <br>
                        Shows how well the AI preserved the anatomical shapes and textures.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Load models
    enhancer = load_models()
    
    if enhancer is None:
        st.error("❌ Failed to load models. Please ensure the trained models are available.")
        st.stop()
    
    st.markdown("""
        <div class='status-badge ready' style='margin: 1rem 0;'>
            <span>✅</span>
            <span>AI Models Ready</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
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

        # Display original image in a glass card
        st.markdown("""
            <div class='glass-card'>
                <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                    <span style='font-size: 1.5rem;'>📥</span>
                    <h4 style='margin: 0; color: rgba(255,255,255,0.9);'>Original Image</h4>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        original_img = Image.open(uploaded_file)
        col_img, col_info = st.columns([2, 1])
        
        with col_img:
            st.image(original_img, width="stretch")
        
        with col_info:
            # Enhanced Image Info Panel with Premium Design
            st.markdown("""
                <div class='glass-card' style='padding: 1.5rem;'>
                    <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.25rem; border-bottom: 1px solid rgba(102, 126, 234, 0.2); padding-bottom: 1rem;'>
                        <span style='font-size: 2rem;'>📊</span>
                        <h3 style='margin: 0; background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.3rem;'>Image Info</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Calculate additional stats
            img_array_info = np.array(original_img)
            total_pixels = original_img.width * original_img.height
            aspect_ratio = original_img.width / original_img.height
            
            # File details
            st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.08); padding: 1rem; border-radius: 12px; margin-bottom: 0.75rem; border-left: 3px solid #667eea;'>
                    <div style='color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem;'>Filename</div>
                    <div style='color: rgba(255,255,255,0.95); font-size: 0.9rem; font-weight: 600; word-break: break-all;'>{}</div>
                </div>
            """.format(uploaded_file.name), unsafe_allow_html=True)
            
            # Dimensions
            st.markdown(f"""
                <div style='background: rgba(118, 75, 162, 0.08); padding: 1rem; border-radius: 12px; margin-bottom: 0.75rem; border-left: 3px solid #b24bf3;'>
                    <div style='color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.25rem;'>Dimensions</div>
                    <div style='color: rgba(255,255,255,0.95); font-size: 1.1rem; font-weight: 700;'>{original_img.width} × {original_img.height}</div>
                    <div style='color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.25rem;'>{total_pixels:,} pixels • {aspect_ratio:.2f}:1</div>
                </div>
            """, unsafe_allow_html=True)
            
            # File size and mode
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                    <div style='background: rgba(0, 212, 255, 0.08); padding: 0.75rem; border-radius: 10px; text-align: center;'>
                        <div style='color: rgba(255,255,255,0.5); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;'>File Size</div>
                        <div style='color: #00d4ff; font-size: 1.1rem; font-weight: 700; margin-top: 0.25rem;'>{uploaded_file.size / 1024:.1f} KB</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                mode_color = "#10b981" if original_img.mode == 'L' else "#f59e0b"
                st.markdown(f"""
                    <div style='background: rgba(16, 185, 129, 0.08); padding: 0.75rem; border-radius: 10px; text-align: center;'>
                        <div style='color: rgba(255,255,255,0.5); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;'>Mode</div>
                        <div style='color: {mode_color}; font-size: 1.1rem; font-weight: 700; margin-top: 0.25rem;'>{original_img.mode}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Histogram preview
            st.markdown("""
                <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);'>
                    <div style='color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;'>📈 Intensity Distribution</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Create simple histogram
            if img_array_info.ndim == 3:
                img_array_info = np.mean(img_array_info, axis=2)
            
            hist, bins = np.histogram(img_array_info.flatten(), bins=50, range=(0, 255))
            hist_normalized = hist / hist.max() * 100
            
            # Create ASCII-style histogram bars
            hist_html = "<div style='display: flex; align-items: flex-end; height: 60px; gap: 1px; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 8px;'>"
            for h in hist_normalized[::2]:  # Sample every other bar for space
                bar_height = max(5, int(h))
                hist_html += f"<div style='flex: 1; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); height: {bar_height}%; border-radius: 2px 2px 0 0; opacity: 0.8;'></div>"
            hist_html += "</div>"
            
            st.markdown(hist_html, unsafe_allow_html=True)
            
            # Stats summary
            mean_val = np.mean(img_array_info)
            std_val = np.std(img_array_info)
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.5);'>
                    <span>Mean: {mean_val:.1f}</span>
                    <span>Std: {std_val:.1f}</span>
                </div>
            """, unsafe_allow_html=True)
        
        # Enhanced process button
        st.markdown("<br>", unsafe_allow_html=True)
        clicked = st.button("🚀 ENHANCE IMAGE", type="primary", width="stretch")
        
        if clicked:
            with st.spinner("⚡ Processing..."):
                original, denoised, enhanced = process_image(enhancer, uploaded_file)
            
            if original is not None and denoised is not None and enhanced is not None:
                st.session_state.original = original
                st.session_state.denoised = denoised
                st.session_state.enhanced = enhanced
                st.session_state.processed_file_id = file_id
                st.session_state.has_results = True
                st.balloons()
                st.success("✅ Enhancement complete! View results below.")
                
        if st.session_state.get("processed_file_id") == file_id and st.session_state.get("has_results"):
            original = st.session_state.original
            denoised = st.session_state.denoised
            enhanced = st.session_state.enhanced
            
            st.markdown("---")
            
            # Results header
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h2>✨ Enhancement Results</h2>
                    <p style='color: rgba(255,255,255,0.6);'>Compare the original with AI-enhanced versions</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display results in enhanced tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Comparison", 
                "🔬 DnCNN Output", 
                "✨ ESRGAN Output",
                "📈 Metrics",
                "🎨 Studio"
            ])
            
            with tab1:
                st.markdown("### 🔍 Side-by-Side Comparison")
                st.markdown("<br>", unsafe_allow_html=True)
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.markdown("""
                        <div class='result-card'>
                            <div class='stage-pill' style='background: rgba(150,150,150,0.2); color: #999; border-color: rgba(150,150,150,0.3);'>
                                Original
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(original, width="stretch", clamp=True)
                
                with comp_col2:
                    st.markdown("""
                        <div class='result-card'>
                            <div class='stage-pill stage-1'>Stage 1 • DnCNN</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(denoised, width="stretch", clamp=True)
                
                with comp_col3:
                    st.markdown("""
                        <div class='result-card'>
                            <div class='stage-pill stage-2'>Stage 2 • ESRGAN</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(enhanced, width="stretch", clamp=True)
            
            with tab2:
                st.markdown("### 🔬 DnCNN Denoising Result")
                st.markdown("""
                    <div class='info-box'>
                        <strong style='color: #667eea;'>Stage 1 Complete</strong> — 
                        Noise removed while preserving structural details
                    </div>
                """, unsafe_allow_html=True)
                
                col_dn_img, col_dn_dl = st.columns([3, 1])
                
                with col_dn_img:
                    st.image(denoised, width="stretch", clamp=True, caption="DnCNN Denoised Output")
                
                with col_dn_dl:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    dncnn_bytes = image_to_bytes(denoised, title="DnCNN Denoised")
                    st.download_button(
                        label="⬇️ Download",
                        data=dncnn_bytes,
                        file_name="dncnn_denoised.png",
                        mime="image/png",
                        width="stretch"
                    )
            
            with tab3:
                st.markdown("### ✨ ESRGAN Enhancement Result")
                st.markdown("""
                    <div class='info-box'>
                        <strong style='color: #b24bf3;'>Stage 2 Complete</strong> — 
                        Image refined and enhanced for maximum clarity
                    </div>
                """, unsafe_allow_html=True)
                
                col_es_img, col_es_dl = st.columns([3, 1])
                
                with col_es_img:
                    st.image(enhanced, width="stretch", clamp=True, caption="ESRGAN Enhanced Output")
                
                with col_es_dl:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    esrgan_bytes = image_to_bytes(enhanced, title="ESRGAN Enhanced")
                    st.download_button(
                        label="⬇️ Download",
                        data=esrgan_bytes,
                        file_name="esrgan_enhanced.png",
                        mime="image/png",
                        width="stretch"
                    )
            
            with tab4:
                st.markdown("""
                    <div style='text-align: center; margin-bottom: 2rem;'>
                        <h2 style='background: linear-gradient(135deg, #667eea 0%, #00d4ff 100%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;'>📈 Image Quality Metrics</h2>
                        <p style='color: rgba(255,255,255,0.6);'>Quantitative analysis of enhancement performance</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Calculate metrics
                psnr_dn, ssim_dn = calculate_metrics(original, denoised)
                psnr_es, ssim_es = calculate_metrics(original, enhanced)
                
                # Metrics comparison cards
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.markdown("""
                        <div class='glass-card' style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(102, 126, 234, 0.05) 100%); border: 2px solid rgba(102, 126, 234, 0.3);'>
                            <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;'>
                                <span style='font-size: 1.8rem;'>🔬</span>
                                <div>
                                    <h3 style='margin: 0; color: #667eea; font-size: 1.2rem;'>DnCNN Stage</h3>
                                    <p style='margin: 0; color: rgba(255,255,255,0.5); font-size: 0.85rem;'>Denoising Performance</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if psnr_dn is not None:
                        # PSNR Gauge
                        psnr_percentage = min(100, (psnr_dn / 50) * 100)  # Scale to 50dB max
                        psnr_color = "#10b981" if psnr_dn > 30 else "#f59e0b" if psnr_dn > 20 else "#ef4444"
                        
                        st.markdown(f"""
                            <div style='background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;'>
                                <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;'>
                                    <span style='color: rgba(255,255,255,0.7); font-size: 0.9rem; font-weight: 600;'>PSNR (Peak Signal-to-Noise Ratio)</span>
                                    <span style='color: {psnr_color}; font-size: 1.8rem; font-weight: 800;'>{psnr_dn:.2f} dB</span>
                                </div>
                                <div style='background: rgba(255,255,255,0.1); height: 12px; border-radius: 10px; overflow: hidden;'>
                                    <div style='background: linear-gradient(90deg, {psnr_color}, {psnr_color}); height: 100%; width: {psnr_percentage}%; border-radius: 10px; transition: width 0.5s ease;'></div>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);'>
                                    <span>0 dB</span>
                                    <span>50 dB</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # SSIM Gauge
                        ssim_percentage = ssim_dn * 100
                        ssim_color = "#10b981" if ssim_dn > 0.9 else "#f59e0b" if ssim_dn > 0.7 else "#ef4444"
                        
                        st.markdown(f"""
                            <div style='background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 16px;'>
                                <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;'>
                                    <span style='color: rgba(255,255,255,0.7); font-size: 0.9rem; font-weight: 600;'>SSIM (Structural Similarity)</span>
                                    <span style='color: {ssim_color}; font-size: 1.8rem; font-weight: 800;'>{ssim_dn:.4f}</span>
                                </div>
                                <div style='background: rgba(255,255,255,0.1); height: 12px; border-radius: 10px; overflow: hidden;'>
                                    <div style='background: linear-gradient(90deg, {ssim_color}, {ssim_color}); height: 100%; width: {ssim_percentage}%; border-radius: 10px; transition: width 0.5s ease;'></div>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);'>
                                    <span>0.0</span>
                                    <span>1.0</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown("""
                        <div class='glass-card' style='background: linear-gradient(135deg, rgba(118, 75, 162, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%); border: 2px solid rgba(118, 75, 162, 0.3);'>
                            <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;'>
                                <span style='font-size: 1.8rem;'>✨</span>
                                <div>
                                    <h3 style='margin: 0; color: #b24bf3; font-size: 1.2rem;'>ESRGAN Stage</h3>
                                    <p style='margin: 0; color: rgba(255,255,255,0.5); font-size: 0.85rem;'>Enhancement Performance</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if psnr_es is not None:
                        # PSNR Gauge
                        psnr_percentage = min(100, (psnr_es / 50) * 100)
                        psnr_color = "#10b981" if psnr_es > 30 else "#f59e0b" if psnr_es > 20 else "#ef4444"
                        
                        st.markdown(f"""
                            <div style='background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;'>
                                <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;'>
                                    <span style='color: rgba(255,255,255,0.7); font-size: 0.9rem; font-weight: 600;'>PSNR (Peak Signal-to-Noise Ratio)</span>
                                    <span style='color: {psnr_color}; font-size: 1.8rem; font-weight: 800;'>{psnr_es:.2f} dB</span>
                                </div>
                                <div style='background: rgba(255,255,255,0.1); height: 12px; border-radius: 10px; overflow: hidden;'>
                                    <div style='background: linear-gradient(90deg, {psnr_color}, {psnr_color}); height: 100%; width: {psnr_percentage}%; border-radius: 10px; transition: width 0.5s ease;'></div>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);'>
                                    <span>0 dB</span>
                                    <span>50 dB</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # SSIM Gauge
                        ssim_percentage = ssim_es * 100
                        ssim_color = "#10b981" if ssim_es > 0.9 else "#f59e0b" if ssim_es > 0.7 else "#ef4444"
                        
                        st.markdown(f"""
                            <div style='background: rgba(0,0,0,0.3); padding: 1.5rem; border-radius: 16px;'>
                                <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;'>
                                    <span style='color: rgba(255,255,255,0.7); font-size: 0.9rem; font-weight: 600;'>SSIM (Structural Similarity)</span>
                                    <span style='color: {ssim_color}; font-size: 1.8rem; font-weight: 800;'>{ssim_es:.4f}</span>
                                </div>
                                <div style='background: rgba(255,255,255,0.1); height: 12px; border-radius: 10px; overflow: hidden;'>
                                    <div style='background: linear-gradient(90deg, {ssim_color}, {ssim_color}); height: 100%; width: {ssim_percentage}%; border-radius: 10px; transition: width 0.5s ease;'>
                                </div>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.4);'>
                                    <span>0.0</span>
                                    <span>1.0</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Comparison Summary
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='glass-card' style='background: rgba(0, 212, 255, 0.05); border: 1px solid rgba(0, 212, 255, 0.2);'>
                        <h3 style='color: #00d4ff; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;'>
                            <span>📊</span> Performance Comparison
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if psnr_dn is not None and psnr_es is not None:
                    # Comparison table
                    psnr_improvement = psnr_es - psnr_dn
                    ssim_improvement = ssim_es - ssim_dn
                    
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    with comp_col1:
                        st.markdown("""
                            <div style='background: rgba(255,255,255,0.03); padding: 1.25rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.08);'>
                                <div style='color: rgba(255,255,255,0.5); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>Metric</div>
                                <div style='color: rgba(255,255,255,0.9); font-weight: 700; font-size: 1rem;'>PSNR</div>
                                <div style='color: rgba(255,255,255,0.9); font-weight: 700; font-size: 1rem; margin-top: 1rem;'>SSIM</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with comp_col2:
                        st.markdown(f"""
                            <div style='background: rgba(102, 126, 234, 0.1); padding: 1.25rem; border-radius: 12px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.3);'>
                                <div style='color: #667eea; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>DnCNN</div>
                                <div style='color: rgba(255,255,255,0.95); font-weight: 700; font-size: 1.1rem;'>{psnr_dn:.2f} dB</div>
                                <div style='color: rgba(255,255,255,0.95); font-weight: 700; font-size: 1.1rem; margin-top: 1rem;'>{ssim_dn:.4f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with comp_col3:
                        psnr_arrow = "↗️" if psnr_improvement > 0 else "↘️" if psnr_improvement < 0 else "→"
                        ssim_arrow = "↗️" if ssim_improvement > 0 else "↘️" if ssim_improvement < 0 else "→"
                        psnr_color = "#10b981" if psnr_improvement > 0 else "#ef4444" if psnr_improvement < 0 else "#f59e0b"
                        ssim_color = "#10b981" if ssim_improvement > 0 else "#ef4444" if ssim_improvement < 0 else "#f59e0b"
                        
                        st.markdown(f"""
                            <div style='background: rgba(118, 75, 162, 0.1); padding: 1.25rem; border-radius: 12px; text-align: center; border: 1px solid rgba(118, 75, 162, 0.3);'>
                                <div style='color: #b24bf3; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>ESRGAN</div>
                                <div style='color: rgba(255,255,255,0.95); font-weight: 700; font-size: 1.1rem;'>{psnr_es:.2f} dB <span style='color: {psnr_color}; font-size: 0.9rem;'>{psnr_arrow}</span></div>
                                <div style='color: rgba(255,255,255,0.95); font-weight: 700; font-size: 1.1rem; margin-top: 1rem;'>{ssim_es:.4f} <span style='color: {ssim_color}; font-size: 0.9rem;'>{ssim_arrow}</span></div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Histogram Comparison
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='glass-card'>
                        <h3 style='color: rgba(255,255,255,0.9); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;'>
                            <span>📊</span> Intensity Distribution Comparison
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                hist_col1, hist_col2, hist_col3 = st.columns(3)
                
                def create_histogram_viz(img_data, title, color_start, color_end):
                    hist, _ = np.histogram(img_data.flatten(), bins=50, range=(0, 1))
                    hist_norm = hist / hist.max() * 100
                    
                    hist_html = f"""
                        <div style='background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 12px;'>
                            <div style='color: rgba(255,255,255,0.7); font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; text-align: center;'>{title}</div>
                            <div style='display: flex; align-items: flex-end; height: 80px; gap: 1px; background: rgba(0,0,0,0.2); padding: 0.5rem; border-radius: 8px;'>
                    """
                    for h in hist_norm[::2]:
                        bar_height = max(3, int(h))
                        hist_html += f"<div style='flex: 1; background: linear-gradient(180deg, {color_start} 0%, {color_end} 100%); height: {bar_height}%; border-radius: 2px 2px 0 0; opacity: 0.9;'></div>"
                    hist_html += "</div></div>"
                    return hist_html
                
                with hist_col1:
                    st.markdown(create_histogram_viz(original, "Original", "#999", "#666"), unsafe_allow_html=True)
                
                with hist_col2:
                    st.markdown(create_histogram_viz(denoised, "DnCNN", "#667eea", "#764ba2"), unsafe_allow_html=True)
                
                with hist_col3:
                    st.markdown(create_histogram_viz(enhanced, "ESRGAN", "#b24bf3", "#f093fb"), unsafe_allow_html=True)
                
                # Detailed Metrics Explanation
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("""
                    <div class='info-box' style='border-left: 4px solid #00d4ff;'>
                        <h3 style='margin-top: 0; color: #00d4ff; font-size: 1.3rem; display: flex; align-items: center; gap: 0.5rem;'>
                            <span>�</span> Deep Dive into Metrics
                        </h3>
                        <p style='color: rgba(255,255,255,0.8); line-height: 1.6;'>
                            To ensure medical accuracy, we use two industry-standard mathematical models to compare the AI results against the original and reference images.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.markdown("""
                        <div style='background: rgba(102, 126, 234, 0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2); height: 100%;'>
                            <h4 style='color: #667eea; margin-top: 0;'>📊 PSNR</h4>
                            <p style='font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 1rem;'>
                                <strong>Peak Signal-to-Noise Ratio</strong> measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
                            </p>
                            <div style='background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;'>
                                <div style='font-size: 0.85rem; margin-bottom: 0.5rem;'><strong>What to look for:</strong></div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                                    <span style='color: #10b981;'>●</span> <strong>30+ dB</strong>: High Quality
                                </div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                                    <span style='color: #f59e0b;'>●</span> <strong>20-30 dB</strong>: Acceptable
                                </div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px;'>
                                    <span style='color: #ef4444;'>●</span> <strong><20 dB</strong>: Significant Noise
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with exp_col2:
                    st.markdown("""
                        <div style='background: rgba(118, 75, 162, 0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(118, 75, 162, 0.2); height: 100%;'>
                            <h4 style='color: #b24bf3; margin-top: 0;'>🔍 SSIM</h4>
                            <p style='font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 1rem;'>
                                <strong>Structural Similarity Index</strong> is a perceptual metric that quantifies image quality degradation caused by processing.
                            </p>
                            <div style='background: rgba(0,0,0,0.2); padding: 1rem; border-radius: 8px;'>
                                <div style='font-size: 0.85rem; margin-bottom: 0.5rem;'><strong>What to look for:</strong></div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                                    <span style='color: #10b981;'>●</span> <strong>0.95 - 1.0</strong>: Perfect Preservation
                                </div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px; margin-bottom: 5px;'>
                                    <span style='color: #f59e0b;'>●</span> <strong>0.80 - 0.95</strong>: Minor Changes
                                </div>
                                <div style='font-size: 0.8rem; display: flex; align-items: center; gap: 10px;'>
                                    <span style='color: #ef4444;'>●</span> <strong><0.80</strong>: Structural Distortion
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander("📝 Implementation Details", expanded=False):
                    st.markdown("""
                    <div style='background: rgba(255,255,255,0.02); padding: 1rem; border-radius: 12px;'>
                        <p style='font-size: 0.9rem; color: rgba(255,255,255,0.7);'>
                            The metrics are calculated in real-time using NumPy by comparing the <strong>Normalized Greayscale Intensity</strong> of each pixel. 
                            The original noisy image is used as the baseline.
                        </p>
                        <code style='color: #00d4ff; font-size: 0.85rem;'>PSNR = 20 * log10(MAX / sqrt(MSE))</code>
                        <br>
                        <code style='color: #b24bf3; font-size: 0.85rem;'>SSIM = luminance * contrast * structure</code>
                    </div>
                    """, unsafe_allow_html=True)

            with tab5:
                st.markdown("### 🎨 Interactive Post-Processing")
                st.markdown("Fine-tune the ESRGAN output with real-time adjustments")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                filter_col1, filter_col2 = st.columns([1, 2])
                
                with filter_col1:
                    st.markdown("""
                        <div class='feature-card'>
                            <span class='feature-icon'>🎛️</span>
                            <h4>Adjustments</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)
                    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
                    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
                    
                    if st.button("🔄 Reset", width="stretch"):
                        st.rerun()
                    
                    st.markdown("---")
                    st.markdown("#### 📊 Live Metrics")
                    
                    # Apply filters
                    filtered_img = apply_filters(enhanced, contrast, brightness, sharpness)
                    
                    # Calculate metrics
                    psnr_filt, ssim_filt = calculate_metrics(original, filtered_img)
                    psnr_orig, ssim_orig = calculate_metrics(original, enhanced)
                    
                    psnr_delta = psnr_filt - (psnr_orig if psnr_orig else 0)
                    ssim_delta = ssim_filt - (ssim_orig if ssim_orig else 0)
                    
                    st.metric("PSNR", f"{psnr_filt:.2f} dB", delta=f"{psnr_delta:+.2f}")
                    st.metric("SSIM", f"{ssim_filt:.4f}", delta=f"{ssim_delta:+.4f}")
                    
                with filter_col2:
                    st.markdown("""
                        <div class='result-card'>
                            <h4 style='color: rgba(255,255,255,0.9);'>🖼️ Filtered Preview</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(filtered_img, width="stretch", clamp=True, caption="Adjusted Output")
                    
                    filt_bytes = image_to_bytes(filtered_img, title="Filtered Enhanced")
                    st.download_button(
                        label="⬇️ Download Filtered Image",
                        data=filt_bytes,
                        file_name="filtered_enhanced.png",
                        mime="image/png",
                        width="stretch"
                    )
            
            
            # Download all results section
            st.markdown("---")
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h3>📦 Download All Results</h3>
                </div>
            """, unsafe_allow_html=True)
            
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            
            with dl_col1:
                st.markdown("""
                    <div class='feature-card'>
                        <span class='feature-icon'>📥</span>
                        <h4>Original</h4>
                    </div>
                """, unsafe_allow_html=True)
                orig_bytes = image_to_bytes(original, title="Original")
                st.download_button(
                    label="⬇️ Download",
                    data=orig_bytes,
                    file_name="original.png",
                    mime="image/png",
                    width="stretch",
                    key="dl_original"
                )
            
            with dl_col2:
                st.markdown("""
                    <div class='feature-card'>
                        <span class='feature-icon'>🔬</span>
                        <h4>DnCNN Output</h4>
                    </div>
                """, unsafe_allow_html=True)
                dncnn_bytes = image_to_bytes(denoised, title="DnCNN Denoised")
                st.download_button(
                    label="⬇️ Download",
                    data=dncnn_bytes,
                    file_name="dncnn_output.png",
                    mime="image/png",
                    width="stretch",
                    key="dl_dncnn"
                )
            
            with dl_col3:
                st.markdown("""
                    <div class='feature-card'>
                        <span class='feature-icon'>✨</span>
                        <h4>ESRGAN Output</h4>
                    </div>
                """, unsafe_allow_html=True)
                esrgan_bytes = image_to_bytes(enhanced, title="ESRGAN Enhanced")
                st.download_button(
                    label="⬇️ Download",
                    data=esrgan_bytes,
                    file_name="esrgan_output.png",
                    mime="image/png",
                    width="stretch",
                    key="dl_esrgan"
                )
    
    else:
        # Enhanced upload prompt - Welcome Screen
        st.markdown("""
            <div class='upload-box'>
                <span style='font-size: 4rem; display: block; margin-bottom: 1rem;'>☁️</span>
                <h2 style='color: #667eea; margin-bottom: 0.5rem; font-size: 1.8rem;'>
                    Drop Your Ultrasound Image Here
                </h2>
                <p style='font-size: 1.1rem; color: rgba(255,255,255,0.6); margin-bottom: 1rem;'>
                    or click to browse your files
                </p>
                <p style='font-size: 0.9rem; color: rgba(255,255,255,0.4);'>
                    Supports PNG, JPG, JPEG, BMP, TIFF
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show workflow explanation
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <h2>🎯 How It Works</h2>
                <p style='color: rgba(255,255,255,0.6);'>Three simple steps to enhance your ultrasound images</p>
            </div>
        """, unsafe_allow_html=True)
        
        workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
        
        with workflow_col1:
            st.markdown("""
                <div class='workflow-step'>
                    <div class='workflow-number'>1</div>
                    <h3 style='margin-bottom: 0.5rem;'>Upload</h3>
                    <p style='color: rgba(255,255,255,0.6); font-size: 0.95rem;'>
                        Select your ultrasound image
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with workflow_col2:
            st.markdown("""
                <div class='workflow-step'>
                    <div class='workflow-number'>2</div>
                    <h3 style='margin-bottom: 0.5rem;'>Enhance</h3>
                    <p style='color: rgba(255,255,255,0.6); font-size: 0.95rem;'>
                        AI processes your image
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with workflow_col3:
            st.markdown("""
                <div class='workflow-step'>
                    <div class='workflow-number'>3</div>
                    <h3 style='margin-bottom: 0.5rem;'>Download</h3>
                    <p style='color: rgba(255,255,255,0.6); font-size: 0.95rem;'>
                        Get enhanced results
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0;'>
                <h2>⚡ Key Features</h2>
            </div>
        """, unsafe_allow_html=True)
        
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        
        with feat_col1:
            st.markdown("""
                <div class='feature-card'>
                    <span class='feature-icon'>🧠</span>
                    <h4>AI-Powered</h4>
                    <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>
                        Deep learning models
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
                <div class='feature-card'>
                    <span class='feature-icon'>⚡</span>
                    <h4>Fast Processing</h4>
                    <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>
                        GPU accelerated
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
                <div class='feature-card'>
                    <span class='feature-icon'>📊</span>
                    <h4>Quality Metrics</h4>
                    <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>
                        PSNR & SSIM scores
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with feat_col4:
            st.markdown("""
                <div class='feature-card'>
                    <span class='feature-icon'>🎨</span>
                    <h4>Fine-Tuning</h4>
                    <p style='color: rgba(255,255,255,0.5); font-size: 0.85rem;'>
                        Interactive adjustments
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
        <div class='footer'>
            <p class='footer-brand'>
                ⚡ Powered by DnCNN + ESRGAN Hybrid AI Pipeline
            </p>
            <p class='footer-text'>
                Built with ❤️ using Streamlit • Advanced Medical Image Enhancement
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
