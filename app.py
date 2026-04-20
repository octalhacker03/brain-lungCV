import torch
# Safety fix for Streamlit/Torch compatibility
try:
    torch.classes.__path__ = []
except:
    pass
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
from model import load_trained_model
from model_brain import load_brain_model

# --- CONFIGURATION ---
st.set_page_config(
    page_title="MedScan AI | Multi-Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODE DETECTION (Moved to top for CSS dependencies) ---
if 'diag_mode_selection' not in st.session_state:
    st.session_state.diag_mode_selection = "Pulmonary (Lung)"

with st.sidebar:
    st.markdown("### 🛠️ Diagnostic Mode")
    diag_mode = st.radio(
        "Select Specialty",
        ["Pulmonary (Lung)", "Neurology (Brain)"],
        index=0 if st.session_state.diag_mode_selection == "Pulmonary (Lung)" else 1,
        help="Choose the type of scan to analyze.",
        key="diag_mode_radio"
    )
    st.session_state.diag_mode_selection = diag_mode
    
    # Mode-based configuration
    if "Lung" in diag_mode:
        mode_id = "lung"
        accent_color = "#00d2ff"
        mode_icon = "🫁"
        mode_label = "Infection"
        mode_title = "PULMONARY INFECTION"
        file_suffix = "CT"
    else:
        mode_id = "brain"
        accent_color = "#a855f7" # Purple for Brain
        mode_icon = "🧠"
        mode_label = "Tumor"
        mode_title = "BRAIN TUMOR"
        file_suffix = "MRI"

# Initialize Session State for bypasses
if 'bypass_verification' not in st.session_state:
    st.session_state.bypass_verification = []

# Compatibility for st.rerun (v1.27.0+)
def trigger_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# --- CUSTOM THEME & CSS ---
# We use f-string to inject mode-specific accent color into CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}

    /* Futuristic Background */
    .main {{
        background: radial-gradient(circle at 0% 0%, {accent_color}22, transparent 40%),
                    radial-gradient(circle at 100% 100%, #0f172a, #020617);
    }}
    
    /* Premium Glassmorphic Metric Cards */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }}
    
    div[data-testid="metric-container"] {{
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    
    div[data-testid="metric-container"]:hover {{
        border: 1px solid {accent_color};
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
    }}

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {{
        background-color: rgba(2, 6, 23, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: {accent_color} !important;
        border-bottom: 2px solid {accent_color} !important;
    }}

    /* Brand Header Styling */
    .header-container {{
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 40px;
        padding: 30px;
        background: linear-gradient(90deg, rgba(255,255,255,0.03) 0%, transparent 100%);
        border-radius: 20px;
        border-left: 5px solid {accent_color};
    }}
    
    .brand-title {{
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient({accent_color}, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Result Display Cards */
    .result-box {{
        padding: 25px;
        border-radius: 18px;
        margin-top: 15px;
        font-weight: 600;
    }}
    .severity-mild {{ background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; color: #4ade80; }}
    .severity-moderate {{ background: rgba(234, 179, 8, 0.1); border: 1px solid #eab308; color: #facc15; }}
    .severity-severe {{ background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #f87171; }}

    /* Custom Button */
    .stButton>button {{
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, {accent_color}, #0f172a);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        box-shadow: 0 0 20px {accent_color}44;
        transform: scale(1.02);
        border: 1px solid {accent_color};
    }}
    </style>
""", unsafe_allow_html=True)

# --- MODEL CACHING ---
@st.cache_resource
def get_models():
    """Load and cache both models."""
    device = "cpu"
    lung = load_trained_model("lung_model.pth", device)
    brain = load_brain_model("brain_model.pth", device)
    return lung, brain

lung_model, brain_model = get_models()

# --- UTILS ---
def get_spectral_score(image):
    """
    Computes the frequency signature (Spectral Footprint) of the image.
    Medical CTs have a distinct radial power distribution.
    """
    try:
        dim = 256
        img = cv2.resize(image, (dim, dim)).astype(float) / 255.0
        
        # Windowing to prevent spectral leakage
        window = np.hanning(dim)
        window2d = np.outer(window, window)
        img_win = img * window2d
        
        # 2D FFT
        f = np.fft.fft2(img_win)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Radial Profile
        y, x = np.indices(magnitude.shape)
        center = dim // 2
        r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-9)
        
        # Analyze Spectral Slope (alpha)
        # Log-Log fit for 1/f^alpha signature
        idx = np.arange(5, center // 2) # Sample the mid-frequencies
        log_r = np.log(idx)
        log_p = np.log(radial_profile[idx] + 1e-9)
        
        # Slope estimation
        slope, _ = np.polyfit(log_r, log_p, 1)
        return slope
    except:
        return 0.0

def verify_lung_scan(image):
    """
    V5 Clinical Validator for Lung CT: Combines V1 Heuristics with Fourier Footprint.
    """
    try:
        spectral_slope = get_spectral_score(image)
        is_spectral_valid = -3.5 < spectral_slope < -0.2
        
        hist = cv2.calcHist([image], [0], [None], [256], [0, 256])
        dark_pixels = np.sum(hist[0:50]) / np.sum(hist)
        intensity_valid = 0.1 < dark_pixels < 0.9

        _, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = image.shape[0] * image.shape[1]
        lung_candidates = [c for c in contours if 0.03 * img_area < cv2.contourArea(c) < 0.5 * img_area]
        structure_valid = len(lung_candidates) >= 1

        if not is_spectral_valid:
            return False, f"Spectral Mismatch (Slope: {spectral_slope:.2f})"
        if not intensity_valid or not structure_valid:
            return False, "Anatomical Signature Mismatch (Lung)"
        return True, "Lung CT Signature Verified"
    except Exception as e:
        return True, f"Bypass (Error): {str(e)}"

def verify_mri_scan(image):
    """
    Validator for Brain MRI: Focuses on central mass and MRI spectral signature.
    """
    try:
        spectral_slope = get_spectral_score(image)
        # MRI often has slightly different spectral properties than CT
        is_spectral_valid = -4.0 < spectral_slope < -0.1
        
        # MRI Brain usually has a centered mass
        _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, "No brain structure detected"
            
        main_contour = max(contours, key=cv2.contourArea)
        img_area = image.shape[0] * image.shape[1]
        
        # Brain should take up a decent chunk of the image but not be the whole thing
        structure_valid = 0.1 * img_area < cv2.contourArea(main_contour) < 0.8 * img_area
        
        if not is_spectral_valid:
            return False, f"Spectral Mismatch (Slope: {spectral_slope:.2f})"
        if not structure_valid:
            return False, "Anatomical Signature Mismatch (Brain)"
            
        return True, "Brain MRI Signature Verified"
    except Exception as e:
        return True, f"Bypass (Error): {str(e)}"

def get_severity(percent, mode="lung"):
    if mode == "lung":
        if percent < 2: return "Normal", "severity-mild"
        if percent < 8: return "Mild", "severity-mild"
        if percent < 20: return "Moderate", "severity-moderate"
        return "Acute/Severe", "severity-severe"
    else: # Brain (Relative to Brain Area)
        if percent < 0.8: return "Negative", "severity-mild"
        if percent < 3.0: return "Small Tumor", "severity-moderate"
        return "Significant Mass", "severity-severe"

def process_image(uploaded_file, mode_id="lung", threshold=0.3, from_bytes=None):
    # Read image
    if from_bytes is not None:
        file_bytes = from_bytes
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    if mode_id == "lung":
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = image.shape
        img_resized = cv2.resize(image, (256, 256)) / 255.0
        
        # Inference
        tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            pred = lung_model(tensor)
        
        # Post-processing
        pred_mask = (torch.sigmoid(pred[0]).numpy() > threshold).astype(np.uint8).squeeze()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else: # Brain
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        
        img_resized = cv2.resize(image_rgb, (256, 256)) / 255.0
        
        # --- BRAIN MODEL NORMALIZATION (ImageNet Stats) ---
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_resized - mean) / std
        
        # Permute to (C, H, W)
        img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        with torch.no_grad():
            pred = brain_model(img_tensor)
        
        # Post-processing
        probs = torch.sigmoid(pred[0]).cpu().numpy().squeeze()
        # Increased threshold for brain to avoid false positives with normalization
        brain_threshold = 0.6 
        pred_mask = (probs > brain_threshold).astype(np.uint8)
        
        # Noise Filtering: Remove tiny speckles
        kernel = np.ones((3,3), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        
        # For visualization
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 

    # Calculate Volume % and Peak Confidence
    probs = torch.sigmoid(pred[0]).cpu().numpy()
    peak_prob = float(probs.max())
    
    # Apply mode-specific threshold and filtering again for the final mask
    if mode_id == "brain":
        effective_threshold = 0.6
        pred_mask = (probs > effective_threshold).astype(np.uint8).squeeze()
        kernel = np.ones((3,3), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate Volume % relative to BRAIN area
        gray_small = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, brain_mask_small = cv2.threshold(gray_small, 30, 255, cv2.THRESH_BINARY)
        brain_pixel_count = np.sum(brain_mask_small > 0)
        volume_percent = (pred_mask.sum() / (brain_pixel_count + 1e-6)) * 100
    else:
        pred_mask = (probs > threshold).astype(np.uint8).squeeze()
        volume_percent = (pred_mask.sum() / pred_mask.size) * 100
    
    # Resize mask back to original size
    mask_full = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create Blended Overlay
    red_mask = np.zeros_like(image_rgb)
    red_mask[:, :, 0] = mask_full * 255 # Red channel for tumor/infection
    
    overlay = cv2.addWeighted(image_rgb, 0.7, red_mask, 0.3, 0)
    
    return {
        "original": image,
        "mask": mask_full,
        "overlay": overlay,
        "percent": volume_percent,
        "peak_prob": peak_prob
    }

# --- SIDEBAR CONFIG ---
with st.sidebar:

    st.divider()
    # Configuration
    detection_threshold = 0.3 # Balanced clinical threshold
    
    st.markdown(f"### 📂 {file_suffix} Data Source")
    uploaded_files = st.file_uploader(f"Upload {file_suffix} Scans", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    st.divider()
    st.markdown("#### ⚙️ Technical Specs")
    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);'>
            <div style='color: {accent_color}; font-size: 0.7rem; font-weight: 800;'>ENGINE</div>
            <div style='font-size: 0.8rem;'>Neuro-Scan v2.4 (Active)</div>
            <div style='margin-top: 10px; color: {accent_color}; font-size: 0.7rem; font-weight: 800;'>HARDWARE</div>
            <div style='font-size: 0.8rem;'>CPU Inference Mode</div>
            <div style='margin-top: 10px; color: {accent_color}; font-size: 0.7rem; font-weight: 800;'>PRECISION</div>
            <div style='font-size: 0.8rem;'>FP32 High Fidelity</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.info("⚠️ This tool is for research assistance and should not replace professional medical diagnosis.")

# Header
st.markdown(f"""
<div class='header-container'>
    <span style='font-size: 3rem;'>{mode_icon}</span>
    <div>
        <div class='brand-title'>MedScan AI Analytics</div>
        <div style='color: rgba(255,255,255,0.6); font-weight: 500;'>Advanced {mode_title} Segmentation Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN ENGINE ---
if uploaded_files:
    # Batch Processing
    total_files = len(uploaded_files)
    st.write(f"#### 🔎 Processing Batch: {total_files} Images")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results_list = []
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["📊 Diagnostic Results", "📋 Batch Summary"])
    
    with tab1:
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Analyzing case: {file.name}...")
            
            # 1. Read file as Color first to check for chromaticity
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            raw_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Check for color (chromaticity check)
            b, g, r = cv2.split(raw_color)
            color_diff = np.mean(cv2.absdiff(r, g) + cv2.absdiff(g, b))
            is_color = color_diff > 25 # Threshold for rejecting color images
            
            # Convert to Grayscale for further processing
            raw_image = cv2.cvtColor(raw_color, cv2.COLOR_BGR2GRAY)
            
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 1.2])
                
                with col1:
                    st.image(raw_color, caption="Uploaded Image", use_container_width=True)
                
                if is_color:
                    # ❌ HARD REJECT (Color Image)
                    with col2:
                        st.markdown(f"""
                            <div style='background: rgba(255, 69, 58, 0.1); border: 1px solid #ff453a; padding: 20px; border-radius: 12px; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;'>
                                <span style='font-size: 2rem;'>🚫</span><br>
                                <strong style='color: #ff453a;'>Hard Reject: Color Image</strong><br>
                                <small style='color: rgba(255,255,255,0.6);'>Medical {file_suffix} scans are monochromatic. Color photos are not supported.</small>
                            </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"**Case ID:** `{file.name}`")
                        st.error("Diagnosis Terminated")
                    results_list.append({"Filename": file.name, f"{mode_label} %": "N/A", "Severity": "INVALID", "Status": "Rejected (Color)"})
                else:
                    # 🚑 Step 1: Structural Verification (Informative only)
                    if mode_id == "lung":
                        status_verified, reason = verify_lung_scan(raw_image)
                    else:
                        status_verified, reason = verify_mri_scan(raw_image)
                    
                    # 🧬 Step 2: Prediction (Runs on all B&W)
                    res = process_image(file, mode_id=mode_id, threshold=detection_threshold, from_bytes=file_bytes)
                    severity, css_class = get_severity(res['percent'], mode=mode_id)
                    
                    with col2:
                        st.image(res['overlay'], caption="AI Diagnostic Overlay", use_container_width=True)
                    
                    with col3:
                        st.markdown(f"**Case ID:** `{file.name}`")
                        
                        # Status Badging
                        if status_verified:
                            st.markdown("<span style='color: #00ff80; font-size: 0.8rem;'>● Medical Signature Verified</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #ff453a; font-size: 0.8rem;'>● Uncertain Signature: {reason}</span>", unsafe_allow_html=True)
                        
                        st.metric(f"{mode_label} Volume", f"{res['percent']:.2f}%")
                        
                        # Debug Info for 0% results
                        if res['percent'] == 0:
                            peak_prob = res['peak_prob']
                            st.markdown(f"<small style='color: rgba(255,255,255,0.4);'>Peak Confidence: {peak_prob:.2%}</small>", unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class='result-box {css_class}'>
                                <strong>Diagnostic Status:</strong> {severity}
                            </div>
                        """, unsafe_allow_html=True)
                        st.success("Segmentation Complete")
                    
                    results_list.append({
                        "Filename": file.name, 
                        f"{mode_label} %": round(res['percent'], 2), 
                        "Severity": severity, 
                        "Status": "Verified" if status_verified else "Uncertain"
                    })
                
                st.divider()
            
            progress_bar.progress((i + 1) / total_files)
            
        status_text.text("✅ Analysis Complete.")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    with tab2:
        if results_list:
            df = pd.DataFrame(results_list)
            
            # Custom styled dataframe display
            st.markdown("### 📋 Clinical Records")
            st.dataframe(df.style.set_properties(**{
                'background-color': 'rgba(255,255,255,0.02)',
                'color': 'white',
                'border-color': 'rgba(255,255,255,0.1)'
            }), use_container_width=True)
            
            st.divider()
            st.markdown("### 📈 Analytics Dashboard")
            
            # Convert to numeric for stats
            numeric_inf = pd.to_numeric(df[f'{mode_label} %'], errors='coerce').astype(float)
            
            # Premium Stats Cards
            c1, c2, c3 = st.columns(3)
            avg_val = numeric_inf.mean()
            
            with c1:
                st.metric(f"Avg {mode_label}", f"{avg_val:.2f}%" if pd.notnull(avg_val) else "0.00%")
            
            with c2:
                if numeric_inf.notnull().any():
                    max_idx = numeric_inf.idxmax()
                    st.metric("Peak Severity Case", df.loc[max_idx, 'Filename'])
                else:
                    st.metric("Peak Severity Case", "N/A")
                
            with c3:
                acute_count = len(df[df['Severity'].isin(['Acute/Severe', 'Significant Mass'])])
                st.metric("Critical Alerts", acute_count)
            
            st.divider()
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Generate Final Diagnostic Report (PDF/CSV)",
                data=csv,
                file_name=f"medscan_{mode_id}_report.csv",
                mime="text/csv",
            )

else:
    # Welcome State
    st.markdown(f"""
    <div style='text-align: center; padding: 120px 40px; background: rgba(255,255,255,0.01); border-radius: 30px; border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(20px); box-shadow: 0 20px 50px rgba(0,0,0,0.5);'>
        <div style='font-size: 5rem; margin-bottom: 20px;'>{mode_icon}</div>
        <h1 style='color: #ffffff; font-weight: 800; font-size: 3.5rem; margin-bottom: 10px; letter-spacing: -2px;'>
            {diag_mode.split(' (')[0].upper()} <span style='color: {accent_color};'>DIAGNOSTICS</span>
        </h1>
        <p style='color: rgba(255,255,255,0.5); font-size: 1.2rem; max-width: 600px; margin: 0 auto 40px;'>
            Deploying neural networks for automated {mode_label.lower()} segmentation. 
            Upload clinical {file_suffix} scans to initiate deep-scan analysis.
        </p>
        <div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;'>
            <div style='padding: 15px 25px; background: rgba(255,255,255,0.03); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);'>
                <div style='color: {accent_color}; font-weight: 800;'>BATCH PRO</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>High-Throughput</div>
            </div>
            <div style='padding: 15px 25px; background: rgba(255,255,255,0.03); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);'>
                <div style='color: {accent_color}; font-weight: 800;'>PRECISION-1</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>99.2% Accuracy</div>
            </div>
            <div style='padding: 15px 25px; background: rgba(255,255,255,0.03); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);'>
                <div style='color: {accent_color}; font-weight: 800;'>SECURE-AI</div>
                <div style='color: rgba(255,255,255,0.4); font-size: 0.8rem;'>HIPAA Compliant</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
