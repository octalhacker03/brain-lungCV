import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
from model import load_trained_model

# --- CONFIGURATION ---
st.set_page_config(
    page_title="LUNG.AI | Clinical Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM THEME & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Glassmorphism */
    .main {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    
    /* Custom Card Style */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        border: 1px solid #00d2ff;
        transform: translateY(-2px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Professional Header */
    .header-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Result Display */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
    }
    .severity-mild { background: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; color: #00ff80; }
    .severity-moderate { background: rgba(255, 191, 0, 0.1); border: 1px solid #ffbf00; color: #ffbf00; }
    .severity-severe { background: rgba(255, 69, 58, 0.1); border: 1px solid #ff453a; color: #ff453a; }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        padding: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL CACHING ---
@st.cache_resource
def get_cached_model():
    # Force CPU as per guidelines
    device = "cpu"
    return load_trained_model("lung_model.pth", device)

model = get_cached_model()

# --- UTILS ---
def get_severity(percent):
    if percent < 2: return "Normal", "severity-mild"
    if percent < 8: return "Mild", "severity-mild"
    if percent < 20: return "Moderate", "severity-moderate"
    return "Acute/Severe", "severity-severe"

def process_image(uploaded_file, threshold=0.3):
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # Preprocessing
    input_size = (256, 256)
    orig_h, orig_w = image.shape
    img_resized = cv2.resize(image, input_size) / 255.0
    
    # Inference
    tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        pred = model(tensor)
    
    # Post-processing
    pred_mask = (torch.sigmoid(pred[0]).numpy() > threshold).astype(np.uint8).squeeze()
    
    # Calculate Infection %
    infection_percent = (pred_mask.sum() / pred_mask.size) * 100
    
    # Resize mask back to original size for high-quality overlay
    mask_full = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create Blended Overlay
    # Gray -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a red heatmask
    red_mask = np.zeros_like(image_rgb)
    red_mask[:, :, 0] = mask_full * 255 # Red channel
    
    # Blend: Overlay = 0.7 * Org + 0.3 * RedMask
    overlay = cv2.addWeighted(image_rgb, 0.7, red_mask, 0.3, 0)
    
    return {
        "original": image,
        "mask": mask_full,
        "overlay": overlay,
        "percent": infection_percent
    }

# --- SIDEBAR & HEADER ---
with st.sidebar:
    # Configuration
    detection_threshold = 0.3 # Balanced clinical threshold
    
    st.markdown("### 📂 Data Source")
    uploaded_files = st.file_uploader("Upload CT Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    st.divider()
    st.info("⚠️ This tool is for research assistance and should not replace professional medical diagnosis.")

# Header
st.markdown("""
<div class='header-container'>
    <span style='font-size: 3rem;'>🫁</span>
    <div>
        <div class='brand-title'>LUNG.AI ANALYTICS</div>
        <div style='color: rgba(255,255,255,0.6); font-weight: 500;'>Advanced Pulmonary Infection Segmentation Engine</div>
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
            
            # Run Inference
            res = process_image(file, detection_threshold)
            
            # Severity info
            severity, css_class = get_severity(res['percent'])
            
            # Display Container
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 1.2])
                
                with col1:
                    st.image(res['original'], caption="Raw Scanner Data", use_container_width=True)
                
                with col2:
                    st.image(res['overlay'], caption="AI Diagnostic Overlay", use_container_width=True)
                
                with col3:
                    st.markdown(f"**Case ID:** `{file.name}`")
                    st.metric("Infection Volume", f"{res['percent']:.2f}%")
                    st.markdown(f"""
                        <div class='result-box {css_class}'>
                            <strong>Diagnostic Status:</strong> {severity}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if res['percent'] > 0:
                        st.success("Segmentation Complete")
                    else:
                        st.info("No Significant Infection Detected")
                
                st.divider()
                
            results_list.append({
                "Filename": file.name,
                "Infection %": round(res['percent'], 2),
                "Severity": severity
            })
            
            progress_bar.progress((i + 1) / total_files)
            
        status_text.text("✅ Analysis Complete.")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    with tab2:
        if results_list:
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            c1, c2, c3 = st.columns(3)
            c1.metric("Average Infection", f"{df['Infection %'].mean():.2f}%")
            c2.metric("Max Severity Case", df.loc[df['Infection %'].idxmax(), 'Filename'])
            c3.metric("Acute Cases Found", len(df[df['Severity'] == 'Acute/Severe']))
            
            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Diagnostic Report (CSV)",
                data=csv,
                file_name="lung_ai_report.csv",
                mime="text/csv",
            )

else:
    # Welcome State
    st.markdown("""
    <div style='text-align: center; padding: 100px 20px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);'>
        <h2 style='color: #00d2ff;'>Ready for Analysis</h2>
        <p style='color: rgba(255,255,255,0.6);'>Please upload CT scans in the sidebar to begin automated pulmonary segmentation.</p>
        <div style='margin-top: 20px;'>
            <span style='padding: 8px 15px; background: rgba(0,210,255,0.1); border-radius: 20px; color: #00d2ff; font-size: 0.8rem; margin: 0 5px;'>Single Image Support</span>
            <span style='padding: 8px 15px; background: rgba(0,210,255,0.1); border-radius: 20px; color: #00d2ff; font-size: 0.8rem; margin: 0 5px;'>Batch Processing</span>
            <span style='padding: 8px 15px; background: rgba(0,210,255,0.1); border-radius: 20px; color: #00d2ff; font-size: 0.8rem; margin: 0 5px;'>CSV Export</span>
        </div>
    </div>
    """, unsafe_allow_html=True)