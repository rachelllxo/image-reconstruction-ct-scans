import streamlit as st
import torch
import cv2
import numpy as np
import os
try:
    from model import MCDropoutRecon
    from utils import simulate_low_dose
    from trust import compute_trust_map
except ImportError as e:
    st.error(f"Missing a required file: {e}. Ensure model.py, utils.py, and trust.py are in this folder.")
    st.stop()
#Model initialization 
@st.cache_resource
def load_model():
    #ensuring map weights correctly
    model = MCDropoutRecon()
    model.eval()
    return model

model = load_model()
#processing function 
def process_scan_trustscore(img_path):
    raw = cv2.imread(img_path, 0)   #loading the raw slice 
    if raw is None: return None, None, None, None
    raw = cv2.resize(raw, (256, 256)) 
    # Cleaning : Denoising 
    denoised = cv2.fastNlMeansDenoising(raw, None, 10, 7, 21)   # Filter strength (7), Template window (7), Search window (21)
    #enhancing the image without brightenning using CLAHE (Constrast limited adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    #using edge detection to show the weights 
    trust = cv2.Canny(enhanced, 100, 200) / 255.0
    return raw, raw, enhanced, trust
#User Interface 
st.set_page_config(page_title="CT Scan Reconstruction and Enhancement", layout="wide")
st.title("CT Scan Reconstruction and Enhancement")

uploaded_file = st.file_uploader("Upload the Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    temp_path = "temp_slice.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    raw, low_dose, recon, trust = process_scan_trustscore(temp_path)
    if raw is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input Image")
            st.image(low_dose, clamp=True, use_container_width=True)
        with col2:
            st.subheader("Reconstructed Image")
            st.image(recon, clamp=True, use_container_width=True)
        with col3:
            st.subheader("Trust Map")
            st.image(trust, clamp=True, use_container_width=True)
        st.divider()
        mean_trust = float(trust.mean())
        min_trust = float(trust.min())
    if os.path.exists(temp_path):
        os.remove(temp_path)