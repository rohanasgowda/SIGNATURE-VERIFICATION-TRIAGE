import streamlit as st
from PIL import Image
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, hog
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="SigVerify AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded" # Tries to keep sidebar open
)

# --- Professional Unified Color Theme (CSS) ---
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp {
        background-color: #F8FAFC; /* Slate-50 */
        color: #334155; /* Slate-700 Text */
    }
    
    /* 2. Sidebar Styling - Deep Midnight Blue */
    section[data-testid="stSidebar"] {
        background-color: #1E293B; /* Slate-800 */
        color: white !important;
    }
    /* Force all text in sidebar to be white/light */
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div {
        color: #F8FAFC !important;
    }
    
    /* 3. Header Styling */
    .title-text {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        font-size: 3rem;
        margin-top: 10px;
    }
    .subtitle-text {
        font-family: 'Segoe UI', sans-serif;
        color: #64748B;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 40px;
    }
    
    /* 4. Card Styling */
    .stCard, .stContainer, div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
    }
    
    /* 5. Buttons */
    .stButton>button {
        background-color: #2563EB; /* Blue-600 */
        color: white;
        border-radius: 6px;
        font-weight: 600;
        height: 3em;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    
    /* 6. Mobile Helper Box (Custom Class) */
    .mobile-hint {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 15px;
        border-radius: 8px;
        color: #1E40AF;
        text-align: center;
        margin-bottom: 20px;
    }

    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def preprocess_image(image_bytes, target_size=(1000, 500)):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Error: Could not read image.")
        return None
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_filtered = cv2.GaussianBlur(img_resized, (5, 5), 0)
    _, img_binary = cv2.threshold(img_filtered, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_binary

def extract_features(binary_image):
    P = 8
    R = 1
    lbp = local_binary_pattern(binary_image, P, R, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    hog_features = hog(binary_image, pixels_per_cell=(32, 32),
                       cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    final_feature_vector = np.hstack([lbp_hist, hog_features])
    return final_feature_vector

@st.cache_resource
def load_models(model_type):
    if "Generic" in model_type:
        scaler_path, pca_path, model_path = 'scaler.joblib', 'pca.joblib', 'model.joblib'
    elif "Custom" in model_type:
        scaler_path, pca_path, model_path = 'custom_scaler.joblib', 'custom_pca.joblib', 'custom_model.joblib'
    else: 
        scaler_path, pca_path, model_path = 'individual_scaler.joblib', 'individual_pca.joblib', 'individual_model.joblib'

    if not all(os.path.exists(p) for p in [scaler_path, pca_path, model_path]):
        st.toast(f"⚠️ Models for {model_type} not found!", icon="⚠️")
        return None
        
    try:
        return joblib.load(scaler_path), joblib.load(pca_path), joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# --- Main Layout ---

st.markdown('<p class="title-text">SigVerify AI 🛡️</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Forensic Signature Verification System</p>', unsafe_allow_html=True)

# --- SIDEBAR (Restored) ---
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.info("Configure your forensic engine below.")
    
    model_choice = st.selectbox(
        "🧠 AI Model Engine",
        ("Generic (CEDAR)", "Custom Group (12 Users)", "Individual (Single User)"),
    )
    
    st.markdown("---")
    st.write("### 📂 Evidence Upload")
    uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ℹ️ System Status")
    st.success(f"Online: {model_choice.split(' ')[0]}")

# --- MAIN LOGIC ---

models = load_models(model_choice)

if not uploaded_file:
    # --- LANDING PAGE WITH MOBILE HINT ---
    st.markdown("---")
    
    # This box only helps mobile users find the sidebar
    st.markdown("""
        <div class="mobile-hint">
            <strong>👈 On Mobile?</strong> Tap the arrow in the top-left corner to open the <strong>Control Panel</strong>.
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            st.markdown("""
            ### 👋 Ready for Analysis
            
            Please upload a signature image using the **Control Panel on the left**.
            
            **System Capabilities:**
            * **Pre-processing:** Automated noise removal.
            * **Biometrics:** LBP (Texture) & HOG (Geometry).
            * **AI Engine:** XGBoost Classification.
            """)
            st.info("Waiting for input file...")

else:
    # --- ANALYSIS DASHBOARD ---
    if models is None:
        st.error("❌ Model files missing. Please run training scripts.")
    else:
        scaler, pca, model = models
        
        col_img, col_result = st.columns([1, 1.2], gap="large")
        
        with col_img:
            st.markdown("### 🖼️ Evidence Preview")
            st.image(Image.open(uploaded_file), use_container_width=True)
            st.caption("Source: Uploaded File")
            
            analyze_btn = st.button("🔍 Run Forensic Verification", type="primary", use_container_width=True)

        with col_result:
            if analyze_btn:
                with st.spinner("🔄 Extracting biometric features..."):
                    image_bytes = uploaded_file.getvalue()
                    preprocessed_img = preprocess_image(image_bytes)
                    
                    if preprocessed_img is not None:
                        features = extract_features(preprocessed_img)
                        features = features.reshape(1, -1)
                        scaled_features = scaler.transform(features)
                        pca_features = pca.transform(scaled_features)
                        
                        prediction = model.predict(pca_features)[0]
                        try:
                            proba = model.predict_proba(pca_features)[0]
                            confidence = max(proba)
                        except:
                            confidence = 1.0

                        st.markdown("### 📊 Analysis Report")
                        with st.container():
                            if prediction == 0:
                                st.success("## ✅ GENUINE")
                                st.write("Matches known stylistic patterns.")
                                st.progress(float(confidence), text=f"Match Confidence: {confidence*100:.1f}%")
                            else:
                                st.error("## ❌ FORGED")
                                st.write("Deviations detected in stroke/texture.")
                                st.progress(float(confidence), text=f"Forgery Probability: {confidence*100:.1f}%")
                        
                        st.divider()
                        st.markdown("#### 🔬 Telemetry Data")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Features", features.shape[1])
                        c2.metric("Components", pca_features.shape[1])
                        c3.metric("Engine", "XGBoost")
            else:
                st.info("System Ready. Click 'Run Forensic Verification' to process.")