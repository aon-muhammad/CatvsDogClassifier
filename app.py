import os
import random
import numpy as np
from PIL import Image
import streamlit as st
import utils

# Set Page Config with clean wide layout (Sidebar Collapsed / No Sidebar needed)
st.set_page_config(
    page_title="PawDetective - Anthropomorphic Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Anthropomorphic Custom CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Quicksand:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Fredoka', 'Quicksand', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #FFFDE7 0%, #FFF8E1 50%, #F1F8E9 100%);
        color: #2C3E50;
    }

    /* Hide Sidebar completely if desired */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(120deg, #FF7043 0%, #FFB74D 100%);
        border-radius: 28px;
        padding: 28px 36px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 24px rgba(255, 112, 67, 0.22);
        margin-bottom: 25px;
        border: 4px solid #FFFFFF;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.12);
    }

    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
        margin-top: 6px;
    }

    /* Inline Control Deck Bar */
    .control-card {
        background: #FFFFFF;
        border-radius: 22px;
        padding: 18px 24px;
        border: 3px solid #FFE082;
        box-shadow: 0 6px 16px rgba(0,0,0,0.05);
        margin-bottom: 22px;
    }

    /* Anthropomorphic Card & Speech Bubble */
    .mascot-card {
        background: #FFFFFF;
        border-radius: 24px;
        padding: 24px;
        border: 3px solid #E0E0E0;
        box-shadow: 0 8px 22px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    .speech-bubble {
        position: relative;
        background: #F4F6F7;
        border-radius: 20px;
        padding: 20px 24px;
        font-size: 1.15rem;
        font-weight: 600;
        color: #2C3E50;
        border: 2px solid #CFD8DC;
        margin-top: 15px;
        line-height: 1.5;
    }

    .speech-bubble:after {
        content: '';
        position: absolute;
        top: -12px;
        left: 35px;
        border-width: 0 12px 12px;
        border-style: solid;
        border-color: #F4F6F7 transparent;
        display: block;
        width: 0;
    }

    .character-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 700;
        color: white;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Progress Meters */
    .meter-bg {
        background-color: #E0E0E0;
        border-radius: 50px;
        height: 22px;
        width: 100%;
        overflow: hidden;
        margin-top: 8px;
    }

    .meter-fill-cat {
        background: linear-gradient(90deg, #FF8A80, #FF5252);
        height: 100%;
        border-radius: 50px;
        transition: width 0.6s ease-in-out;
    }

    .meter-fill-dog {
        background: linear-gradient(90deg, #4FC3F7, #0288D1);
        height: 100%;
        border-radius: 50px;
        transition: width 0.6s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def get_model():
    return utils.load_classifier_model("CatsvsDogs.keras")

try:
    model = get_model()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    model_loaded = False

# Header Banner
st.markdown("""
<div class="hero-container">
    <div style="font-size: 3.5rem; margin-bottom: 5px;">🐱 🐾 🐶</div>
    <h1 class="hero-title">Paw Detective AI</h1>
    <div class="hero-subtitle">Anthropomorphic Pet Classifier • Funny AI Pet Analysis</div>
</div>
""", unsafe_allow_html=True)

# Top Control Deck (Inline controls right on the main page!)
with st.expander("🎛️ Paw Control Deck (Customize Mascot & Display Settings)", expanded=False):
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        confidence_threshold = st.slider(
            "🎯 Mascot Threshold (%)",
            min_value=50,
            max_value=95,
            value=60,
            step=5,
            help="Low confidence triggers Professor Paw's puzzling warning!"
        )
    
    with col_c2:
        show_preprocessed = st.checkbox("🔍 Show 150x150 Model Input View", value=True)
    
    with col_c3:
        show_stats = st.checkbox("📊 Show Image Color & Tensor Stats", value=False)

if 'quote_seed' not in st.session_state:
    st.session_state.quote_seed = random.randint(0, 10000)

# Input Tabs
tab_upload, tab_sample = st.tabs(["📤 Upload Any Pet Photo", "🖼️ Try Sample Pet Gallery"])

selected_image = None
image_source_label = ""

with tab_upload:
    st.markdown("##### Drop or choose any image file format (PNG, JPG, WEBP, GIF, BMP, TIFF):")
    uploaded_file = st.file_uploader(
        "Upload image file",
        type=None,
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        selected_image = uploaded_file
        image_source_label = f"Uploaded Photo: {uploaded_file.name}"

with tab_sample:
    st.markdown("##### Click on any sample pet to analyze instantly:")
    sample_dir = "sample_images"
    sample_files = {
        "🐱 Cat Sample 1 (Orange Tabby)": os.path.join(sample_dir, "cat_1.png"),
        "🐱 Cat Sample 2 (Fluffy Kitten)": os.path.join(sample_dir, "cat_2.png"),
        "🐶 Dog Sample 1 (Golden Retriever)": os.path.join(sample_dir, "dog_1.png"),
        "🐶 Dog Sample 2 (Playful Beagle)": os.path.join(sample_dir, "dog_2.png"),
    }

    cols = st.columns(4)
    for idx, (label, path) in enumerate(sample_files.items()):
        with cols[idx]:
            if os.path.exists(path):
                img_preview = Image.open(path)
                st.image(img_preview, use_container_width=True)
                if st.button(label, key=f"sample_btn_{idx}"):
                    selected_image = path
                    image_source_label = label

# Prediction Display
if selected_image is not None and model_loaded:
    try:
        # Preprocessing & Inference
        original_img, resized_img, img_batch = utils.preprocess_image(selected_image)
        predicted_class, confidence, raw_score, cat_prob, dog_prob = utils.predict_pet(model, img_batch)
        
        # Anthropomorphic Mascot metadata with funny quote
        mascot = utils.get_anthropomorphic_mascot(
            predicted_class, 
            confidence, 
            threshold=confidence_threshold, 
            seed=st.session_state.quote_seed
        )

        st.markdown("---")
        
        # Content Columns
        col_img, col_results = st.columns([1, 1.25])

        with col_img:
            st.markdown(f"#### 📸 Input Photo ({image_source_label})")
            st.image(original_img, use_container_width=True)
            
            if show_preprocessed:
                st.markdown("##### 🔬 Preprocessed Model Input (150x150 RGB)")
                st.image(resized_img, width=200)

        with col_results:
            st.markdown("#### 🕵️ Detective Mascot Report")
            
            # Anthropomorphic Mascot Funny Reply Card
            st.markdown(f"""
            <div class="mascot-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span class="character-badge" style="background-color: {mascot['badge_color']};">
                            {mascot['name']}
                        </span>
                        <div style="font-size: 0.9rem; color: #7F8C8D; margin-top: 4px; font-weight: 600;">
                            {mascot['title']} • {mascot['funny_fact']}
                        </div>
                    </div>
                    <div style="font-size: 3.2rem;">{mascot['emoji']}</div>
                </div>
                <div class="speech-bubble">
                    "{mascot['quote']}"
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Button to refresh funny reply
            if st.button("🎲 Tell Me Another Funny Reply!", use_container_width=True):
                st.session_state.quote_seed = random.randint(0, 100000)
                st.rerun()

            # AI Verdict Card
            verdict_bg = "#FFEBEE" if predicted_class == "Cat" else "#E1F5FE"
            verdict_color = "#C62828" if predicted_class == "Cat" else "#0277BD"
            
            st.markdown(f"""
            <div style="background-color: {verdict_bg}; border-radius: 20px; padding: 18px; text-align: center; border: 2.5px solid {verdict_color}; margin: 20px 0;">
                <div style="font-size: 0.9rem; color: #555; font-weight: 700; text-transform: uppercase;">
                    AI Verdict
                </div>
                <div style="font-size: 2.4rem; font-weight: 700; color: {verdict_color}; margin: 4px 0;">
                    {predicted_class.upper()} {('🐱' if predicted_class == 'Cat' else '🐶')}
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #333;">
                    {confidence:.2f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability Meters
            st.markdown("##### 📈 Class Probabilities")
            
            st.markdown(f"**🐱 Cat Probability:** `{cat_prob:.2f}%`")
            st.markdown(f"""
            <div class="meter-bg">
                <div class="meter-fill-cat" style="width: {cat_prob:.1f}%;"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<br>**🐶 Dog Probability:** `{dog_prob:.2f}%`", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="meter-bg">
                <div class="meter-fill-dog" style="width: {dog_prob:.1f}%;"></div>
            </div>
            """, unsafe_allow_html=True)

        # Image Stats Diagnostics
        if show_stats:
            st.markdown("---")
            with st.expander("📊 Image Color Distribution & Tensor Stats", expanded=True):
                col_s1, col_s2, col_s3 = st.columns(3)
                np_orig = np.array(original_img)
                col_s1.metric("Original Dimensions", f"{original_img.width} x {original_img.height} px")
                col_s2.metric("RGB Mean Pixel Value", f"{np_orig.mean():.1f} / 255")
                col_s3.metric("Model Tensor Batch Shape", f"{img_batch.shape}")

    except Exception as err:
        st.error(f"❌ Error analyzing photo: {err}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #78909C; font-weight: 600; padding: 10px;'>"
    "🐾 Built with Streamlit & TensorFlow • Anthropomorphic UI Theme"
    "</div>",
    unsafe_allow_html=True
)
