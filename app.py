import os
import time
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=128, margin=40, post_process=False)

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Mustache Power Rating™",
    page_icon="🥸",
    layout="centered"
)

# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():
    mustache_model = tf.keras.models.load_model(
        "models/mustache_detector2.keras"
    )
    epic_model = tf.keras.models.load_model(
        "models/epic_detector.keras"
    )
    return mustache_model, epic_model


mustache_model, epic_model = load_models()

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = (128, 128)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.image("assets/logo.png", use_container_width=True)

st.caption(
    "Official Facial Hair Assessment "
    "by the International Mustache Authority"
)

# --------------------------------------------------
# Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Submit specimen for analysis",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def prepare_image(image):
    image = image.convert("RGB")

    face = mtcnn(image)

    if face is not None:
        # post_process=False → tensor i [0, 255] direkt
        arr = face.permute(1, 2, 0).numpy().astype(np.uint8)
    else:
        # Fallback om inget ansikte hittas
        arr = np.array(image.resize(IMG_SIZE))

    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_epicness(score):
    if score >= 0.92:
        return (
            "🏆 LEGENDARY STACHE",
            "The International Mustache Authority is speechless.",
            "assets/legendary.mp4"
        )

    elif score >= 0.80:
        return (
            "🔥 EPIC STACHE",
            "Alert the International Mustache Authority immediately.",
            "assets/epic.mp4"
        )

    elif score >= 0.65:
        return (
            "🎩 Distinguished Gentleman",
            "A respectable upper-lip performance.",
            "assets/medium.mp4"
        )

    elif score >= 0.45:
        return (
            "🧔 Standard Issue Mustache",
            "Acceptable. Not historic.",
            "assets/medium.mp4"
        )

    elif score >= 0.25:
        return (
            "🌱 Apprentice Stache",
            "Mustache growth is currently in beta testing.",
            "assets/apprentice.mp4"
        )

    else:
        return (
            "🪶 Fjunig Mustache",
            "The mustache exists mostly as a theoretical concept.",
            "assets/fjunig.mp4"
        )


def animate_scanner():
    labels = [
        "🔬 Scanning upper lip region...",
        "🧬 Analyzing hair density matrix...",
        "⚗️  Cross-referencing mustache database...",
        "🏛️  Consulting the Authority archives...",
        "📊 Computing power rating...",
    ]

    progress_bar = st.progress(0)
    status = st.empty()
    meter1 = st.empty()
    meter2 = st.empty()

    for i, label in enumerate(labels):
        status.markdown(f"**{label}**")

        for v in list(range(0, 101, 5)) + list(range(100, -1, -5)):
            meter1.progress(v / 100)
            meter2.progress(abs(v - 100) / 100)
            time.sleep(0.015)

        progress_bar.progress((i + 1) / len(labels))

    status.empty()
    meter1.empty()
    meter2.empty()
    progress_bar.empty()

# --------------------------------------------------
# Analysis
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Submitted specimen", use_container_width=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze = st.button("🔬 Initiate Analysis", use_container_width=True)

    if analyze:

        img_array = prepare_image(image)

        animate_scanner()

        mustache_prob = float(
            mustache_model.predict(img_array, verbose=0)[0][0]
        )

        st.markdown("---")

        st.write("mustache_prob:", mustache_prob)

        if mustache_prob < 0.45:
            st.error("❌ NO CERTIFIED MUSTACHE DETECTED")
            st.write(
                "Verdict: Upper lip currently operating "
                "without sufficient authority."
            )
            if os.path.exists("assets/no.mp4"):
                st.video("assets/no.mp4")

        else:
            thin_prob = float(
                epic_model.predict(img_array, verbose=0)[0][0]
            )

            epic_score_raw = 1 - thin_prob

            if mustache_prob > 0.80:
                boost = (mustache_prob - 0.80) / 0.20 * 0.12
            else:
                boost = 0

            epic_score = min(1.0, epic_score_raw + boost)

            st.write("thin_prob:", thin_prob)
            st.write("epic_score:", epic_score)

            title, description, video_file = classify_epicness(epic_score)

            # Video överst om det finns en
            if video_file and os.path.exists(video_file):
                st.video(video_file)

            st.markdown("<br>", unsafe_allow_html=True)

            st.metric(
                "Mustache Power Rating™",
                f"{epic_score * 100:.1f} / 100"
            )
            st.progress(float(epic_score))

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader(title)
            st.write(description)

            if epic_score >= 0.85:
                st.balloons()

            st.divider()
            st.caption(
                "Results certified by the "
                "International Mustache Authority™"
            )
