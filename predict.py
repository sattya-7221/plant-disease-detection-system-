import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==============================
# 🔧 Page Configuration
# ==============================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

IMAGE_SIZE = 256

# ==============================
# 🌐 Language Selector
# ==============================
language = st.radio(
    "🌍 भाषा चुनें / Select Language",
    ("हिन्दी", "English"),
    horizontal=True
)

# ==============================
# 🌍 Language Text Dictionary
# ==============================
text = {
    "हिन्दी": {
        "title": "🌿 AI आधारित पौधा रोग पहचान प्रणाली",
        "upload": "📸 पत्ते की फोटो अपलोड करें",
        "result": "🔍 परिणाम",
        "plant": "🌱 पौधा",
        "disease": "🦠 रोग",
        "confidence": "📊 विश्वास स्तर",
        "low_conf": "⚠ विश्वास स्तर कम है। कृपया स्पष्ट फोटो अपलोड करें।",
        "loading": "🔄 रोग की पहचान की जा रही है...",
        "model_error": "⚠ मॉडल लोड नहीं हो पाया।",
        "folder_error": "⚠ Training folder नहीं मिला।",
        "developer": "👨‍💻 विकसितकर्ता: Satyendra Saini (NIELIT Ajmer)"
    },
    "English": {
        "title": "🌿 AI Based Plant Disease Detection System",
        "upload": "📸 Upload Leaf Image",
        "result": "🔍 Prediction Result",
        "plant": "🌱 Plant",
        "disease": "🦠 Disease",
        "confidence": "📊 Confidence Level",
        "low_conf": "⚠ Low confidence. Please upload a clear image.",
        "loading": "🔄 Detecting disease...",
        "model_error": "⚠ Model could not be loaded.",
        "folder_error": "⚠ Training folder not found.",
        "developer": "👨‍💻 Developer: Satyendra Saini (NIELIT Ajmer)"
    }
}

t = text[language]

st.title(t["title"])

# ==============================
# 🦠 Disease Hindi Translation Dictionary (PlantVillage Correct)
# ==============================
disease_translation = {

    # 🌶 Pepper (Capsicum)
    "Pepper__bell___Bacterial_spot": "शिमला मिर्च बैक्टीरियल स्पॉट रोग",
    "Pepper__bell___healthy": "शिमला मिर्च स्वस्थ है",

    # 🥔 Potato
    "Potato___Early_blight": "आलू अर्ली ब्लाइट रोग",
    "Potato___Late_blight": "आलू लेट ब्लाइट रोग",
    "Potato___healthy": "आलू स्वस्थ है",

    # 🍅 Tomato
    "Tomato___Bacterial_spot": "टमाटर बैक्टीरियल स्पॉट रोग",
    "Tomato___Early_blight": "टमाटर अर्ली ब्लाइट रोग",
    "Tomato___Late_blight": "टमाटर लेट ब्लाइट रोग",
    "Tomato___Leaf_Mold": "टमाटर लीफ मोल्ड रोग",
    "Tomato___Septoria_leaf_spot": "टमाटर सेप्टोरिया पत्ती धब्बा रोग",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "टमाटर स्पाइडर माइट्स रोग",
    "Tomato___Target_Spot": "टमाटर टार्गेट स्पॉट रोग",
    "Tomato___Tomato_mosaic_virus": "टमाटर मोज़ेक वायरस रोग",
    "Tomato___Tomato_YellowLeaf_Curl_Virus": "टमाटर पीला पत्ता मरोड़ वायरस",
    "Tomato___healthy": "टमाटर स्वस्थ है"
}

# ==============================
# 🤖 Load Model
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/2.keras")

try:
    model = load_model()
except:
    st.error(t["model_error"])
    st.stop()

# ==============================
# 📂 Load Class Names
# ==============================
DATA_DIR = "training/PlantVillage"

if os.path.exists(DATA_DIR):
    class_names = sorted(os.listdir(DATA_DIR))
else:
    st.error(t["folder_error"])
    st.stop()

# ==============================
# 📤 Upload Section
# ==============================
uploaded_file = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Preview", use_column_width=True)

    with st.spinner(t["loading"]):

        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        predicted_class = class_names[predicted_index]

        # Split
        if "___" in predicted_class:
            plant_name, disease_name = predicted_class.split("___")
        else:
            plant_name = predicted_class
            disease_name = "Unknown"

        # Hindi disease name
        hindi_disease = disease_translation.get(
            predicted_class,
            disease_name.replace("_", " ")
        )

    st.subheader(t["result"])

    if confidence > 80:
        st.success(f"{t['plant']}: {plant_name.replace('_', ' ')}")
        
        if language == "हिन्दी":
            st.success(f"{t['disease']}: {hindi_disease}")
        else:
            st.success(f"{t['disease']}: {disease_name.replace('_', ' ')}")

    elif confidence > 50:
        st.warning(f"{t['plant']}: {plant_name.replace('_', ' ')}")
        
        if language == "हिन्दी":
            st.warning(f"{t['disease']}: {hindi_disease}")
        else:
            st.warning(f"{t['disease']}: {disease_name.replace('_', ' ')}")
    else:
        st.error(t["low_conf"])

    st.info(f"{t['confidence']}: {round(confidence, 2)} %")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(t["developer"])