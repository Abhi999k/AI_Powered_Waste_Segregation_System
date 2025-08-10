import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("beta_model_v1.keras")
class_names = ["Metal", "Organic", "Paper", "Plastic", "glass"]

st.set_page_config(page_title="Waste Classification", page_icon="♻️")
st.title("♻️ Waste Classification App")
st.write("Upload an image of waste and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_file = Image.open(uploaded_file).convert("RGB")
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    img_resized = image_file.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100 

    st.subheader("Prediction Result")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.bar_chart(predictions[0])
