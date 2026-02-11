import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

st.set_page_config(page_title="CT Heart Segmentation", layout="centered")

st.title("CT Heart Segmentation using U-Net")
st.write("Upload a **CT scan image** and the model will predict the **heart segmentation mask**, then provide an **AI recommendation** based on the result.")


def build_unet(input_shape=(128, 128, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model



@st.cache_resource
def load_unet_model():
    model = build_unet()
    model.load_weights("heart_segmentation_model.h5")
    return model

model = load_unet_model()

# -------------------------------------------------
# Upload and predict
# -------------------------------------------------
uploaded_file = st.file_uploader(" Upload a CT image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess
    image = Image.open(uploaded_file).convert('L')
    image_resized = image.resize((128, 128))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, 3))

    # Prediction
    with st.spinner(" Segmenting the heart region..."):
        pred_mask = model.predict(img_array)[0, :, :, 0]
        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)

    # Display results
    st.subheader(" Segmentation Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original CT Image")
    with col2:
        st.image(pred_mask, caption="Predicted Mask (Raw)")
    with col3:
        overlay = np.array(image_resized.convert("RGB"))
        red_mask = np.zeros_like(overlay)
        red_mask[..., 0] = pred_mask_bin * 255
        overlay_img = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)
        st.image(overlay_img, caption="Overlay (Heart in Red)")

    st.success(" Segmentation complete!")

    # -------------------------------------------------
    # AI Recommendation Section
    # -------------------------------------------------
    heart_area = np.sum(pred_mask_bin)
    total_area = pred_mask_bin.shape[0] * pred_mask_bin.shape[1]
    heart_ratio = heart_area / total_area

    st.write(f" **Detected Heart Area Ratio:** {heart_ratio*100:.2f}%")
    st.subheader(" AI Recommendation")

    if heart_ratio < 0.01:
        st.warning(" The heart region is almost invisible — please verify the image quality or upload another scan.")
    elif heart_ratio < 0.15:
        st.success(" Heart region looks within expected range — no obvious abnormality detected.")
    elif heart_ratio < 0.35:
        st.info(" Slightly larger detected heart area — recommend reviewing with a cardiologist.")
    else:
        st.error(" Abnormally large heart region detected! Please consult a medical specialist immediately.")

else:
    st.info(" Please upload a CT scan image to begin.")

st.markdown("---")
st.caption("Developed by Ahmed Mokhtar · U-Net for Medical Image Segmentation")