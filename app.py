import streamlit as st
st.set_page_config(page_title="AI Image Colorizer", layout="wide")

import cv2
import numpy as np
from PIL import Image
import os

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))

    prototxt = os.path.join(base, "models", "colorization_deploy_v2.prototxt")
    model = os.path.join(base, "models", "colorization_release_v2.caffemodel")
    points = os.path.join(base, "models", "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, dtype="float32")]

    return net

net = load_model()

# ---------------- Colorize ----------------
def colorize(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    return (255 * colorized).astype("uint8")

# ---------------- UI ----------------
st.title("🎨 AI Black & White Image Colorizer")
st.write("Upload a black & white image and convert it into color using AI.")

file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Colorized")
        result = colorize(img)
        st.image(result, use_column_width=True)
else:
    st.info("Please upload a black & white image")
