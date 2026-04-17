import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import io
import os
import requests
import json

# --- ASSET LOADING ---


def download_labels():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    filename = "imagenet_class_index.json"
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)


@st.cache_resource
def load_model_assets():
    download_labels()
    model = models.resnet50(weights='IMAGENET1K_V1').eval()
    with open("imagenet_class_index.json") as f:
        labels_map = json.load(f)
    return model, labels_map


def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        conf, idx = torch.max(prob, 1)
    return idx.item(), conf.item()

# --- THE DEFENSE ENGINE ---


def rand_disc_cleaner(image_tensor, sigma, k_clusters):
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    h, w, c = img_np.shape

    noise = np.random.normal(0, sigma, img_np.shape)
    noisy_img = img_np + noise

    pixels = img_np.reshape(-1, c)
    kmeans = KMeans(n_clusters=k_clusters, n_init=5).fit(pixels)
    centers = kmeans.cluster_centers_

    noisy_pixels = noisy_img.reshape(-1, c)
    distances = np.linalg.norm(noisy_pixels[:, np.newaxis] - centers, axis=2)
    closest_indices = np.argmin(distances, axis=1)
    cleaned_img = centers[closest_indices].reshape(h, w, c)

    return np.clip(cleaned_img, 0, 1)


def auto_purify(dirty_tensor, model, k_range=[5, 10, 12, 15], sigma_range=[0.05, 0.1, 0.15]):
    best_cleaned_np = None
    best_confidence = 0.0
    best_label = None
    best_params = (0, 0)

    # Grid search for the optimal classification margin
    for k in k_range:
        for sigma in sigma_range:
            cleaned_np = rand_disc_cleaner(
                dirty_tensor, sigma=sigma, k_clusters=k)
            cleaned_tensor = torch.from_numpy(
                cleaned_np).permute(2, 0, 1).float().unsqueeze(0)

            idx, conf = predict(model, cleaned_tensor)

            if conf > best_confidence:
                best_confidence = conf
                best_cleaned_np = cleaned_np
                best_label = idx
                best_params = (k, sigma)

    return best_cleaned_np, best_label, best_confidence, best_params


# --- STREAMLIT UI (Features Intact) ---
st.set_page_config(page_title="Auto-RandDisc Purifier", layout="centered")
st.title("🛡️ Auto-Adversarial Cleaner")

model, labels_map = load_model_assets()

# Feature 1: Uploading
uploaded_file = st.file_uploader(
    "Upload an Attacked (PGD) Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Attacked Image (Before Cleaning)",
             use_column_width=True)

    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    dirty_tensor = preprocess(img).unsqueeze(0)

    # Feature 2: Original Prediction
    d_idx, d_conf = predict(model, dirty_tensor)
    st.info(
        f"**Current Model Prediction:** {labels_map[str(d_idx)][1]} ({d_conf*100:.2f}%)")

    # The Auto-Tune Button replacing the sliders
    if st.button("Auto-Purify Image"):
        with st.spinner("Scanning for optimal clusters and noise... this takes a few seconds..."):

            # Run the Auto-Purify logic
            cleaned_np, c_idx, c_conf, best_params = auto_purify(
                dirty_tensor, model)
            cleaned_img_pil = Image.fromarray(
                (cleaned_np * 255).astype(np.uint8))

            st.divider()
            st.subheader("✨ Post-Cleaning Results")

            # Display best parameters found
            st.caption(
                f"Optimal Settings Found: {best_params[0]} Clusters, {best_params[1]} Noise")

            # Feature 3: Display Image & Verification
            st.image(
                cleaned_img_pil, caption="Purified Image (Auto-RandDisc)", use_column_width=True)
            st.success(
                f"**Cleaned Prediction:** {labels_map[str(c_idx)][1]} ({c_conf*100:.2f}%)")

            # Feature 4: Download
            buf = io.BytesIO()
            cleaned_img_pil.save(buf, format="PNG")
            st.download_button(
                label="Download Purified Image",
                data=buf.getvalue(),
                file_name="auto_purified.png",
                mime="image/png"
            )
