import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import json

# Load ImageNet labels


@st.cache_resource
def get_model_and_labels():
    model = models.resnet50(weights='IMAGENET1K_V1').eval()
    # Standard labels for interpretation
    # You can download this file online
    labels = json.load(open("imagenet_class_index.json"))
    return model, labels


def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        prob = F.softmax(output, dim=1)
        conf, idx = torch.max(prob, 1)
    return idx.item(), conf.item()

# PGD logic as per paper: x_{t+1} = Π(x_t + η * sign(∇x L(x_t))) [cite: 64]


def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, steps=10):
    images = images.clone().detach()
    adv_images = images.clone().detach()
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    return adv_images


st.title("PGD Attacker & Predictor")
model, labels_map = get_model_and_labels()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    input_tensor = preprocess(img).unsqueeze(0)

    # Original Prediction
    idx, conf = predict(model, input_tensor)
    st.write(
        f"**Original Prediction:** {labels_map[str(idx)][1]} ({conf*100:.2f}%)")
    st.image(img, width=300)

    if st.button("Generate PGD Attack"):
        adv_tensor = pgd_attack(model, input_tensor, torch.tensor([idx]))

        # Attack Prediction
        adv_idx, adv_conf = predict(model, adv_tensor)
        adv_img = transforms.ToPILImage()(adv_tensor.squeeze())

        st.subheader("Post-Attack Results")
        st.write(
            f"**Model now thinks this is:** {labels_map[str(adv_idx)][1]} ({adv_conf*100:.2f}%)")
        st.image(adv_img, width=300)

        # Download
        buf = io.BytesIO()
        adv_img.save(buf, format="PNG")
        st.download_button("Download Adversarial Image",
                           buf.getvalue(), "attacked.png", "image/png")
