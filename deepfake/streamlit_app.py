"""Fake vs Real face detection — Deep Learning project, ENSA Tétouan (BDIA).
Team project supervised by Pr. Anass Belcaid — demo by Fatima Zahrae Ahannuk.

The trained weights are downloaded from the team's Hugging Face repository (merouane02/DL1).
Every model is loaded with strict=True: a model whose weights do not match its architecture
is left out instead of silently running with random weights."""
import io
import os
import urllib.request

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Fake vs Real Face Detection", page_icon="🧠", layout="wide")
torch.set_grad_enabled(False)
torch.set_num_threads(2)
WEIGHTS = "https://huggingface.co/spaces/merouane02/DL1/resolve/main/{}"

imagenet_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
custom_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class VGG16Custom(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        layers, c = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU()]
                c = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(4096), nn.ReLU(),
                                        nn.LazyLinear(4096), nn.ReLU(), nn.LazyLinear(num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class AlexNetInspo(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(4096), nn.ReLU(),
                                        nn.LazyLinear(4096), nn.ReLU(), nn.LazyLinear(num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class FiveBlockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block1, self.block2, self.block3 = blk(3, 32), blk(32, 64), blk(64, 128)
        self.block4, self.block5 = blk(128, 256), blk(256, 256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        for b in (self.block1, self.block2, self.block3, self.block4, self.block5):
            x = b(x)
        return torch.sigmoid(self.out(self.drop(torch.relu(self.fc1(self.flatten(x))))))


def densenet():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.classifier.in_features, 512), nn.ReLU(inplace=True),
                                 nn.Dropout(0.3), nn.Linear(512, 2))
    return m


SPECS = [  # name, constructor, weight file, transform, output type
    ("DenseNet-121 (transfer learning)", densenet, "densenet_ai_detection.pth", imagenet_tf, "softmax"),
    ("FiveBlockCNN (custom model)", FiveBlockCNN, "mymodel3.pth", custom_tf, "sigmoid"),
    ("AlexNet (from scratch)", AlexNetInspo, "alex_binary.pth", custom_tf, "softmax"),
]


@st.cache_resource(show_spinner=False)
def load_models():
    loaded, status = {}, []
    for name, build, fname, tf, kind in SPECS:
        try:
            path = os.path.join("/tmp", fname)
            if not os.path.exists(path):
                urllib.request.urlretrieve(WEIGHTS.format(fname), path)
            model = build()
            model(torch.zeros(1, 3, 224, 224))  # materialise LazyLinear layers before loading
            model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
            loaded[name] = (model.eval(), tf, kind)
            status.append(f"✅ {name}")
        except Exception as err:
            status.append(f"❌ {name}: {type(err).__name__}")
            print(name, "not loaded:", err)
    return loaded, status


def predict(model_name, image):
    model, tf, kind = MODELS[model_name]
    x = tf(image.convert("RGB")).unsqueeze(0)
    if kind == "sigmoid":
        p_fake = float(model(x).item())
        return np.array([1 - p_fake, p_fake])
    return torch.softmax(model(x), dim=1)[0].numpy()


with st.spinner("Downloading the trained CNNs (first visit only, ~1 min)…"):
    MODELS, STATUS = load_models()

st.title("🧠 Fake vs Real Face Detection")
st.markdown(
    "Deep Learning project at ENSA Tétouan (Big Data & AI): CNNs trained to tell real face photos from AI-generated "
    "ones. 4 architectures were compared — DenseNet-121 (transfer learning), VGG16 and AlexNet trained from scratch, and "
    "a custom 5-block CNN — up to **~87% test accuracy** (VGG16, 537 MB, is not loaded in this free demo).  \n"
    "Team project supervised by Pr. Anass Belcaid · [Code](https://github.com/merouane01/DL1) · Demo by "
    "**Fatima Zahrae Ahannuk**"
)
left, right = st.columns(2)
with left:
    if st.button("🎲 Get a random AI-generated face (thispersondoesnotexist.com)"):
        req = urllib.request.Request("https://thispersondoesnotexist.com/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            st.session_state["img"] = r.read()
    up = st.file_uploader("…or upload a face photo", type=["jpg", "jpeg", "png", "webp"])
    if up is not None:
        st.session_state["img"] = up.getvalue()
    image = Image.open(io.BytesIO(st.session_state["img"])) if "img" in st.session_state else None
    if image is not None:
        st.image(image, width=320)
with right:
    if not MODELS:
        st.error("No model could be loaded. " + " · ".join(STATUS))
    elif image is None:
        st.info("Upload a face photo, or click the button to get an AI-generated face.")
    else:
        name = st.selectbox("Model", list(MODELS))
        probs = predict(name, image)
        st.metric("Verdict", "FAKE (AI-generated)" if probs[1] > 0.5 else "REAL", f"{probs.max():.0%} confidence", delta_color="off")
        st.progress(float(probs[1]), text=f"P(fake) = {probs[1]:.0%}")
        st.markdown("**All models on this image**")
        rows = []
        for n in MODELS:
            p = predict(n, image)
            rows.append({"Model": n, "P(fake)": f"{p[1]:.0%}", "Verdict": "FAKE" if p[1] > 0.5 else "REAL"})
        st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Models loaded: " + " · ".join(STATUS))
    st.caption("Limits: trained on one dataset; a detector can generalise poorly to images from other generators, to heavy "
               "compression or to non-face images. Best results on a cropped, front-facing face.")
