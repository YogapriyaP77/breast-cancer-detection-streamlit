
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms

# 🔹 Load model
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 🔹 Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔹 UI
st.title("Breast Cancer Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        # 🔹 Change label names here
        if pred.item() == 0:
            label = "Non-Cancer"
        else:
            label = "Cancer"

        st.subheader(f"Prediction: {label}")
        st.write(f"confidence: {confidence.item()*100:.2f}%")