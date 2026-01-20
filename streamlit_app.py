import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

try:
    from eurosat.model import create_model, ModelConfig
except ImportError:
    # Bu hack demo ortamında gereklidir
    sys.path.append(str(Path(__file__).parent / "src"))
    from eurosat.model import create_model, ModelConfig

# Page config
st.set_page_config(page_title="EuroSAT Classifier", page_icon="🛰️")


@st.cache_resource
def load_model(model_path="models/model.pth"):
    """Loads the trained model with caching to speed up reloads."""
    # TODO: Prod ortamında config parametrelerini config.json'dan okumalıyız.
    # Şimdilik demo için default değerleri kullanıyoruz.
    cfg = ModelConfig()
    model = create_model(cfg)

    # Check if weights exist
    if os.path.exists(model_path):

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        st.warning(f"Model file not found at {model_path}. Using random weights.")

    model.eval()
    return model


def process_image(image):
    """Prepares image for the model (Resize -> Tensor -> Normalize)."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


# --- UI Layout ---
st.title("EuroSAT Land Use Classification")
st.markdown("Upload a satellite image to classify its land use type.")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.text("Architecture: ResNet18")
st.sidebar.text("Dataset: EuroSAT")
st.sidebar.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            model = load_model()
            input_tensor = process_image(image)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

            # Get top prediction
            conf, pred_idx = torch.max(probs, 0)
            prediction = classes[pred_idx]

            st.success(f"Prediction: **{prediction}**")
            st.info(f"Confidence: {conf.item():.2%}")


    st.subheader("Class Probabilities")
    
    # Create a simple dataframe for the chart
    chart_data = pd.DataFrame(
        {"Class": classes, "Probability": probs.numpy()}
    ).set_index("Class")
    
    st.bar_chart(chart_data)

else:
    st.info("Please upload an image to start.")
