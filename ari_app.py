


#model = load_model(r"C:\Users\nisha\best_resnet50.pth")


  # ari_app.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource  # caches the model for faster reloads
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create ResNet50 model architecture
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Bird, Drone
    
    # Load model state
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    class_names = checkpoint["class_names"]
    return model, class_names

model_path = r"C:\Users\nisha\best_resnet50.pth"# path to your saved model
model, class_names = load_model(model_path)

# -----------------------------
# Image Upload
# -----------------------------
st.title("Aerial Object Classification: Bird vs Drone")
st.write("Upload an image and the model will predict if it's a Bird or a Drone.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # -----------------------------
    # Preprocess Image
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    
    # -----------------------------
    # Make Prediction
    # -----------------------------
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    st.write(f"**Predicted Class:** {class_names[pred]}")


