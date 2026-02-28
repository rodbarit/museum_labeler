import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Museum Image Archive", page_icon="🏛️", layout="wide")

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model():
    # Using BLIP-base: reliable, fast, and fits in free-tier memory
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Check for GPU (CUDA), otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# --- User Interface ---
st.title("🏛️ Museum Image-to-Text Archive")
st.write("Upload artifact photos to generate descriptive metadata for your digital collection.")

# Multi-file uploader
uploaded_files = st.file_uploader(
    "Upload images (JPG, PNG)...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    data_records = []
    
    # Process button
    if st.button("Generate Descriptions"):
        # Grid layout for results
        cols = st.columns(2)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Load and convert image
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner(f'Analyzing {uploaded_file.name}...'):
                # AI Inference
                inputs = processor(image, return_tensors="pt").to(device)
                # max_new_tokens=75 provides a decent sentence length
                out = model.generate(**inputs, max_new_tokens=75)
                description = processor.decode(out[0], skip_special_tokens=True).capitalize()
                
                # Store data for CSV
                data_records.append({
                    "Filename": uploaded_file.name,
                    "Generated_Description": description
                })
                
                # Display in the UI
                with cols[idx % 2]:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    st.info(f"**Description:** {description}")

        # --- Export Section ---
        st.divider()
        st.subheader("📋 Metadata Export")
        df = pd.DataFrame(data_records)
        st.dataframe(df, use_container_width=True)

        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
