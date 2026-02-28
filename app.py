import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="Museum Archive AI", page_icon="🏛️", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Stable BLIP model for Streamlit Cloud compatibility
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# --- UI Layout ---
st.title("🏛️ Museum Image-to-Text Archive")
st.write("Generating high-detail descriptive metadata for digital collections.")

# Sidebar for tweaking length on the fly
st.sidebar.header("Description Settings")
min_length = st.sidebar.slider("Minimum Length", 30, 100, 50)
max_length = st.sidebar.slider("Maximum Length", 100, 250, 150)

uploaded_files = st.file_uploader(
    "Upload artifact photos...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    data_records = []
    
    if st.button("Generate Detailed Analysis"):
        cols = st.columns(2)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner(f'Analyzing {uploaded_file.name}...'):
                # We provide a 'prompt' to nudge the AI into a professional tone
                prompt = "A detailed museum catalog description of"
                inputs = processor(image, text=prompt, return_tensors="pt").to(device)
                
                # ADVANCED GENERATION SETTINGS
                out = model.generate(
                    **inputs, 
                    max_new_tokens=max_length,
                    min_length=min_length,
                    num_beams=5,             # High quality search
                    repetition_penalty=1.5,  # Avoids loops
                    length_penalty=1.2,      # Encourages more words
                    early_stopping=True
                )
                
                description = processor.decode(out[0], skip_special_tokens=True).capitalize()
                
                # Store data
                data_records.append({
                    "Filename": uploaded_file.name,
                    "Generated_Description": description
                })
                
                # UI Display
                with cols[idx % 2]:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    st.success(f"**Analysis:** {description}")

        # --- Export Section ---
        st.divider()
        st.subheader("📋 Metadata Export")
        df = pd.DataFrame(data_records)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Archive CSV",
            data=csv,
            file_name="museum_detailed_metadata.csv",
            mime="text/csv",
        )
else:
    st.info("Upload images to begin generating descriptions.")
