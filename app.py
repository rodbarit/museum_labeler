import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Museum Image Archive", page_icon="🏛️")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Back to the standard BLIP-base
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# --- UI ---
st.title("🏛️ Museum Image Archive")
st.write("Generate clean, searchable descriptions for artifact photos.")

uploaded_files = st.file_uploader("Upload artifacts...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    data_records = []
    
    if st.button("Generate Descriptions"):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=300)
            
            with st.spinner(f'Analyzing {uploaded_file.name}...'):
                inputs = processor(image, return_tensors="pt").to(device)
                # Using the original simple generation parameters
                out = model.generate(**inputs, max_new_tokens=50)
                description = processor.decode(out[0], skip_special_tokens=True).capitalize()
                
                st.success(f"**Label:** {description}")
                
                data_records.append({
                    "Filename": uploaded_file.name, 
                    "Description": description
                })

        # --- Export ---
        st.divider()
        df = pd.DataFrame(data_records)
        st.dataframe(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "museum_labels.csv", "text/csv")
