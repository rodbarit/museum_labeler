import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Museum Archive AI", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

st.title("🏛️ Museum Image-to-Text Labeling System")
st.write("Upload multiple images to generate a descriptive archive for semantic search.")

# 1. Multi-file uploader
uploaded_files = st.file_uploader("Upload artifacts...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    data_records = []
    
    if st.button("Process All Images"):
        cols = st.columns(3) # Display in a grid
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner(f'Analyzing {uploaded_file.name}...'):
                inputs = processor(image, return_tensors="pt").to(device)
                out = model.generate(**inputs, max_new_tokens=60)
                description = processor.decode(out[0], skip_special_tokens=True).capitalize()
                
                # Store record
                data_records.append({
                    "Filename": uploaded_file.name,
                    "Generated_Description": description
                })
                
                # Show in UI
                with cols[idx % 3]:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    st.info(description)

        # 2. Display the Searchable Table
        st.divider()
        st.subheader("📋 Generated Metadata Table")
        df = pd.DataFrame(data_records)
        st.dataframe(df, use_container_width=True)

        # 3. Export to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Labels as CSV",
            data=csv,
            file_name="museum_labels.csv",
            mime="text/csv",
        )
