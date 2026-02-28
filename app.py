import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import pandas as pd

st.set_page_config(page_title="Museum Dual-Labeler", page_icon="🏛️", layout="wide")

@st.cache_resource
def load_model():
    # Using Florence-2-base
    model_id = "microsoft/Florence-2-base" 
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

def get_florence_output(image, task_prompt):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3
    )
    results = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return results.replace(task_prompt, "").strip()

st.title("🏛️ Dual-Layer Museum Labeling")
st.write("Generating both short captions and detailed visual analysis.")

uploaded_files = st.file_uploader("Upload artifacts...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    data_records = []
    
    if st.button("Process Archive"):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner(f'Processing {uploaded_file.name}...'):
                # Task 1: Short Caption
                short_desc = get_florence_output(image, "<CAPTION>")
                
                # Task 2: Detailed Description
                long_desc = get_florence_output(image, "<DETAILED_CAPTION>")
                
                data_records.append({
                    "Filename": uploaded_file.name,
                    "Short_Caption": short_desc,
                    "Detailed_Description": long_desc
                })
                
                # Display to Curator
                with st.expander(f"View Results: {uploaded_file.name}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, use_container_width=True)
                    with col2:
                        st.subheader("Short Caption (Alt-Text)")
                        st.info(short_desc)
                        st.subheader("Detailed Analysis")
                        st.success(long_desc)

        # Final Table and Export
        st.divider()
        df = pd.DataFrame(data_records)
        st.write("### Review Final Metadata")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Master CSV", csv, "museum_dual_labels.csv", "text/csv")
