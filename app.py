import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# --- Page Config ---
st.set_page_config(page_title="Museum Image Labeler", page_icon="🖼️")

# --- Model Loading (Cached to prevent reloading) ---
@st.cache_resource
def load_model():
    # 'base' is fast; swap to 'large' for more detail if your PC has a GPU
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# --- UI Components ---
st.title("🏛️ Museum Image-to-Text Labeler")
st.write("Upload a digital image of an artifact or painting to generate a searchable description.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Museum Object', use_container_width=True)
    
    # Generate Button
    if st.button("Generate Description"):
        with st.spinner('Analyzing the artwork...'):
            # Process the image
            inputs = processor(image, return_tensors="pt").to(device)
            
            # Generate the text
            # max_new_tokens controls the length of the description
            out = model.generate(**inputs, max_new_tokens=50)
            description = processor.decode(out[0], skip_special_tokens=True)
            
            st.success("Analysis Complete!")
            
            # Display Results
            st.subheader("Generated Metadata")
            edited_description = st.text_area("Descriptive Label:", value=description.capitalize())
            
            # Technical Tags (Extracted from the text)
            tags = edited_description.split()
            st.write("**Suggested Search Tags:**")
            st.info(", ".join([f"#{tag.strip(',.')}" for tag in tags if len(tag) > 3]))

            # Future "Save" function placeholder
            if st.button("Confirm and Save to Archive"):
                st.write("Description saved to database (simulated)!")
