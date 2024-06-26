import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the model and processor
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

def generate_caption(image_path):
    # Load and process the image
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    caption = model.generate(**inputs)

    # Decode the caption
    caption_text = processor.decode(caption[0], skip_special_tokens=True)

    return caption_text

# Streamlit app
st.title("Image Captioning Model")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Generating caption...")
    caption = generate_caption(uploaded_file)
    st.write("Caption:", caption)