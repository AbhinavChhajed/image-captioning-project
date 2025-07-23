import streamlit as st
from PIL import Image
st.set_page_config(layout="wide")

st.title("Image Captioning Project")

st.header("By - Abhinav Chhajed")

file = st.file_uploader("Upload Image Here",['jpeg','jpg','png'])

if file is not None:
    img = Image.open(file)
    st.image(img,"Uploaded Image",width=500)
    st.button("genererate caption")