import streamlit as st
from PIL import Image
import numpy as np

def display_image(image_np, caption="Satellite Image"):
    image = Image.fromarray(image_np)
    st.image(image, caption=caption, use_column_width=True)