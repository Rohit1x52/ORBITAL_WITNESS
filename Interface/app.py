import streamlit as st
import numpy as np
from PIL import Image
from app.nasa_api import fetch_imagery
from app.agent import image_analysis_agent
from Interface.visualizer import display_image

st.set_page_config(page_title="🛰️ Satellite Vision Agent", layout="centered")
st.title("🌍 Satellite Image Analysis Agent")

location = st.text_input("Enter a location:", placeholder="e.g., Mumbai, India")

if location:
    with st.spinner("Fetching satellite image..."):
        try:
            image_array = fetch_imagery(location)
            st.success("Image fetched successfully!")

            # Display image
            display_image(image_array)

            # Run Agent
            with st.spinner("Analyzing image with Vision Agent..."):
                result = image_analysis_agent.invoke(image_array)
                st.subheader("📝 Report")
                st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")