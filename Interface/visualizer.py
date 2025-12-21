import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

def _prepare_image(image_np, target_channels="RGB"):
    if image_np is None:
        return None

    img = image_np.copy()

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if target_channels == "RGB" and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)

def display_image(image_np, caption="Satellite Analysis", enable_download=True):
    if image_np is None:
        st.warning(f"No image data available for: {caption}")
        return

    try:
        pil_image = _prepare_image(image_np)
        
        st.image(pil_image, caption=caption, use_column_width=True, clamp=True)
        
        if enable_download:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"Download {caption}",
                data=byte_im,
                file_name=f"{caption.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def display_results(report_dict):
    if not report_dict:
        st.error("Report is empty.")
        return

    st.divider()
    st.subheader("Visual Change Detection")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline Imagery (Before)**")
        if "before_image" in report_dict:
            display_image(report_dict["before_image"], caption="Baseline", enable_download=False)
        else:
            st.info("No baseline image provided.")

    with col2:
        st.markdown("**Current Imagery (After)**")
        if "after_image" in report_dict:
            display_image(report_dict["after_image"], caption="Current Status", enable_download=False)
        else:
            st.info("No current image provided.")

    st.divider()
    st.subheader("Strategic Analysis")
    
    analysis_text = report_dict.get("analysis", "No analysis text returned.")
    
    st.info(analysis_text)