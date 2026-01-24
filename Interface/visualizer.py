import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import base64


@dataclass
class ImageMetrics:
    brightness: float
    contrast: float
    sharpness: float
    resolution: Tuple[int, int]
    file_size: int


class ImageProcessor:
    @staticmethod
    def prepare_image(image_np: np.ndarray, target_channels: str = "RGB") -> Optional[Image.Image]:
        if image_np is None:
            return None

        img = image_np.copy()

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        if target_channels == "RGB" and len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        return Image.fromarray(img)

    @staticmethod
    def calculate_metrics(image_np: np.ndarray) -> ImageMetrics:
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_np

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.var(laplacian))
        
        return ImageMetrics(
            brightness=brightness,
            contrast=contrast,
            sharpness=sharpness,
            resolution=(image_np.shape[1], image_np.shape[0]),
            file_size=image_np.nbytes
        )

    @staticmethod
    def create_comparison_slider(before_img: Image.Image, after_img: Image.Image) -> str:
        before_buffer = io.BytesIO()
        after_buffer = io.BytesIO()
        
        before_img.save(before_buffer, format='PNG')
        after_img.save(after_buffer, format='PNG')
        
        before_b64 = base64.b64encode(before_buffer.getvalue()).decode()
        after_b64 = base64.b64encode(after_buffer.getvalue()).decode()
        
        return f"""
        <div style="position: relative; width: 100%; aspect-ratio: 16/9; overflow: hidden; border-radius: 12px;">
            <img src="data:image/png;base64,{before_b64}" style="position: absolute; width: 100%; height: 100%; object-fit: cover;">
            <img src="data:image/png;base64,{after_b64}" style="position: absolute; width: 100%; height: 100%; object-fit: cover; clip-path: inset(0 50% 0 0);">
        </div>
        """


class VisualizationManager:
    def __init__(self):
        self.processor = ImageProcessor()

    def create_metrics_chart(self, before_metrics: ImageMetrics, after_metrics: ImageMetrics) -> go.Figure:
        categories = ['Brightness', 'Contrast', 'Sharpness']
        
        before_values = [
            before_metrics.brightness / 2.55,
            before_metrics.contrast / 2.55,
            min(before_metrics.sharpness / 10, 100)
        ]
        
        after_values = [
            after_metrics.brightness / 2.55,
            after_metrics.contrast / 2.55,
            min(after_metrics.sharpness / 10, 100)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=categories,
            y=before_values,
            marker_color='rgba(96, 165, 250, 0.7)',
            marker_line_color='rgba(96, 165, 250, 1)',
            marker_line_width=2
        ))
        
        fig.add_trace(go.Bar(
            name='Current',
            x=categories,
            y=after_values,
            marker_color='rgba(139, 92, 246, 0.7)',
            marker_line_color='rgba(139, 92, 246, 1)',
            marker_line_width=2
        ))
        
        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            yaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.1)',
                range=[0, 100],
                title='Quality Score'
            ),
            xaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.1)'
            )
        )
        
        return fig

    def create_change_heatmap(self, before_img: np.ndarray, after_img: np.ndarray) -> go.Figure:
        if before_img.shape != after_img.shape:
            after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
        
        if len(before_img.shape) == 3:
            before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        else:
            before_gray = before_img
            after_gray = after_img
        
        diff = cv2.absdiff(before_gray, after_gray)
        diff_resized = cv2.resize(diff, (100, 100))
        
        fig = go.Figure(data=go.Heatmap(
            z=diff_resized,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title='Change Intensity',
                titlefont=dict(color='#cbd5e1'),
                tickfont=dict(color='#cbd5e1')
            )
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False)
        )
        
        return fig

    def display_image_with_controls(
        self,
        image_np: np.ndarray,
        caption: str,
        enable_download: bool = True,
        show_metrics: bool = False
    ):
        if image_np is None:
            st.warning(f"No image data available for: {caption}")
            return

        try:
            pil_image = self.processor.prepare_image(image_np)
            
            st.image(pil_image, caption=caption, use_container_width=True)
            
            if show_metrics:
                metrics = self.processor.calculate_metrics(image_np)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Resolution", f"{metrics.resolution[0]}x{metrics.resolution[1]}")
                with col2:
                    st.metric("Brightness", f"{metrics.brightness:.1f}")
                with col3:
                    st.metric("Contrast", f"{metrics.contrast:.1f}")
            
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
            st.error(f"Error displaying image: {str(e)}")

    def render_classification_card(self, classification: Dict[str, Any]):
        label = classification.get('label', 'Unknown')
        confidence = classification.get('confidence', 0.0)
        
        confidence_color = '#10b981' if confidence > 0.75 else '#fbbf24' if confidence > 0.5 else '#ef4444'
        confidence_pct = f"{confidence * 100:.1f}%" if isinstance(confidence, float) else f"{confidence}%"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
                        border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(148, 163, 184, 0.2);
                        margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="color: #60a5fa; margin: 0; font-size: 1.3rem;">Event Classification</h3>
                    <div style="background: rgba(59, 130, 246, 0.1); padding: 0.5rem 1rem; border-radius: 20px;
                                border: 1px solid {confidence_color};">
                        <span style="color: {confidence_color}; font-weight: 700; font-size: 0.9rem;">
                            {confidence_pct} Confidence
                        </span>
                    </div>
                </div>
                <div style="background: rgba(59, 130, 246, 0.05); padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #3b82f6;">
                    <p style="color: #e2e8f0; font-size: 1.5rem; font-weight: 700; margin: 0; text-transform: uppercase;">
                        {label}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def render_analysis_section(self, summary: str, solutions: str):
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
                        border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(148, 163, 184, 0.2);
                        margin-bottom: 1rem;">
                <h3 style="color: #60a5fa; margin-bottom: 1rem; font-size: 1.3rem;">Situation Report</h3>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.05); padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #3b82f6; margin-bottom: 1rem;">
                    <p style="color: #cbd5e1; line-height: 1.8; margin: 0;">{summary}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
                        border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(148, 163, 184, 0.2);">
                <h3 style="color: #60a5fa; margin-bottom: 1rem; font-size: 1.3rem;">Recommended Actions</h3>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.05); padding: 1rem; border-radius: 8px;
                            border-left: 4px solid #10b981;">
                    <div style="color: #cbd5e1; line-height: 1.8;">{solutions}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def display_results(report_dict: Dict[str, Any]):
    if not report_dict:
        st.error("Report is empty.")
        return

    viz_manager = VisualizationManager()
    
    if 'classification' in report_dict:
        viz_manager.render_classification_card(report_dict['classification'])
    
    st.markdown("---")
    st.markdown("### Visual Change Detection")
    
    tab1, tab2, tab3 = st.tabs(["Side-by-Side Comparison", "Change Heatmap", "Image Metrics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Baseline Imagery**")
            if "images" in report_dict and "before" in report_dict["images"]:
                viz_manager.display_image_with_controls(
                    report_dict["images"]["before"],
                    caption="Baseline",
                    enable_download=True,
                    show_metrics=False
                )
            else:
                st.info("No baseline image provided")

        with col2:
            st.markdown("**Current Imagery**")
            if "images" in report_dict and "after" in report_dict["images"]:
                viz_manager.display_image_with_controls(
                    report_dict["images"]["after"],
                    caption="Current Status",
                    enable_download=True,
                    show_metrics=False
                )
            else:
                st.info("No current image provided")
        
        if "images" in report_dict and "difference" in report_dict["images"]:
            st.markdown("**Difference Map**")
            viz_manager.display_image_with_controls(
                report_dict["images"]["difference"],
                caption="Change Detection",
                enable_download=True,
                show_metrics=False
            )
    
    with tab2:
        if "images" in report_dict and "before" in report_dict["images"] and "after" in report_dict["images"]:
            st.markdown("**Change Intensity Heatmap**")
            heatmap_fig = viz_manager.create_change_heatmap(
                report_dict["images"]["before"],
                report_dict["images"]["after"]
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Insufficient data for heatmap generation")
    
    with tab3:
        if "images" in report_dict and "before" in report_dict["images"] and "after" in report_dict["images"]:
            st.markdown("**Comparative Image Quality Analysis**")
            
            before_metrics = viz_manager.processor.calculate_metrics(report_dict["images"]["before"])
            after_metrics = viz_manager.processor.calculate_metrics(report_dict["images"]["after"])
            
            metrics_fig = viz_manager.create_metrics_chart(before_metrics, after_metrics)
            st.plotly_chart(metrics_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Metrics**")
                st.json({
                    "resolution": f"{before_metrics.resolution[0]}x{before_metrics.resolution[1]}",
                    "brightness": f"{before_metrics.brightness:.2f}",
                    "contrast": f"{before_metrics.contrast:.2f}",
                    "sharpness": f"{before_metrics.sharpness:.2f}",
                    "file_size": f"{before_metrics.file_size / 1024:.2f} KB"
                })
            
            with col2:
                st.markdown("**Current Metrics**")
                st.json({
                    "resolution": f"{after_metrics.resolution[0]}x{after_metrics.resolution[1]}",
                    "brightness": f"{after_metrics.brightness:.2f}",
                    "contrast": f"{after_metrics.contrast:.2f}",
                    "sharpness": f"{after_metrics.sharpness:.2f}",
                    "file_size": f"{after_metrics.file_size / 1024:.2f} KB"
                })
        else:
            st.info("Insufficient data for metrics analysis")
    
    st.markdown("---")
    
    summary = report_dict.get("summary", "No situation report available")
    solutions = report_dict.get("solutions", "No recommended actions available")
    
    viz_manager.render_analysis_section(summary, solutions)


def display_image(image_np: np.ndarray, caption: str = "Satellite Analysis", enable_download: bool = True):
    viz_manager = VisualizationManager()
    viz_manager.display_image_with_controls(image_np, caption, enable_download)


if __name__ == "__main__":
    st.set_page_config(page_title="Visualizer Test", layout="wide")
    
    test_report = {
        "classification": {
            "label": "wildfire",
            "confidence": 0.87
        },
        "images": {
            "before": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            "after": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            "difference": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        },
        "summary": "Active wildfire detected with rapid expansion rate. High thermal signature indicates extreme combustion temperatures.",
        "solutions": "Immediate deployment of aerial suppression units required. Establish containment lines along southwestern perimeter."
    }
    
    display_results(test_report)