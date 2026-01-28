import streamlit as st
import time
from datetime import datetime, timedelta
from Interface import visualizer
from Interface.api_client import OrbitalWitnessAPIClient
import plotly.graph_objects as go


def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            background-attachment: fixed;
        }
        
        .stApp {
            background: transparent;
        }
        
        h1 {
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 800;
            font-size: 3.5rem !important;
            letter-spacing: -0.02em;
            text-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: #e2e8f0;
            font-weight: 700;
            font-size: 1.8rem !important;
            margin-top: 2rem !important;
        }
        
        h3 {
            color: #cbd5e1;
            font-weight: 600;
            font-size: 1.3rem !important;
        }
        
        h4 {
            color: #94a3b8;
            font-weight: 500;
            font-style: italic;
            font-size: 1.1rem !important;
        }
        
        .stMarkdown p {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.7;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
        }
        
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #f1f5f9;
        }
        
        [data-testid="stSidebar"] .stMarkdown p {
            color: #cbd5e1;
        }
        
        .stButton>button {
            border-radius: 12px;
            height: 3.2em;
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
            color: white;
            font-weight: 700;
            font-size: 1rem;
            border: none;
            width: 100%;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4), 
                        0 8px 10px -6px rgba(59, 130, 246, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }
        
        .stButton>button:hover::before {
            left: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 20px 40px -10px rgba(59, 130, 246, 0.6),
                        0 15px 20px -10px rgba(59, 130, 246, 0.4);
        }
        
        .stButton>button:active {
            transform: translateY(-1px);
        }
        
        .stTextInput>div>div>input,
        .stDateInput>div>div>input {
            border-radius: 10px;
            border: 2px solid rgba(148, 163, 184, 0.2);
            background: rgba(30, 41, 59, 0.6);
            color: #f1f5f9;
            font-size: 0.95rem;
            padding: 0.75rem;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stTextInput>div>div>input:focus,
        .stDateInput>div>div>input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
            background: rgba(30, 41, 59, 0.8);
        }
        
        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
        }
        
        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        [data-testid="stMetricDelta"] {
            color: #34d399;
        }
        
        div[data-testid="stStatusWidget"] {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        .stDivider {
            border-color: rgba(148, 163, 184, 0.2);
        }
        
        .stSuccess {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1));
            border-left: 4px solid #10b981;
        }
        
        .stError {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(220, 38, 38, 0.1));
            border-left: 4px solid #ef4444;
        }
        
        .stWarning {
            background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
            border-left: 4px solid #fbbf24;
        }
        
        .stInfo {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(37, 99, 235, 0.1));
            border-left: 4px solid #3b82f6;
        }
        
        [data-testid="stContainer"] {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            padding: 1.5rem;
            backdrop-filter: blur(10px);
        }
        
        .custom-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(20px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.5);
        }
        
        .satellite-icon {
            animation: orbit 20s linear infinite;
        }
        
        @keyframes orbit {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .pulse {
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #60a5fa, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
        }
        
        code {
            background: rgba(30, 41, 59, 0.8);
            color: #60a5fa;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }
        
        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }
        
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .stat-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            border-color: #3b82f6;
            box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        }
        
        .loading-spinner {
            border: 3px solid rgba(148, 163, 184, 0.2);
            border-top: 3px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        hr {
            border-color: rgba(148, 163, 184, 0.2);
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def create_animated_header():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div class="satellite-icon" style="font-size: 4rem; margin-bottom: 1rem;">🛰️</div>
        </div>
    """, unsafe_allow_html=True)


def create_metric_card(label, value, delta=None, icon="📊"):
    delta_html = f'<div style="color: #34d399; font-size: 0.9rem; margin-top: 0.5rem;">▲ {delta}</div>' if delta else ''
    
    st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{label}</div>
            <div style="color: #60a5fa; font-size: 2rem; font-weight: 700;">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def create_loading_animation(text="Processing"):
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem;">
            <div class="loading-spinner"></div>
            <div style="color: #cbd5e1; font-weight: 500;">{text}...</div>
        </div>
    """, unsafe_allow_html=True)


def create_status_badge(status, color="#10b981"):
    st.markdown(f"""
        <div style="display: inline-flex; align-items: center; gap: 0.5rem; 
                    background: rgba(16, 185, 129, 0.1); padding: 0.5rem 1rem; 
                    border-radius: 20px; border: 1px solid {color};">
            <div class="pulse" style="width: 8px; height: 8px; background: {color}; border-radius: 50%;"></div>
            <span style="color: {color}; font-weight: 600; font-size: 0.9rem;">{status}</span>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Orbital Witness AI",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/orbital-witness',
            'Report a bug': 'https://github.com/yourusername/orbital-witness/issues',
            'About': '# Orbital Witness AI\nAutonomous Satellite Intelligence System'
        }
    )
    
    apply_custom_styles()
    
    if 'api_client' not in st.session_state:
        with st.status("System Initialization...", expanded=True) as status:
            st.write("Connecting to API Server...")
            st.session_state.api_client = OrbitalWitnessAPIClient()
            time.sleep(0.5)
            
            if st.session_state.api_client.is_api_available():
                st.write("API Connection Established...")
                time.sleep(0.3)
                st.write("System Ready...")
                time.sleep(0.2)
                status.update(label="Orbital Witness AI Ready", state="complete", expanded=False)
            else:
                status.update(label="API Server Unavailable", state="error", expanded=True)
                st.error("Cannot connect to API server at http://localhost:8000")
                st.info("Please ensure the API server is running: `python api_server.py`")
                st.stop()

    with st.sidebar:
        st.markdown('<div style="text-align: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3662/3662817.png", width=100)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.title("Control Panel")
        st.markdown("---")
        
        st.subheader("Target Area")
        location_str = st.text_input(
            "Coordinates (Lat, Lon)",
            value="40.7128, -74.0060",
            help="Enter latitude and longitude separated by a comma",
            placeholder="e.g., 34.0522, -118.2437"
        )

        st.subheader("Timeframe")
        col1, col2 = st.columns(2)
        with col1:
            before_date = st.date_input(
                "Baseline Date",
                value=datetime.now() - timedelta(days=365),
                help="Select the reference date for comparison"
            )
        with col2:
            after_date = st.date_input(
                "Analysis Date",
                value=datetime.now(),
                help="Select the current date for analysis"
            )

        st.markdown("---")
        
        with st.expander("Advanced Settings", expanded=False):
            st.slider("Detection Sensitivity", 0, 100, 75, help="Adjust change detection threshold")
            st.selectbox("Analysis Mode", ["Standard", "High Resolution", "Fast Scan"])
            st.checkbox("Enable Real-time Monitoring", value=False)
        
        st.markdown("---")
        
        st.info("**Pro Tip:** Ensure dates are at least 30 days apart for optimal change detection accuracy.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("LAUNCH ANALYSIS", type="primary")
        
        st.markdown("---")
        st.caption("Secure Connection Established")
        st.caption("Satellites: 3 Active")

    create_animated_header()
    
    st.title("Orbital Witness AI")
    st.markdown("#### *Autonomous Satellite Intelligence & Change Detection System*")
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1)); 
                    padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin: 1.5rem 0;">
            <p style="margin: 0; color: #cbd5e1; line-height: 1.8;">
                Orbital Witness leverages <strong class="gradient-text">LCEL (LangChain Expression Language)</strong> 
                and <strong class="gradient-text">RAG (Retrieval-Augmented Generation)</strong> to detect environmental shifts, 
                urbanization, and disaster impacts from orbit. Powered by advanced AI models and real-time satellite imagery analysis.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if analyze_button:
        try:
            col_main, col_viz = st.columns([1, 2], gap="large")
            
            with col_main:
                st.markdown("### Mission Status")
                
                with st.status("Processing Satellite Data...", expanded=True) as status:
                    st.write("Parsing coordinates...")
                    time.sleep(0.3)
                    lat, lon = map(float, location_str.split(','))
                    
                    input_data = {
                        "location": (lat, lon),
                        "before_date": str(before_date),
                        "after_date": str(after_date)
                    }
                    
                    st.write("Fetching imagery tiles...")
                    st.write("Running AI detection models...")
                    st.write("Processing neural network analysis...")
                    
                    api_response = st.session_state.api_client.analyze(
                        location=(lat, lon),
                        before_date=str(before_date),
                        after_date=str(after_date)
                    )
                    
                    report = {
                        "classification": api_response.get("classification", {}),
                        "summary": api_response.get("summary", ""),
                        "solutions": api_response.get("solutions", ""),
                        "images": {},
                        "input_params": {
                            "location": (lat, lon),
                            "before_date": str(before_date),
                            "after_date": str(after_date)
                        }
                    }
                    
                    st.write("Generating intelligence report...")
                    time.sleep(0.2)
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="custom-card">
                        <h4 style="color: #60a5fa; margin-bottom: 1rem;">Mission Parameters</h4>
                        <div style="margin-bottom: 0.8rem;">
                            <span style="color: #94a3b8;">Target Coordinates:</span><br>
                            <strong style="color: #e2e8f0; font-size: 1.1rem;">{lat}°, {lon}°</strong>
                        </div>
                        <div style="margin-bottom: 0.8rem;">
                            <span style="color: #94a3b8;">Analysis Period:</span><br>
                            <strong style="color: #e2e8f0;">{before_date} → {after_date}</strong>
                        </div>
                        <div>
                            <span style="color: #94a3b8;">Duration:</span><br>
                            <strong style="color: #60a5fa;">{(after_date - before_date).days} days</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                create_status_badge("ANALYSIS SUCCESSFUL", "#10b981")
                
            with col_viz:
                st.markdown("### Intelligence Report")
                
                container = st.container(border=True)
                with container:
                    if report:
                        visualizer.display_results(report)
                    else:
                        st.warning("No data returned from the agent. Please try again.")

        except ValueError:
            st.error("**Invalid Coordinates**: Please ensure format is `Lat, Lon` (e.g., 40.7128, -74.0060)")
        except Exception as e:
            st.error(f"**System Error**: {str(e)}")
            st.info("Try adjusting the date range or coordinates and run the analysis again.")
    
    else:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05)); 
                        padding: 2rem; border-radius: 16px; border: 2px dashed rgba(148, 163, 184, 0.3); 
                        text-align: center; margin: 2rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;"></div>
                <h3 style="color: #cbd5e1; margin-bottom: 1rem;">Ready for Launch</h3>
                <p style="color: #94a3b8; margin-bottom: 0;">
                    Configure the parameters in the sidebar and click <strong style="color: #60a5fa;">LAUNCH ANALYSIS</strong> to begin satellite reconnaissance.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### System Overview", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Satellites Active", "3", "100% Uptime", "")
        
        with col2:
            create_metric_card("Database Records", "1.2M", "+400 today", "")
        
        with col3:
            create_metric_card("AI Models", "12", "Running", "")
        
        with col4:
            create_metric_card("System Status", "100%", "Operational", "")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("### Recent Activity", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="custom-card">
                    <h4 style="color: #60a5fa; margin-bottom: 1rem;">Active Monitoring Zones</h4>
                    <ul style="color: #cbd5e1; line-height: 2;">
                        <li>Amazon Rainforest (Deforestation Detection)</li>
                        <li>California Coast (Wildfire Risk Assessment)</li>
                        <li>Arctic Circle (Ice Melt Tracking)</li>
                        <li>Southeast Asia (Flood Monitoring)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="custom-card">
                    <h4 style="color: #60a5fa; margin-bottom: 1rem;">Capabilities</h4>
                    <ul style="color: #cbd5e1; line-height: 2;">
                        <li>Real-time Change Detection</li>
                        <li>Multi-spectral Analysis</li>
                        <li>AI-Powered Classification</li>
                        <li>Automated Report Generation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()