import streamlit as st
import time
from datetime import datetime, timedelta
from app.agent import create_satellite_agent
from Interface import visualizer

def local_css():
    st.markdown("""
        <style>
        /* Global Settings */
        .main {
            background-color: #f8f9fa;
        }
        h1 {
            color: #1E3A8A;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        h2, h3 {
            color: #1f2937;
            font-family: 'Helvetica Neue', sans-serif;
        }
        /* Card Styling for Results */
        .stAlert {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Custom Button */
        .stButton>button {
            border-radius: 8px;
            height: 3em;
            background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%);
            color: white;
            font-weight: bold;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3);
        }
        /* Input Fields */
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        /* Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Orbital Witness AI", 
        page_icon="", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    local_css()
    if 'agent_chain' not in st.session_state:
        with st.status(" System Initialization...", expanded=True) as status:
            st.write("Loading Satellite Agent...")
            st.session_state.agent_chain = create_satellite_agent()
            time.sleep(1) 
            st.write("Connecting to Knowledge Base...")
            status.update(label="Orbital Witness AI Ready", state="complete", expanded=False)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3662/3662817.png", width=80)
        st.title("Control Panel")
        st.markdown("---")
        
        st.subheader(" Target Area")
        location_str = st.text_input(
            "Coordinates (Lat, Lon)", 
            value="40.7128, -74.0060",
            help="Enter latitude and longitude separated by a comma."
        )

        st.subheader(" Timeframe")
        col1, col2 = st.columns(2)
        with col1:
            before_date = st.date_input("Baseline Date", value=datetime.now() - timedelta(days=365))
        with col2:
            after_date = st.date_input("Analysis Date", value=datetime.now())

        st.markdown("---")
        st.info(" **Tip:** Ensure dates are at least 1 month apart for significant satellite changes.")
        
        analyze_button = st.button(" Launch Analysis", type="primary")
    st.title(" Orbital Witness AI")
    st.markdown("#### *Autonomous Satellite Intelligence & Change Detection System*")
    st.markdown("""
        Orbital Witness leverages **LCEL (LangChain Expression Language)** and **RAG (Retrieval-Augmented Generation)** to detect environmental shifts, urbanization, and disaster impacts from orbit.
    """)
    st.divider()
    if analyze_button:
        try:
            col_main, col_viz = st.columns([1, 2])
            
            with col_main:
                st.write("###  Mission Status")
                with st.status("Processing Satellite Data...", expanded=True) as status:
                    st.write("Parsing coordinates...")
                    lat, lon = map(float, location_str.split(','))
                    
                    input_data = {
                        "location": (lat, lon),
                        "before_date": str(before_date),
                        "after_date": str(after_date)
                    }
                    
                    st.write("Fetching imagery tiles...")
                    st.write("Running AI detection models...")
                    report = st.session_state.agent_chain.invoke(input_data)
                    status.update(label=" Analysis Complete", state="complete", expanded=False)
                st.success(f"**Target:** {lat}, {lon}")
                st.caption(f"**Period:** {before_date} → {after_date}")
            with col_viz:
                st.write("### Intelligence Report")
                container = st.container(border=True)
                with container:
                    if report:
                        visualizer.display_results(report)
                    else:
                        st.warning("No data returned from the agent.")

        except ValueError:
            st.error(" Invalid Coordinates. Please ensure format is `Lat, Lon` (e.g., 40.7128, -74.0060)")
        except Exception as e:
            st.error(f" An unexpected error occurred: {e}")
    else:
        st.info(" Please configure the parameters in the sidebar and click **Launch Analysis** to begin.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Satellites Active", value="3", delta="Online")
        with col2:
            st.metric(label="Database Records", value="1.2M", delta="+400 today")
        with col3:
            st.metric(label="System Status", value="Operational", delta_color="normal")

if __name__ == "__main__":
    main()