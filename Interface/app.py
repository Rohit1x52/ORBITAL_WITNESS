# Interface/app.py
import streamlit as st
# --- CHANGE: Import the create_agent function ---
from app.agent import create_satellite_agent
from Interface import visualizer

def main():
    st.set_page_config(page_title="GeoGuardian AI", layout="wide")
    st.title("🛰️ GeoGuardian: Satellite Image Analyzer (LCEL+RAG)") # Updated title
    st.markdown("An AI agent to detect and analyze changes from satellite imagery.")
    
    # --- CHANGE: Initialize the agent chain ---
    if 'agent_chain' not in st.session_state:
        with st.spinner("Initializing AI Agent and Knowledge Base..."):
            st.session_state.agent_chain = create_satellite_agent()

    with st.sidebar:
        st.header("Analysis Parameters")
        location_str = st.text_input("Location (Lat, Lon)", "40.7128, -74.0060")
        before_date = st.date_input("Before Date")
        after_date = st.date_input("After Date")
        
        analyze_button = st.button("Analyze Changes", type="primary")

    if analyze_button:
        try:
            lat, lon = map(float, location_str.split(','))
            
            # --- CHANGE: Prepare input and invoke the chain ---
            input_data = {
                "location": (lat, lon),
                "before_date": str(before_date),
                "after_date": str(after_date)
            }
            
            with st.spinner('Agent is analyzing the area... This may take a moment.'):
                report = st.session_state.agent_chain.invoke(input_data)
                
            visualizer.display_results(report)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()