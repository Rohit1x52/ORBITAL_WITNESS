import streamlit.web.cli as stcli
import os
import sys

def main():
    """
    This script is the main entry point to run the Streamlit application.
    It adds the project root to the Python path to ensure imports work.
    """
    # Get the absolute path of the project root directory (ORBITAL_WITNESS)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the system's path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # Define the path to the Streamlit app script
    app_path = os.path.join(project_root, "Interface", "main.py")
    
    # Use the Streamlit command line interface to run the app
    # This directly runs the app without printing any "Usage" message.
    args = ["run", app_path, "--server.port", "8501"]
    stcli.main(args)

if __name__ == "__main__":
    main()
