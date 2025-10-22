# run.py
import streamlit.web.cli as stcli
import os
import sys

def main():
    """
    This script is the main entry point to run the Streamlit application.
    It tells Streamlit to run the 'app.py' file located inside the 'Interface' folder.
    """
    
    # Get the directory where this run.py script is located
    current_dir = os.path.dirname(__file__)
    
    # Define the path to the Streamlit app script
    app_path = os.path.join(current_dir, "Interface", "app.py")
    
    # Add the project's root directory to the Python path
    # This allows 'Interface/app.py' to import modules from the 'app/' directory
    sys.path.insert(0, current_dir)

    # Check if the app.py file exists before trying to run it
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        print("Please make sure your project structure is correct.")
        return

    # These are the command line arguments to pass to Streamlit
    # It's the same as running: streamlit run Interface/app.py
    args = ["run", app_path]
    
    # Launch Streamlit
    stcli.main(args)

if __name__ == "__main__":
    main()
