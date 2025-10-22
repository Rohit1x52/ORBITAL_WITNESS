import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "app":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", "interface/streamlit_app.py"]
        sys.exit(stcli.main())
    else:
        print("\nUsage:")
        print("  python run.py app     # Launch Streamlit interface\n")