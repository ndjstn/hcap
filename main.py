import os
import sys
import streamlit.cli as stcli

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main()) 