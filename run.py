#!/usr/bin/env python3
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now run streamlit
if __name__ == "__main__":
    os.system(f"streamlit run {os.path.join(project_root, 'src', 'app.py')}") 