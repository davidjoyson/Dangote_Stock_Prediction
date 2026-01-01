import runpy
import os
import sys

# Ensure 'src' is on sys.path so package imports work when executing the script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Execute the main script now at src/main.py
runpy.run_path(os.path.join(os.path.dirname(__file__), 'src', 'main.py'), run_name='__main__')
