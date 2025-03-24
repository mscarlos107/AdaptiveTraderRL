# Configuration file for pytest
import os
import sys

# Add the parent directory to sys.path to allow imports from the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))