import sys
import os

# Make sure the backend root is on the path so main.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
