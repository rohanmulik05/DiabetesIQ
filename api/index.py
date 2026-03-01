import sys
import os

# Make sure the project root is on the path so app.py can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app

# Vercel expects a variable named `app`
# handler is the WSGI callable Vercel uses
handler = app
