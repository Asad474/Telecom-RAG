import subprocess
import sys
from pathlib import Path


def chat():
    app = Path(__file__).parent / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
