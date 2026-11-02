import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Tests must be deterministic and offline: never use a real key.
os.environ["GEMINI_API_KEY"] = ""
