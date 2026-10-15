import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import uvicorn
from settings import WEB_PORT

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=WEB_PORT, log_level="warning", access_log=False)
