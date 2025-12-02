from __future__ import annotations

import threading

import uvicorn

from api import app as fastapi_app
from notebooks.gradio_ui import main as launch_gradio


def run_api() -> None:
    """
    Démarre l'API FastAPI (backend) sur le port 8000.
    """
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")


def run_gradio() -> None:
    """
    Lance l'interface Gradio qui interroge l'API.
    """
    launch_gradio()


if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    run_gradio()

