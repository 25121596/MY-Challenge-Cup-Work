"""Cloud server entry point.

Start with::

    python main.py

or::

    uvicorn main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the cloud_server package (parent dir of this file) is importable.
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("cloud_server")


# ── Lifespan ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    from inference.model_manager import model_manager

    await model_manager.initialize()
    logger.info(
        "Cloud server started on %s:%s (backend=%s)",
        config.host,
        config.port,
        config.model_backend,
    )
    yield
    await model_manager.close()
    logger.info("Cloud server shut down")


# ── App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Cloud Edge Server",
    description="Cloud-side service for edge-cloud collaborative inference",
    version=config.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow edge devices and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routes ──────────────────────────────────────────────
from api.health import router as health_router
from api.heartbeat import router as heartbeat_router
from api.tasks import router as tasks_router
from api.inference import router as inference_router
from api.chat import router as chat_router
from api.model_updates import router as model_updates_router
from api.neighbors import router as neighbors_router
from api.conflicts import router as conflicts_router

app.include_router(health_router)
app.include_router(heartbeat_router)
app.include_router(tasks_router)
app.include_router(inference_router)
app.include_router(chat_router)
app.include_router(model_updates_router)
app.include_router(neighbors_router)
app.include_router(conflicts_router)


# ── CLI entry ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="debug" if config.debug else "info",
    )
