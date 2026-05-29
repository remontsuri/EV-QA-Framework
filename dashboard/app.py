"""
Dashboard Application: FastAPI-based real-time telemetry visualization.
Backend hardening: health checks, API auth, CORS, structured logging, graceful shutdown.
"""
import json
import asyncio
import random
import sys
import os
from datetime import datetime, timezone
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Setup system path to include parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pylint: disable=wrong-import-position
from ev_qa_framework import SOHPredictor, setup_logging  # noqa: E402
from api.routes import router  # noqa: E402

logger = setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE") or None,
    json_format=os.getenv("LOG_JSON_FORMAT", "true").lower() == "true",
)

# ── Configuration from environment ────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ── Global state for health / shutdown ─────────────────────────────────────
_ml_model_ready: bool = False
_background_tasks: List[asyncio.Task] = []
_shutdown_event = asyncio.Event()


# ── Lifespan context manager (replaces @app.on_event) ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup tasks and graceful shutdown."""
    logger.info("Application starting up", extra={"environment": ENVIRONMENT})
    # Start background telemetry streamer
    task = asyncio.create_task(telemetry_streamer())
    _background_tasks.append(task)
    yield
    # ── Graceful shutdown ──────────────────────────────────────────────
    logger.info("Shutting down application...")
    _shutdown_event.set()
    # Cancel background tasks with timeout
    for t in _background_tasks:
        t.cancel()
    if _background_tasks:
        await asyncio.wait(_background_tasks, timeout=5.0)
    # Close all WebSocket connections
    await manager.close_all()
    logger.info("Application shutdown complete")


app = FastAPI(title="EV Battery Monitor", version="1.0.0", lifespan=lifespan)

# ── CORS middleware ─────────────────────────────────────────────────────────
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
if ENVIRONMENT == "production" and origins and "*" not in origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured", extra={"allowed_origins": origins})
else:
    # Development: allow all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS: wide open (development mode)", extra={"allowed_origins": origins})

# Include API routes
app.include_router(router, prefix="/api")

# Ensure directories exist
os.makedirs("dashboard/static/css", exist_ok=True)
os.makedirs("dashboard/static/js", exist_ok=True)
os.makedirs("dashboard/templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


# ── API Key auth middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Require X-API-Key header for /api/* routes in production."""
    if ENVIRONMENT == "production" and API_KEY:
        path = request.url.path
        # Open paths: no auth needed
        open_paths = ("/health", "/ready", "/ws", "/static", "/")
        if not path.startswith("/api") and not any(
            path.startswith(p) for p in open_paths
        ):
            return await call_next(request)

        if path.startswith("/api"):
            client_key = request.headers.get("X-API-Key", "")
            if client_key != API_KEY:
                logger.warning(
                    "Unauthorized API access attempt",
                    extra={"path": path, "client_ip": request.client.host if request.client else "unknown"},
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid or missing API key"},
                )
    return await call_next(request)


# ── Health check endpoints ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Basic health check."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready")
async def readiness():
    """Readiness probe — checks if ML model and other services are ready."""
    return {
        "status": "ready" if _ml_model_ready else "not_ready",
        "services": {
            "ml_model": _ml_model_ready,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Connection manager (WebSocket) ─────────────────────────────────────────
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket connected", extra={"count": len(self.active_connections)})

    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass
        logger.debug("WebSocket disconnected", extra={"count": len(self.active_connections)})

    async def close_all(self):
        """Close all active WebSocket connections."""
        for ws in self.active_connections[:]:
            try:
                await ws.close(code=1001, reason="Server shutting down")
            except Exception:
                pass
        self.active_connections.clear()
        logger.info("All WebSocket connections closed")

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections[:]:
            try:
                await connection.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                self.disconnect(connection)


manager = ConnectionManager()


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket endpoint."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for keepalive if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


async def telemetry_streamer():
    """
    Generate realistic telemetry with occasional anomalies
    and ML-based SOH prediction.
    """
    global _ml_model_ready

    # Simulation data for SOH predictor
    df_history = pd.DataFrame({
        'voltage': [396.0] * 20,
        'current': [100.0] * 20,
        'temperature': [35.0] * 20,
        'soh': np.linspace(100, 99.8, 20)
    })

    predictor = SOHPredictor(sequence_length=10)
    try:
        predictor.train(df_history, epochs=5)
        _ml_model_ready = True
        logger.info("SOH predictor trained successfully")
    except Exception as e:
        logger.error("SOH predictor training failed", extra={"error": str(e)})
        _ml_model_ready = False

    current_soh = 99.8

    while not _shutdown_event.is_set():
        base_voltage = 396.0
        base_temp = 35.0

        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "voltage": round(base_voltage + random.uniform(-5, 5), 2),
            "current": round(random.uniform(50, 150), 2),
            "temperature": round(base_temp + random.uniform(-2, 5), 1),
            "soc": round(random.uniform(70, 90), 1),
            "soh": round(current_soh, 2),
            "is_anomaly": False,
        }

        # 5% chance of anomaly
        if random.random() > 0.95:
            anomaly_type = random.choice(['voltage', 'temperature'])
            if anomaly_type == 'voltage':
                data['voltage'] = round(random.uniform(950, 1000), 2)
            else:
                data['temperature'] = round(base_temp + random.uniform(30, 45), 1)
            data['is_anomaly'] = True

        # Update history for prediction
        new_row = pd.DataFrame([data])[['voltage', 'current', 'temperature', 'soh']]
        df_history = pd.concat([df_history, new_row]).tail(20)

        # Predict SOH degradation (very slowly)
        try:
            predictor.predict_next(df_history)
            degradation = 0.001 if data['temperature'] < 45 else 0.005
            current_soh -= degradation
        except (ValueError, RuntimeError):
            pass

        await manager.broadcast(json.dumps(data))
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            logger.info("Telemetry streamer cancelled")
            break


# ── Command-line entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting EV Battery Monitor",
        extra={"host": HOST, "port": PORT, "environment": ENVIRONMENT},
    )
    uvicorn.run(
        "dashboard.app:app",
        host=HOST,
        port=PORT,
        reload=(ENVIRONMENT != "production"),
        log_level="info",
    )
