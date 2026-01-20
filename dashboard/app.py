from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import json
import asyncio
import random
from datetime import datetime
import os
import sys
sys.path.append('..')
from api.routes import router
from ev_qa_framework import EVQAFramework, EVBatteryAnalyzer

app = FastAPI(title="EV Battery Monitor", version="1.0.0")

# Include API routes
app.include_router(router, prefix="/api")

# Ensure directories exist
os.makedirs("dashboard/static/css", exist_ok=True)
os.makedirs("dashboard/static/js", exist_ok=True)
os.makedirs("dashboard/templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def telemetry_streamer():
    """Generate realistic telemetry with occasional anomalies"""
    qa = EVQAFramework("Dashboard-Generator")
    
    while True:
        # Generate more realistic data
        base_voltage = 396.0
        base_temp = 35.0
        
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "voltage": round(base_voltage + random.uniform(-10, 10), 2),
            "current": round(random.uniform(50, 150), 2),
            "temperature": round(base_temp + random.uniform(-5, 15), 1),
            "soc": round(random.uniform(70, 90), 1),
            "is_anomaly": False
        }
        
        # 5% chance of anomaly
        if random.random() > 0.95:
            anomaly_type = random.choice(['voltage', 'temperature'])
            if anomaly_type == 'voltage':
                data['voltage'] = round(random.uniform(450, 500), 2)  # High voltage
            else:
                data['temperature'] = round(base_temp + random.uniform(30, 50), 1)  # High temp
            data['is_anomaly'] = True
        
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_streamer())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
