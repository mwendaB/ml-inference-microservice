from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.routes import router, _engine
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

config = ConfigLoader("config/api.yaml").get().get("api", {})

app = FastAPI(title=config.get("title", "Detection API"), version=config.get("version", "v1"))

if config.get("enable_cors", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )


@app.middleware("http")
async def security_guard(request: Request, call_next):
    api_keys = config.get("api_keys", [])
    if api_keys:
        header = config.get("api_key_header", "x-api-key")
        provided = request.headers.get(header)
        if provided not in api_keys:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    max_upload_mb = config.get("max_upload_mb", 25)
    content_length = request.headers.get("content-length")
    if content_length:
        if int(content_length) > max_upload_mb * 1024 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
    return await call_next(request)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


app.include_router(router)


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            result = await _engine.infer_async(data)
            await ws.send_json(result)
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
