"""FastAPI entrypoint."""

from __future__ import annotations

from collections.abc import Awaitable, AsyncIterator, Callable

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from api.deps import load_state
from api.routers import analytics, customers, datasets, predict
from src.utils.logging_setup import get_logger

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and connect/seed the database once at startup (a test may pre-set app.state.s)."""
    if not hasattr(app.state, "s"):
        app.state.s = load_state()
    log.info("api ready")
    yield


app = FastAPI(title="Vantara Customer Behavior Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """INFO log per request with latency."""
    t = time.perf_counter()
    resp = await call_next(request)
    log.info(
        "request method=%s path=%s status=%d ms=%.1f",
        request.method,
        request.url.path,
        resp.status_code,
        (time.perf_counter() - t) * 1000,
    )
    return resp


@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Turn model-input ValueErrors into 422 responses."""
    log.warning("validation failure: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


app.include_router(analytics.router)
app.include_router(customers.router)
app.include_router(datasets.router)
app.include_router(predict.router)
