import os, time, uuid, joblib
import pandas as pd
from typing import List
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, model_validator

bundle = joblib.load(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
model, FEATURES, VERSION = bundle["model"], bundle["feature_columns"], bundle["version"]

REGISTRY = CollectorRegistry()
REQUESTS = Counter("eta_requests", "Requests by path and status", ["path", "status"], registry=REGISTRY)
LATENCY = Histogram("eta_request_latency_seconds", "Request latency", registry=REGISTRY)

class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible: distance_km > 150 with n_stops > 20")
        return self

class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)

app = FastAPI(title="SwiftParcel ETA service")

@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse({"error": "internal_error", "request_id": request_id}, status_code=500)
    response.headers["X-Request-ID"] = request_id
    if request.url.path != "/metrics":
        REQUESTS.labels(path=request.url.path, status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter() - t0)
    return response

@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(map(str, e["loc"])), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse({"error": "validation_failed", "details": details, "request_id": request.state.request_id},
                        status_code=422)

@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
    return {"predictions": [round(float(p), 1) for p in model.predict(X)],
            "model_version": VERSION, "request_id": request.state.request_id}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
