import hashlib, time, joblib
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.25}, max_ongoing_requests=16,
                  health_check_period_s=5, logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        self.extra = extra_work_ms / 1000.0
        b = joblib.load(path)
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

    async def __call__(self, rows: List[Dict]):
        p = self.model.predict(pd.DataFrame(rows)[self.cols])
        if self.extra: time.sleep(self.extra)
        return {"predictions": [round(float(x), 1) for x in p], "model_version": self.version}

class Delivery(BaseModel):
    distance_km: float; n_stops: int; hour: int; weekday: int
    traffic_index: float; rain: int; weight_kg: float

class Req(BaseModel):
    customer_id: str
    instances: List[Delivery] = Field(min_length=1, max_length=500)

def bucket(customer_id: str) -> float:
    return int(hashlib.sha256(customer_id.encode()).hexdigest()[:8], 16) / 16**8

api = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, v1: DeploymentHandle, v2: DeploymentHandle, weight: float):
        self.h = {"v1": v1, "v2": v2}
        self.w = weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: Req, x_beta: str = Header(default="")):
        arm = "v2" if x_beta == "1" or bucket(req.customer_id) < self.w else "v1"
        out = await self.h[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm}

def build(weight=0.1, extra_work_ms=0.0):
    v1 = EtaModel.options(name="EtaModelV1").bind("data/eta_v1.joblib", extra_work_ms)
    v2 = EtaModel.options(name="EtaModelV2", num_replicas=None, autoscaling_config=dict(
        min_replicas=1, max_replicas=3, target_ongoing_requests=2, upscale_delay_s=5, downscale_delay_s=30,
    )).bind("data/eta_v2.joblib", extra_work_ms)
    return Router.bind(v1, v2, weight)
