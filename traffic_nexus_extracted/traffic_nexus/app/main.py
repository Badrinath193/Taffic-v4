from __future__ import annotations

import re
from pathlib import Path
from typing import Dict
import csv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ml import train_shared_dqn, evaluate_policy
from .osm import geocode, build_query, query_overpass
from .simulator import MultiIntersectionEnv


BASE = Path(__file__).resolve().parents[1]
ART = BASE / "artifacts"
FRONT = BASE / "frontend"
ART.mkdir(exist_ok=True)

app = FastAPI(title="Traffic Nexus")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONT)), name="static")


class TrainRequest(BaseModel):
    episodes: int = 40
    seed: int = 7


class GeocodeRequest(BaseModel):
    place: str
    radius: int = 1800


@app.get("/")
def root():
    return FileResponse(str(FRONT / "index.html"))


@app.get("/api/network/synthetic")
def synthetic_network(rows: int = 2, cols: int = 2):
    env = MultiIntersectionEnv(rows=rows, cols=cols)
    return env.export_network()


@app.post("/api/ml/train")
def train(req: TrainRequest):
    result = train_shared_dqn(str(ART), episodes=req.episodes, seed=req.seed)
    eval_metrics = evaluate_policy(result.model_path, result.forecast_path)
    return {
        "model_path": result.model_path,
        "forecast_path": result.forecast_path,
        "metrics_path": result.metrics_path,
        "reward_tail": result.rewards[-5:],
        "queue_tail": result.queues[-5:],
        "fairness_tail": result.fairness[-5:],
        "evaluation": eval_metrics,
    }


@app.get("/api/ml/metrics")
def metrics():
    path = ART / "training_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Train the model first")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


@app.post("/api/osm/import")
def osm_import(req: GeocodeRequest):
    if not re.match(r'^[a-zA-Z0-9\s,\.\-]+$', req.place):
        raise HTTPException(status_code=400, detail="Invalid characters in place string")
    try:
        loc = geocode(req.place)
        query = build_query(loc["lat"], loc["lon"], req.radius)
        data = query_overpass(query)
        return {"location": loc, "overpass_endpoint": data["endpoint"], "raw": data["data"]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
