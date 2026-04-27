"""Traffic Nexus — FastAPI server.

Real features:
  - /api/network/synthetic : returns grid network
  - /api/sim/start|stop|step|state : live vehicle-level simulation
  - /api/sim/set_policy : switch between fixed / pressure / learned
  - /api/ml/train : starts REAL DQN training (async, streams progress)
  - /api/ml/train_status : current progress
  - /api/ml/metrics : historical training metrics
  - /api/ml/evaluate : compare learned vs fixed vs pressure
  - /api/osm/import : OSM with 5 mirrors + Mongo cache + offline fallback
  - /api/v2x/tail : last N real V2X messages
  - WS /ws/stream : realtime snapshot + V2X stream (Unity + Three.js protocol)
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ml import (
    DEVICE,
    attach_forecast,
    evaluate_policy,
    load_forecast,
    load_qnet,
    train_shared_dqn,
)
from osm import import_osm
from simulator import MultiIntersectionEnv, VehicleSim
from unity_bridge import ConnectionManager
from v2x import V2XBus


load_dotenv(Path(__file__).parent / ".env")

ART_DIR = Path(__file__).parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

# Parse ALLOWED_ORIGINS from env, fallback to defaults for local development
allowed_origins_str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8001"
)
allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]

app = FastAPI(title="Traffic Nexus", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]


# ---------------- Simulation state ----------------

class SimRuntime:
    def __init__(self):
        self.sim: Optional[VehicleSim] = None
        self.v2x: Optional[V2XBus] = None
        self.running: bool = False
        self.policy: str = "fixed"   # fixed | pressure | learned
        self.q = None
        self.forecast = None
        self.forecast_blob = None
        self.loop_task: Optional[asyncio.Task] = None

    def reset(self, rows: int = 3, cols: int = 3, max_vehicles: int = 250, seed: int = 11):
        self.sim = VehicleSim(rows=rows, cols=cols, max_vehicles=max_vehicles, seed=seed)
        self.v2x = V2XBus(self.sim, max_log=200)
        self.running = False

    def ensure_models(self):
        model_path = ART_DIR / "shared_dqn.pt"
        forecast_path = ART_DIR / "forecast_model.pt"
        if model_path.exists() and forecast_path.exists():
            if self.q is None:
                try:
                    self.q = load_qnet(str(model_path))
                    self.forecast, self.forecast_blob = load_forecast(str(forecast_path))
                except Exception as exc:
                    print("[warn] could not load models:", exc)

    def decide(self) -> Dict[str, int]:
        if self.sim is None:
            return {}
        if self.policy == "fixed":
            return {nid: (self.sim.step_id // 30) % 2 for nid in self.sim.tls.keys()}
        if self.policy == "pressure":
            out = {}
            for nid in self.sim.tls.keys():
                ob = self.sim.intersection_obs(nid)
                out[nid] = 0 if ob[0] >= ob[1] else 1
            return out
        # learned
        self.ensure_models()
        if self.q is None or self.forecast is None:
            # fall back to pressure
            out = {}
            for nid in self.sim.tls.keys():
                ob = self.sim.intersection_obs(nid)
                out[nid] = 0 if ob[0] >= ob[1] else 1
            return out
        obs = {nid: self.sim.intersection_obs(nid) for nid in self.sim.tls.keys()}
        obs = attach_forecast(self.forecast, self.forecast_blob, obs)
        out = {}
        with torch.no_grad():
            for nid, ob in obs.items():
                qv = self.q(torch.tensor(ob, dtype=torch.float32, device=DEVICE))
                out[nid] = int(torch.argmax(qv).item())
        return out


RT = SimRuntime()
RT.reset()
WS = ConnectionManager()

# Train status (thread -> main)
TRAIN_STATUS: Dict = {"state": "idle", "events": []}
TRAIN_LOCK = threading.Lock()


# ---------------- Models ----------------

class TrainRequest(BaseModel):
    episodes: int = 30
    seed: int = 7


class SimStartRequest(BaseModel):
    rows: int = 3
    cols: int = 3
    max_vehicles: int = 250
    seed: int = 11


class PolicyRequest(BaseModel):
    policy: str   # fixed | pressure | learned


class OSMRequest(BaseModel):
    place: str
    radius: int = 1500


class OSMLoadSimRequest(BaseModel):
    place: str
    radius: int = 1500
    max_nodes: int = 600
    max_vehicles: int = 300
    autostart: bool = True


# ---------------- Endpoints ----------------

@app.get("/api/health")
async def health():
    return {"ok": True, "models_loaded": RT.q is not None}


@app.get("/api/network/synthetic")
async def synthetic_network(rows: int = 3, cols: int = 3):
    env = MultiIntersectionEnv(rows=rows, cols=cols)
    return env.export_network()


# ---- Sim control ----

@app.post("/api/sim/start")
async def sim_start(req: SimStartRequest):
    RT.reset(rows=req.rows, cols=req.cols, max_vehicles=req.max_vehicles, seed=req.seed)
    RT.running = True
    if RT.loop_task is None or RT.loop_task.done():
        RT.loop_task = asyncio.create_task(_sim_loop())
    return {"ok": True, "network": RT.sim.export_network()}


@app.post("/api/sim/pause")
async def sim_pause():
    """Pause the sim without tearing down the world — keeps OSM map, vehicles, signals."""
    RT.running = False
    return {"ok": True, "running": False}


@app.post("/api/sim/resume")
async def sim_resume():
    """Resume without resetting — preserves OSM map, vehicles, and training artifacts."""
    if RT.sim is None:
        RT.reset()
    RT.running = True
    if RT.loop_task is None or RT.loop_task.done():
        RT.loop_task = asyncio.create_task(_sim_loop())
    return {"ok": True, "running": True, "source": getattr(RT.sim, "_source", "synthetic-grid")}


@app.post("/api/sim/stop")
async def sim_stop():
    RT.running = False
    return {"ok": True}


@app.post("/api/sim/reset")
async def sim_reset(req: SimStartRequest):
    was_running = RT.running
    RT.reset(rows=req.rows, cols=req.cols, max_vehicles=req.max_vehicles, seed=req.seed)
    RT.running = was_running
    return {"ok": True, "network": RT.sim.export_network()}


@app.post("/api/sim/set_policy")
async def sim_set_policy(req: PolicyRequest):
    if req.policy not in ("fixed", "pressure", "learned"):
        raise HTTPException(status_code=400, detail="Invalid policy")
    RT.policy = req.policy
    RT.ensure_models()
    model_loaded = RT.q is not None
    return {"ok": True, "policy": RT.policy, "learned_available": model_loaded}


@app.get("/api/sim/state")
async def sim_state():
    if RT.sim is None:
        return {"running": False}
    snap = RT.sim.snapshot()
    snap["running"] = RT.running
    snap["policy"] = RT.policy
    snap["metrics"] = RT.sim.metrics_history[-1] if RT.sim.metrics_history else None
    return snap


@app.get("/api/sim/metrics")
async def sim_metrics(limit: int = 200):
    if RT.sim is None:
        return []
    return RT.sim.metrics_history[-limit:]


# ---- V2X ----

@app.get("/api/v2x/tail")
async def v2x_tail(n: int = 50):
    if RT.v2x is None:
        return []
    return RT.v2x.tail(n)


# ---- ML ----

def _train_thread(episodes: int, seed: int):
    with TRAIN_LOCK:
        TRAIN_STATUS["state"] = "running"
        TRAIN_STATUS["events"] = []
        TRAIN_STATUS["episodes"] = episodes

    def cb(ev: Dict):
        with TRAIN_LOCK:
            TRAIN_STATUS["events"].append(ev)
            if len(TRAIN_STATUS["events"]) > 400:
                TRAIN_STATUS["events"] = TRAIN_STATUS["events"][-400:]

    try:
        result = train_shared_dqn(str(ART_DIR), episodes=episodes, seed=seed, progress_cb=cb)
        # reload into runtime
        RT.q = load_qnet(result.model_path)
        RT.forecast, RT.forecast_blob = load_forecast(result.forecast_path)
        evaluation = evaluate_policy(result.model_path, result.forecast_path, episodes=2)
        with TRAIN_LOCK:
            TRAIN_STATUS["state"] = "done"
            TRAIN_STATUS["result"] = {
                "rewards_tail": result.rewards[-10:],
                "queues_tail": result.queues[-10:],
                "fairness_tail": result.fairness[-10:],
                "evaluation": evaluation,
            }
    except Exception as exc:
        with TRAIN_LOCK:
            TRAIN_STATUS["state"] = "error"
            TRAIN_STATUS["error"] = str(exc)


@app.post("/api/ml/train")
async def ml_train(req: TrainRequest):
    with TRAIN_LOCK:
        if TRAIN_STATUS["state"] == "running":
            raise HTTPException(status_code=409, detail="A training run is already in progress")
    t = threading.Thread(target=_train_thread, args=(req.episodes, req.seed), daemon=True)
    t.start()
    return {"ok": True, "episodes": req.episodes}


@app.get("/api/ml/train_status")
async def ml_train_status():
    with TRAIN_LOCK:
        return dict(TRAIN_STATUS)


@app.get("/api/ml/metrics")
async def ml_metrics():
    path = ART_DIR / "training_metrics.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if k != "episode" else int(float(v)) for k, v in r.items()})
    return rows


@app.get("/api/ml/summary")
async def ml_summary():
    path = ART_DIR / "summary.json"
    if not path.exists():
        return {"trained": False}
    with open(path) as f:
        blob = json.load(f)
    blob["trained"] = True
    return blob


@app.post("/api/ml/evaluate")
async def ml_evaluate():
    model_path = ART_DIR / "shared_dqn.pt"
    forecast_path = ART_DIR / "forecast_model.pt"
    if not model_path.exists() or not forecast_path.exists():
        raise HTTPException(status_code=404, detail="Train the model first")
    result = evaluate_policy(str(model_path), str(forecast_path), episodes=2)
    return result


# ---- OSM ----

async def _cache_get(key: str):
    doc = await db.osm_cache.find_one({"_id": key}, {"_id": 0})
    return doc["value"] if doc else None


async def _cache_put(key: str, value: Dict):
    await db.osm_cache.update_one({"_id": key}, {"$set": {"value": value}}, upsert=True)


@app.post("/api/osm/import")
async def osm_import(req: OSMRequest):
    # cache is async so we wrap sync calls
    try:
        ck_hit = await _cache_get(f"osm:{req.place.lower()}|{req.radius}")
    except Exception as exc:
        print("[warn] cache get failed:", exc)
        ck_hit = None
    if ck_hit:
        ck_hit["cache"] = "hit"
        return ck_hit
    try:
        result = import_osm(req.place, req.radius)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        await _cache_put(f"osm:{req.place.lower()}|{req.radius}", result)
    except Exception as exc:
        print("[warn] cache put failed:", exc)
    return result


@app.get("/api/osm/cached")
async def osm_cached():
    docs = await db.osm_cache.find({}, {"value.place": 1, "value.radius": 1, "value.source": 1}).to_list(length=50)
    return [d.get("value", {}) for d in docs]


@app.post("/api/osm/load_sim")
async def osm_load_sim(req: OSMLoadSimRequest):
    """Import OSM for a place and load it into the running simulator.

    The live WebSocket broadcast immediately reflects the new city graph,
    so both the React Three.js viewer and any connected Unity client
    re-render the world using the real OSM nodes, edges and traffic signals.
    """
    # 1. reuse cache if available
    try:
        ck_hit = await _cache_get(f"osm:{req.place.lower()}|{req.radius}")
    except Exception as exc:
        print("[warn] cache get failed:", exc)
        ck_hit = None
    if ck_hit:
        result = ck_hit
        result["cache"] = "hit"
    else:
        try:
            result = await run_in_threadpool(import_osm, req.place, req.radius)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        try:
            await _cache_put(f"osm:{req.place.lower()}|{req.radius}", result)
        except Exception as exc:
            print("[warn] cache put failed:", exc)

    graph = result.get("graph") or {}
    if not graph.get("nodes"):
        raise HTTPException(status_code=422, detail="OSM import returned an empty graph")

    # 2. rebuild the sim on the OSM network
    if RT.sim is None:
        RT.reset(max_vehicles=req.max_vehicles)
    try:
        info = RT.sim.load_from_osm(graph, max_nodes=req.max_nodes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"load_from_osm failed: {exc}")
    RT.sim.max_vehicles = req.max_vehicles
    # v2x bus must be rebuilt to reference the new sim
    RT.v2x = V2XBus(RT.sim, max_log=200)

    # 3. start loop if requested
    if req.autostart:
        RT.running = True
        if RT.loop_task is None or RT.loop_task.done():
            RT.loop_task = asyncio.create_task(_sim_loop())

    # 4. immediately broadcast the new layout so clients can rebuild geometry
    snap = RT.sim.snapshot()
    await WS.broadcast({
        "type": "snapshot",
        "snapshot": snap,
        "v2x": [],
        "metrics": None,
        "policy": RT.policy,
        "osm_loaded": True,
        "place": req.place,
        "center": result.get("location"),
    })

    return {
        "ok": True,
        "place": req.place,
        "location": result.get("location"),
        "loaded": info,
        "running": RT.running,
        "source": result.get("source"),
    }


# ---- Simulation loop ----

async def _sim_loop():
    tick_hz = 8
    dt = 1.0 / tick_hz
    broadcast_every = 2   # broadcast 4 times/sec
    step_cnt = 0
    while True:
        await asyncio.sleep(dt)
        if not RT.running or RT.sim is None:
            continue
        try:
            phases = RT.decide()
            RT.sim.set_phases(phases)
            RT.sim.step(dt=dt * 2.0, spawn_rate=1.1)
            RT.v2x.tick()
        except Exception as exc:
            print("[sim_loop]", exc)
            continue
        step_cnt += 1
        if step_cnt % broadcast_every == 0:
            snap = RT.sim.snapshot()
            v2x_tail = RT.v2x.tail(10)
            await WS.broadcast({
                "type": "snapshot",
                "snapshot": snap,
                "v2x": v2x_tail,
                "metrics": RT.sim.metrics_history[-1] if RT.sim.metrics_history else None,
                "policy": RT.policy,
            })


# ---- WebSocket ----

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await WS.connect(ws)
    try:
        await ws.send_json({"type": "hello", "msg": "Traffic Nexus Unity Bridge v2.0"})
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "cmd":
                    act = msg.get("action")
                    if act == "pause":
                        RT.running = False
                    elif act == "resume":
                        RT.running = True
                    elif act == "set_policy":
                        RT.policy = msg.get("value", RT.policy)
                        RT.ensure_models()
                    await ws.send_json({"type": "ack", "action": act})
            except Exception as exc:
                await ws.send_json({"type": "error", "detail": str(exc)})
    except WebSocketDisconnect:
        await WS.disconnect(ws)


# ---- Lifespan ----

@app.on_event("startup")
async def on_startup():
    RT.ensure_models()
    print("[startup] models loaded:", RT.q is not None)
    # start sim loop running but paused until /api/sim/start
    if RT.loop_task is None:
        RT.loop_task = asyncio.create_task(_sim_loop())


@app.on_event("shutdown")
async def on_shutdown():
    RT.running = False
    if RT.loop_task:
        RT.loop_task.cancel()
    mongo_client.close()
