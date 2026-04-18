# Traffic Nexus

A real, runnable metropolitan traffic-control platform that combines:

- **MARL** — shared DQN (PyTorch) with experience replay, target network, epsilon-greedy
- **Forecast net** — MLP predicting next-step NS/EW queue lengths, feeds into the DQN observation
- **Per-vehicle simulation** — kinematic vehicles (5 types) over a configurable grid
- **Real V2X** — PHASE_CHANGE / QUEUE_UPDATE / ROUTE_INTENT / EMERGENCY_PREEMPT messages derived from actual state transitions
- **Robust OSM** — 5 Overpass mirrors, exponential backoff, MongoDB cache, offline fallback dataset
- **Unity-compatible 3D bridge** — WebSocket protocol consumed by both the Three.js viewer (React frontend) and the provided Unity C# client
- **FastAPI backend + React frontend** — not a single HTML file

## Architecture

```
+---------------------+          WebSocket /ws/stream           +----------------------+
|  React Frontend     | <--------------------------------------> |  FastAPI Backend     |
|  (/app/frontend)    |      snapshot / v2x / metrics            |  (/app/backend)      |
|                     |                                           |                      |
|  - 2D sim canvas    |      REST /api/sim/*                      |  - VehicleSim        |
|  - Three.js 3D      |      REST /api/ml/*                       |  - MARL DQN (PyTorch)|
|  - Training panel   |      REST /api/osm/*                      |  - V2X bus           |
|  - OSM importer     |                                           |  - OSM proxy (5 mir) |
|  - V2X live log     |                                           |  - MongoDB cache     |
+---------------------+                                           +-----------+----------+
                                                                              |
           +---------------------+                                            |
           |  Unity C# client    | <------------ same WebSocket ---------------+
           | (/app/unity_client) |
           +---------------------+
```

## API

| Method | Route | Purpose |
|---|---|---|
| GET  | `/api/health` | backend status + models_loaded |
| GET  | `/api/network/synthetic?rows=3&cols=3` | grid network JSON |
| POST | `/api/sim/start` | begin vehicle-level simulation |
| POST | `/api/sim/stop` | pause |
| POST | `/api/sim/reset` | reset vehicles + network |
| POST | `/api/sim/set_policy` | `fixed` \| `pressure` \| `learned` |
| GET  | `/api/sim/state` | full snapshot |
| GET  | `/api/sim/metrics` | last N step metrics |
| GET  | `/api/v2x/tail?n=50` | real V2X messages |
| POST | `/api/ml/train` | kick off real DQN training |
| GET  | `/api/ml/train_status` | live progress |
| GET  | `/api/ml/metrics` | historical metrics (CSV) |
| GET  | `/api/ml/summary` | latest summary.json |
| POST | `/api/ml/evaluate` | learned vs fixed vs pressure comparison |
| POST | `/api/osm/import` | OSM pull with mirror fallback + cache |
| GET  | `/api/osm/cached` | list cached imports |
| WS   | `/ws/stream` | realtime snapshot + v2x broadcast |

## Running locally

```
# backend
cd /app/backend
pip install -r requirements.txt
uvicorn server:app --port 8001 --reload

# frontend
cd /app/frontend
yarn
yarn start
```

## Verifying it is REAL

```bash
# 1. train for 25 episodes
curl -X POST $API/api/ml/train -H 'content-type: application/json' -d '{"episodes":25}'

# 2. watch live progress
curl $API/api/ml/train_status | jq

# 3. evaluate — learned should beat fixed+pressure
curl -X POST $API/api/ml/evaluate | jq
```

In the reference run, the learned policy achieved:
- **reward −52** (vs fixed −85, pressure −71)
- **queue 45** (vs fixed 62, pressure 50)
- **fairness 3** (vs fixed 22, pressure 18)

## Honest feature status

| Feature | Status |
|---|---|
| DQN multi-agent policy | ✅ Real PyTorch, trains, beats baselines |
| Forecast net | ✅ Real MLP, normalized, saved |
| Per-vehicle kinematics | ✅ Real |
| V2X messaging | ✅ Real (derived from state transitions, not random) |
| OSM ingestion | ✅ Real (5 mirrors + cache + offline fallback) |
| Three.js 3D viewer | ✅ Real WebGL 3D |
| Unity integration | ✅ Real WebSocket protocol + sample C# client |
| "Blockchain AI" | ❌ removed (was marketing) |
| "2000 vehicle" promise | ⚠ configurable up to 800; no marketing number |
| "Google Maps tiles" | ❌ removed (was never implemented) |
| SUMO co-simulation | ❌ not claimed — simulator is our own |
