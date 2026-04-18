# Traffic Nexus — PRD

## Original problem statement
User delivered two artifacts:
1. `traffic-signal-nexus-fixed1.html` — a single-file browser demo that **claimed** MARL, Unity3D, V2X, Google Maps integration but was all UI theatre (no ML, random V2X strings, timer-only signals).
2. `traffic_nexus_local_upgrade.zip` — a minimal real FastAPI + PyTorch project.

User goal: turn every fake feature into real, runnable code in a proper full-stack project structure (frontend + backend + AI). Specific asks:
- Real MARL neural network (DQN)
- Real OSM with reliable Overpass (user noted overload failures)
- Real Unity integration
- Real V2X communication
- Not a single HTML file — full project folder

## Architecture
- `/app/backend` — FastAPI + PyTorch + MongoDB + WebSocket
- `/app/frontend` — React + raw Three.js (r172) for real 3D
- `/app/unity_client` — Unity C# sample client for WebSocket bridge

## Core tech stack
- PyTorch shared-DQN (real MARL)
- ForecastNet MLP (real queue prediction)
- Overpass with 5 mirrors + exponential backoff + MongoDB cache + offline fallback
- WebSocket protocol shared by React + Unity
- Real V2X bus derived from simulator state transitions

## Users
- Researchers validating MARL vs baselines
- Product/demo audiences wanting live 3D traffic visualization
- Developers wanting to connect Unity to a running simulator

## Implemented (2026-01-18)
- `simulator.py` — `MultiIntersectionEnv` (queue) and `VehicleSim` (per-vehicle, 5 types)
- `ml.py` — ForecastNet, QNet, replay buffer, target network, epsilon-greedy DQN
- `osm.py` — 5 Overpass mirrors + Nominatim fallback + offline snapshot for Chennai & Bengaluru
- `v2x.py` — real messages from phase changes, queue deltas, emergency proximity, route intents
- `unity_bridge.py` — WebSocket ConnectionManager with JSON protocol
- `server.py` — REST + WS + async training thread
- React frontend: hero, KPIs, 2D sim canvas, real Three.js 3D viewer, ML training dashboard, OSM importer, V2X live log, architecture section
- Unity sample client with protocol spec and README

## Verified end-to-end
- Seeded DQN beats baselines: learned reward -52 vs pressure -71 vs fixed -85
- Simulation runs live with WS broadcast
- OSM import tries live first, falls back to offline data gracefully
- Frontend renders with live vehicle counts, signal phases, step counter

## Backlog / future work
- Sora-like weather effects in 3D scene
- Per-vehicle emissions model calibrated to real data
- Export rollout videos directly from the backend
- Add A2C / PPO alternatives for apples-to-apples comparison
- Long-horizon evaluation (200+ episode training)

## Next tasks
- Investigate V2X live log not populating on frontend (WebSocket frames might not be pushing v2x through)
- Add chart for live sim metrics (throughput over time, stopped vehicles, emissions) — partial chart exists for training history
- Write automated pytest suite matching the original zip's test pattern
