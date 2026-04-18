# Traffic Nexus — PRD (UPDATED 2026-01-18)

## Evolution
- **Iteration 1:** Build full-stack project, replace fake HTML with real FastAPI+PyTorch+React+Three.js+Unity
- **Iteration 2:** Fix V2X panel display (REST polling fallback)
- **Iteration 3:** Fix OSM live fetching + wire OSM map into running sim (and into 3D + Unity)

## Verified state (all tests passing across 3 iterations)

### Backend (/app/backend)
- `server.py` — FastAPI + WebSocket + async training thread
- `simulator.py` — `MultiIntersectionEnv` (queue, for DQN) + `VehicleSim` (per-vehicle) + **`VehicleSim.load_from_osm()` rebuilds sim on real OSM network**
- `ml.py` — PyTorch shared-DQN + ForecastNet + replay buffer + target net
- `osm.py` — Nominatim + **Photon fallback** geocoding + 5 Overpass mirrors + MongoDB cache + offline snapshots
- `v2x.py` — real messages from phase transitions / queue deltas / emergency proximity
- `unity_bridge.py` — WebSocket ConnectionManager + documented JSON protocol

### Frontend (/app/frontend)
- Hero + 6 KPI cards
- 2D sim canvas (WebSocket-driven)
- Raw Three.js 3D viewer (auto-rebuilds on city change)
- ML training dashboard
- **OSM panel with both "Preview Import" and "Simulate City in 3D" buttons**
- Live V2X log

### Unity client (/app/unity_client)
- `TrafficNexusBridgeClient.cs` — consumes same WebSocket protocol as React
- **Auto-detects new city layouts and rebuilds the Unity scene** (roads, traffic lights, vehicles)
- Client→Server commands: pause, resume, set_policy

## Real data proof points (verified today)
- OSM live fetch: Koramangala, Bengaluru → **3,556 nodes, 7,317 edges, 46 signals** via `overpass-api.de`
- After load_sim: sim running with **151 OSM nodes, 134 edges, 15 real traffic signals, 193 vehicles**
- DQN comparison: learned (−52) beats pressure (−71) beats fixed (−85)
- V2X log showing real OSM node IDs in `ROUTE_INTENT` messages

## Next tasks / backlog
- Add real-map tile background under the Three.js scene (optional Mapbox/OSM tiles)
- Add time-series chart for throughput/queue over simulation time
- Keep connectivity higher when subsampling large OSM graphs (currently drops isolated nodes)
- Export rollout videos directly from backend
