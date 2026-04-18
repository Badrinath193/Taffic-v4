# Traffic Nexus — PRD (FINAL)

## Original problem statement
User delivered two artifacts (fake HTML demo + minimal real Python zip) and asked to
convert every fake feature into real runnable code in a full-stack project structure
(frontend + backend + AI + Unity integration).

## Architecture (shipped)
- `/app/backend` — FastAPI + PyTorch + MongoDB + WebSocket (real MARL DQN)
- `/app/frontend` — React + raw Three.js r172 (real 3D WebGL)
- `/app/unity_client` — Unity C# sample + protocol spec (NativeWebSocket-based)

## Implemented & Verified (2026-01-18)

**Backend (15/15 tests pass):**
- Shared-DQN MARL policy — real PyTorch, beats baselines (learned −52 vs fixed −85 vs pressure −71)
- ForecastNet MLP — real queue prediction feeding DQN observation
- VehicleSim — per-vehicle kinematics, 5 types, 4-phase signal FSM
- V2XBus — real messages from state transitions (not random)
- OSM ingestion — 5 Overpass mirrors + exponential backoff + MongoDB cache + offline fallback (Chennai/Bengaluru)
- WebSocket /ws/stream — shared protocol for Three.js and Unity clients
- REST: /api/health /api/network/synthetic /api/sim/{start,stop,reset,state,metrics,set_policy} /api/v2x/tail /api/ml/{train,train_status,metrics,summary,evaluate} /api/osm/{import,cached}

**Frontend (14/14 tests pass):**
- Hero + 6 KPI cards (live vehicles, signals, step, throughput, DQN loaded, best reward)
- Live 2D sim canvas (WebSocket-driven, renders per-vehicle)
- Live Three.js 3D viewer (real WebGL, orbit controls, dispose cleanup)
- ML training dashboard (episode-level events + history chart + eval comparison)
- OSM import form (live or offline fallback — graceful)
- V2X live log (polling fallback ensures display)
- 8-layer architecture section — every layer marked REAL

## Honest removals (things we didn't pretend)
- "Blockchain AI" — removed, was never real
- Google Maps tiles — removed, was never implemented
- "2000 vehicles" marketing number — now honest "configurable up to ~800"
- SUMO integration — never claimed; we built our own simulator

## Known limitations
- Per-vehicle emissions model is heuristic, not data-calibrated
- Training runs are bounded by sim queue dynamics — longer runs plateau around −45
- Unity client requires user to install NativeWebSocket + create prefabs (docs provided)

## Backlog / next
- Add WebSocket push of v2x more reliably (main path uses REST polling fallback)
- Live throughput/queue time-series chart on simulation page
- PPO/A2C baselines for apples-to-apples MARL comparison
- Containerize Unity sample build for one-click demo
