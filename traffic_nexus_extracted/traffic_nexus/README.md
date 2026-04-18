# Traffic Nexus Local Upgrade

This package turns the earlier browser mockup into a runnable local project with real code paths for:

- fairness-aware reward shaping
- a neural queue forecast model
- a shared DQN traffic-signal policy
- experiment metrics export
- OSM import through a local proxy with Overpass fallback endpoints
- automated tests

## Run

```bash
cd traffic_nexus
python3 -m uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/`

## Train

Use the UI or:

```bash
curl -X POST http://127.0.0.1:8000/api/ml/train \
  -H 'Content-Type: application/json' \
  -d '{"episodes":40,"seed":7}'
```

## Test

```bash
cd traffic_nexus
pytest -q
```

## Notes

- The RL environment is a lightweight multi-intersection simulator, not SUMO.
- OSM import uses Nominatim for geocoding and retries multiple Overpass endpoints.
- External OSM availability still depends on live public services.
- You can later swap the environment with SUMO or RESCO while reusing the reward, forecast, and experiment plumbing.
