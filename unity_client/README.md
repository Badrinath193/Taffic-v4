# Traffic Nexus — Unity Client

This directory contains a **sample Unity client** that connects to the Traffic Nexus
backend via WebSocket and renders vehicles + traffic lights in real 3D.

## Setup

1. Create a new Unity project (2022.3+).
2. Install [NativeWebSocket](https://github.com/endel/NativeWebSocket) via Package Manager:
   - `Window → Package Manager → + → Add package from git URL…`
   - Paste: `https://github.com/endel/NativeWebSocket.git#upm`
3. Copy `TrafficNexusBridgeClient.cs` into `Assets/Scripts/`.
4. Create an empty GameObject in your scene, attach `TrafficNexusBridgeClient`.
5. Create simple vehicle and traffic-light prefabs and drag them into the inspector slots.
6. Set **wsUrl** to your backend (e.g. `ws://localhost:8001/ws/stream` or your deployed URL).
7. Press Play. The scene will synchronize with whatever the FastAPI simulator is currently emitting.

## Protocol

```jsonc
// server → client, every ~250ms:
{
  "type": "snapshot",
  "snapshot": {
    "step": 123,
    "nodes":    [ { "id": "N0_0", "x": 0,   "y": 0   }, ... ],
    "edges":    [ { "from": "N0_0", "to": "N0_1" }, ... ],
    "tls":      [ { "nid": "N0_0", "phase": 0 }, ... ],
    "vehicles": [ { "id": 42, "t": "car", "fx":0, "fy":0, "tx":150, "ty":0, "p":0.4, "s":8.2, "st": false }, ... ]
  },
  "v2x":    [ { "ts": 171..., "t": "PHASE_CHANGE", "src": "TL:N0_0", "p": {"from":0,"to":1} }, ... ],
  "metrics":{ "step": 123, "vehicles": 200, "stopped": 17, "throughput": 14, "emissions": 52.1 },
  "policy": "learned"
}

// client → server (optional):
{ "type": "cmd", "action": "pause" }
{ "type": "cmd", "action": "resume" }
{ "type": "cmd", "action": "set_policy", "value": "learned" }
```

The React frontend uses exactly the same WebSocket and protocol — so running both side-by-side
gives you a real multi-client visualization of the same simulation.
