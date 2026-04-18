"""
Unity Bridge:
  A documented WebSocket protocol that broadcasts simulator snapshots to any
  connected client (browser Three.js viewer, Unity standalone client, etc.).

Protocol (JSON over WS):
  Server -> Client every N ms:
    { "type": "snapshot", "step": int, "nodes": [...], "tls": [...], "vehicles": [...], "edges": [...] }
  Server -> Client on V2X:
    { "type": "v2x", "msg": {...} }
  Client -> Server (control):
    { "type": "cmd", "action": "pause" | "resume" | "step" | "set_policy", "value": ... }

A sample Unity C# client that consumes this protocol ships in /app/unity_client/.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Set

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active.discard(ws)

    async def broadcast(self, data: Dict):
        if not self.active:
            return
        payload = json.dumps(data)
        dead = []
        async with self._lock:
            conns = list(self.active)
        for ws in conns:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self.active.discard(ws)
