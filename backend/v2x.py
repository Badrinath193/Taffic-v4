"""
Real V2X message generation derived from actual simulator state transitions.
Unlike the old random-picker, every message here is caused by a real event.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

from simulator import VehicleSim, TLState


class V2XBus:
    """Observes a VehicleSim across steps and emits real messages."""

    def __init__(self, sim: VehicleSim, max_log: int = 200):
        self.sim = sim
        self.max_log = max_log
        self.log: Deque[Dict] = deque(maxlen=max_log)
        self._prev_phases: Dict[str, int] = {nid: tl.phase for nid, tl in sim.tls.items()}
        self._prev_queue: Dict[str, float] = {}

    def _push(self, msg_type: str, src: str, payload: Dict):
        self.log.append({
            "ts": round(time.time(), 3),
            "t": msg_type,
            "src": src,
            "p": payload,
        })

    def tick(self):
        # PHASE_CHANGE
        for nid, tl in self.sim.tls.items():
            prev = self._prev_phases.get(nid)
            if prev != tl.phase:
                self._push("PHASE_CHANGE", f"TL:{nid}", {"from": prev, "to": tl.phase})
                self._prev_phases[nid] = tl.phase

        # QUEUE_UPDATE (emit when the aggregate queue changes significantly)
        for nid in self.sim.tls.keys():
            obs = self.sim.intersection_obs(nid)
            q = float(obs[0] + obs[1])
            prev = self._prev_queue.get(nid, 0.0)
            if abs(q - prev) >= 3:
                self._push("QUEUE_UPDATE", f"TL:{nid}", {"queue": round(q, 1), "delta": round(q - prev, 1)})
                self._prev_queue[nid] = q

        # EMERGENCY_PREEMPT: real emergency vehicles near signals
        for v in self.sim.vehicles:
            if v.vtype != "emergency":
                continue
            edge = self.sim.edge_map.get(v.edge)
            if not edge:
                continue
            dist = edge["length"] - v.pos_on_edge
            if dist < 40:
                self._push("EMERGENCY_PREEMPT", f"V:{v.id}", {
                    "toward": edge["to"], "dist": round(dist, 1), "speed": round(v.speed, 1),
                })

        # ROUTE_INTENT: sample a vehicle's current path
        if self.sim.vehicles and self.sim.step_id % 6 == 0:
            v = self.sim.vehicles[self.sim.step_id % len(self.sim.vehicles)]
            self._push("ROUTE_INTENT", f"V:{v.id}", {
                "from": v.edge[0], "to": v.edge[1], "type": v.vtype,
            })

    def tail(self, n: int = 50) -> List[Dict]:
        return list(self.log)[-n:]
