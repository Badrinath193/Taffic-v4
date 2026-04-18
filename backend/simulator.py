"""
Real multi-intersection traffic simulator with per-vehicle kinematics.
Two modes:
  - queue mode: used for fast MARL training (aggregate NS/EW queues)
  - vehicle mode: per-vehicle simulation for live visualization
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------- Queue-based env (for training) ----------------

@dataclass
class IntersectionState:
    id: str
    ns_queue: float
    ew_queue: float
    ns_wait: float
    ew_wait: float
    phase: int  # 0 => NS green, 1 => EW green
    predicted_inflow_ns: float = 0.0
    predicted_inflow_ew: float = 0.0

    def obs(self, neighbor_pressure: float) -> np.ndarray:
        return np.array([
            self.ns_queue, self.ew_queue,
            self.ns_wait, self.ew_wait,
            float(self.phase),
            self.predicted_inflow_ns, self.predicted_inflow_ew,
            neighbor_pressure,
        ], dtype=np.float32)


class MultiIntersectionEnv:
    """Queue-level multi-agent env used by the DQN trainer."""

    def __init__(self, rows: int = 3, cols: int = 3, seed: int = 7, max_steps: int = 150):
        self.rows = rows
        self.cols = cols
        self.n_agents = rows * cols
        self.agent_ids = [f"I{i}" for i in range(self.n_agents)]
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.step_count = 0
        self.history: List[Dict[str, float]] = []
        self.states: Dict[str, IntersectionState] = {}
        self.neighbors = self._build_neighbors()
        self.last_metrics: Dict[str, float] = {}
        self.reset()

    def _build_neighbors(self) -> Dict[str, List[str]]:
        n: Dict[str, List[str]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                ids: List[str] = []
                for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        ids.append(f"I{rr * self.cols + cc}")
                n[f"I{idx}"] = ids
        return n

    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.history.clear()
        self.states = {}
        for aid in self.agent_ids:
            self.states[aid] = IntersectionState(
                id=aid,
                ns_queue=self.rng.uniform(5, 18),
                ew_queue=self.rng.uniform(5, 18),
                ns_wait=self.rng.uniform(0, 6),
                ew_wait=self.rng.uniform(0, 6),
                phase=self.rng.randint(0, 1),
            )
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        out = {}
        for aid, st in self.states.items():
            neighbors = self.neighbors[aid]
            if neighbors:
                pressure = float(np.mean([self.states[n].ns_queue + self.states[n].ew_queue for n in neighbors]))
            else:
                pressure = 0.0
            out[aid] = st.obs(pressure)
        return out

    def _arrival_rate(self, idx: int) -> Tuple[float, float]:
        base = 3.2 + 0.6 * math.sin((self.step_count + idx * 3) / 7)
        pulse = 1.2 if 40 <= self.step_count % 120 <= 80 else 0.0
        ns = max(0.6, base + pulse + self.np_rng.normal(0, 0.35))
        ew = max(0.6, 2.8 + 0.5 * math.cos((self.step_count + idx * 5) / 9) + self.np_rng.normal(0, 0.35))
        return float(ns), float(ew)

    def step(self, actions: Dict[str, int]):
        self.step_count += 1
        rewards: Dict[str, float] = {}
        throughput = 0.0
        total_queue = 0.0
        fairness_penalties: List[float] = []

        pressures = {aid: (self.states[aid].ns_queue + self.states[aid].ew_queue) for aid in self.agent_ids}
        mean_pressure = float(np.mean(list(pressures.values())))

        for idx, aid in enumerate(self.agent_ids):
            st = self.states[aid]
            action = int(actions.get(aid, st.phase))
            switched = 1 if action != st.phase else 0
            st.phase = action

            inflow_ns, inflow_ew = self._arrival_rate(idx)
            neighbor_factor = 0.02 * sum(max(0.0, pressures[n] - mean_pressure) for n in self.neighbors[aid])
            neighbor_factor = min(neighbor_factor, 0.8)
            inflow_ns += neighbor_factor
            inflow_ew += neighbor_factor
            st.predicted_inflow_ns = inflow_ns
            st.predicted_inflow_ew = inflow_ew

            service_green = 6.0
            service_red = 0.6
            served_ns = min(st.ns_queue, service_green if st.phase == 0 else service_red)
            served_ew = min(st.ew_queue, service_green if st.phase == 1 else service_red)
            st.ns_queue = min(80.0, max(0.0, st.ns_queue - served_ns) + inflow_ns)
            st.ew_queue = min(80.0, max(0.0, st.ew_queue - served_ew) + inflow_ew)
            # wait accumulates on red and decays on green (with served vehicles)
            if st.phase == 0:  # NS green
                st.ns_wait = max(0.0, st.ns_wait * 0.85 - served_ns * 0.15)
                st.ew_wait = st.ew_wait + 0.12 * st.ew_queue
            else:
                st.ew_wait = max(0.0, st.ew_wait * 0.85 - served_ew * 0.15)
                st.ns_wait = st.ns_wait + 0.12 * st.ns_queue
            # cap waits to keep reward bounded
            st.ns_wait = min(st.ns_wait, 60.0)
            st.ew_wait = min(st.ew_wait, 60.0)

            fairness = abs(st.ns_wait - st.ew_wait)
            fairness_penalties.append(fairness)
            queue_penalty = st.ns_queue + st.ew_queue
            wait_penalty = 0.25 * (st.ns_wait + st.ew_wait)
            switch_penalty = 0.6 * switched
            rewards[aid] = -(queue_penalty + wait_penalty + 0.7 * fairness + switch_penalty)
            throughput += served_ns + served_ew
            total_queue += st.ns_queue + st.ew_queue

        done = self.step_count >= self.max_steps
        self.last_metrics = {
            "step": self.step_count,
            "throughput": throughput,
            "avg_queue": total_queue / self.n_agents,
            "avg_fairness_penalty": float(np.mean(fairness_penalties)),
            "avg_reward": float(np.mean(list(rewards.values()))),
        }
        return self.get_obs(), rewards, done, self.last_metrics

    def export_network(self) -> Dict[str, object]:
        nodes = []
        edges = []
        spacing = 180
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                x, y = c * spacing, r * spacing
                nodes.append({"id": f"I{idx}", "x": x, "y": y, "is_signal": True})
                if c + 1 < self.cols:
                    edges.append({"from": f"I{idx}", "to": f"I{idx+1}", "lanes": 2})
                    edges.append({"from": f"I{idx+1}", "to": f"I{idx}", "lanes": 2})
                if r + 1 < self.rows:
                    edges.append({"from": f"I{idx}", "to": f"I{idx+self.cols}", "lanes": 2})
                    edges.append({"from": f"I{idx+self.cols}", "to": f"I{idx}", "lanes": 2})
        return {"nodes": nodes, "edges": edges, "source": "synthetic-grid"}

    def snapshot(self) -> List[Dict[str, float]]:
        return [asdict(st) for st in self.states.values()]


# ---------------- Vehicle-level sim (for live visualization) ----------------

VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "emergency"]
TYPE_WEIGHTS = [0.62, 0.12, 0.08, 0.15, 0.03]
TYPE_LENGTHS = {"car": 4.5, "truck": 11.0, "bus": 12.5, "motorcycle": 2.0, "emergency": 5.5}
TYPE_MAX_SPEED = {"car": 14.0, "truck": 10.0, "bus": 11.0, "motorcycle": 18.0, "emergency": 20.0}


@dataclass
class Vehicle:
    id: int
    vtype: str
    edge: Tuple[str, str]  # (from_node_id, to_node_id)
    pos_on_edge: float     # meters along edge
    speed: float
    max_speed: float
    length: float
    stopped: bool = False


@dataclass
class TLState:
    node_id: str
    phase: int   # 0..3 : 0=NS green, 1=NS amber, 2=EW green, 3=EW amber
    timer: float = 0.0
    # durations can be overridden by DQN via set_phase
    dur_green: float = 18.0
    dur_amber: float = 3.0


class VehicleSim:
    """Network + per-vehicle kinematics. Can be driven by DQN actions."""

    def __init__(self, rows: int = 3, cols: int = 3, spacing: float = 150.0, max_vehicles: int = 300, seed: int = 11):
        self.rows = rows
        self.cols = cols
        self.spacing = spacing
        self.max_vehicles = max_vehicles
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict] = []
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}   # node -> list of (neighbor, edge_id)
        self.edge_map: Dict[str, Dict] = {}   # edge_id -> edge record
        self.tls: Dict[str, TLState] = {}
        self.vehicles: List[Vehicle] = []
        self.next_vid = 0
        self.step_id = 0
        self.metrics_history: List[Dict] = []
        self._build_grid()

    def _build_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                nid = f"N{r}_{c}"
                self.nodes[nid] = {"id": nid, "x": c * self.spacing, "y": r * self.spacing, "is_signal": True}
                self.tls[nid] = TLState(node_id=nid, phase=self.rng.randint(0, 3))

        def add_edge(a: str, b: str):
            eid = f"{a}->{b}"
            length = math.hypot(self.nodes[a]["x"] - self.nodes[b]["x"], self.nodes[a]["y"] - self.nodes[b]["y"])
            rec = {"id": eid, "from": a, "to": b, "length": length, "lanes": 2}
            self.edges.append(rec)
            self.edge_map[eid] = rec
            self.adjacency.setdefault(a, []).append((b, eid))

        for r in range(self.rows):
            for c in range(self.cols):
                nid = f"N{r}_{c}"
                if c + 1 < self.cols:
                    other = f"N{r}_{c+1}"
                    add_edge(nid, other); add_edge(other, nid)
                if r + 1 < self.rows:
                    other = f"N{r+1}_{c}"
                    add_edge(nid, other); add_edge(other, nid)

    def export_network(self) -> Dict:
        return {
            "nodes": list(self.nodes.values()),
            "edges": [{"id": e["id"], "from": e["from"], "to": e["to"], "length": e["length"], "lanes": e["lanes"]} for e in self.edges],
            "source": getattr(self, "_source", "synthetic-grid"),
            "rows": self.rows,
            "cols": self.cols,
        }

    def load_from_osm(self, graph: Dict, max_nodes: int = 800):
        """Replace the current network with an OSM-derived graph.

        - `graph.nodes`: [{id, x, y, is_signal, lat, lon}]
        - `graph.edges`: [{from, to, highway}]
        Traffic lights are placed at every node with is_signal=True.
        Large graphs are sub-sampled to at most `max_nodes` to keep the
        per-vehicle sim interactive in the browser / Unity.
        """
        raw_nodes = graph.get("nodes", [])
        raw_edges = graph.get("edges", [])
        if not raw_nodes or not raw_edges:
            raise ValueError("OSM graph has no nodes or edges")

        # Sub-sample: keep every k-th node but always keep is_signal nodes
        signals = [n for n in raw_nodes if n.get("is_signal")]
        non_signals = [n for n in raw_nodes if not n.get("is_signal")]
        if len(raw_nodes) > max_nodes:
            keep_non = max(0, max_nodes - len(signals))
            step = max(1, len(non_signals) // max(1, keep_non))
            non_signals = non_signals[::step][:keep_non]
        kept_nodes = signals + non_signals
        kept_ids = {n["id"] for n in kept_nodes}

        # reset
        self.nodes = {}
        self.edges = []
        self.edge_map = {}
        self.adjacency = {}
        self.tls = {}
        self.vehicles = []
        self.next_vid = 0
        self.step_id = 0
        self.metrics_history = []
        self._source = "osm"

        for n in kept_nodes:
            self.nodes[n["id"]] = {
                "id": n["id"], "x": float(n["x"]), "y": float(n["y"]),
                "is_signal": bool(n.get("is_signal")),
                "lat": n.get("lat"), "lon": n.get("lon"),
            }
            if n.get("is_signal"):
                self.tls[n["id"]] = TLState(node_id=n["id"], phase=self.rng.randint(0, 3))

        for e in raw_edges:
            a, b = e.get("from"), e.get("to")
            if a in kept_ids and b in kept_ids:
                length = math.hypot(
                    self.nodes[a]["x"] - self.nodes[b]["x"],
                    self.nodes[a]["y"] - self.nodes[b]["y"],
                )
                if length < 5.0:
                    continue
                eid = f"{a}->{b}"
                rec = {"id": eid, "from": a, "to": b, "length": length, "lanes": 2}
                self.edges.append(rec)
                self.edge_map[eid] = rec
                self.adjacency.setdefault(a, []).append((b, eid))

        # Drop isolated nodes to keep sim valid
        reachable = set()
        for e in self.edges:
            reachable.add(e["from"]); reachable.add(e["to"])
        self.nodes = {k: v for k, v in self.nodes.items() if k in reachable}
        self.tls = {k: v for k, v in self.tls.items() if k in reachable}

        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "signals": len(self.tls),
            "source": "osm",
        }

    def _random_edge(self) -> Dict:
        return self.rng.choice(self.edges)

    def _spawn_vehicle(self):
        if len(self.vehicles) >= self.max_vehicles:
            return
        e = self._random_edge()
        vtype = self.rng.choices(VEHICLE_TYPES, TYPE_WEIGHTS, k=1)[0]
        v = Vehicle(
            id=self.next_vid, vtype=vtype, edge=(e["from"], e["to"]),
            pos_on_edge=self.rng.uniform(0, e["length"] * 0.3),
            speed=self.rng.uniform(3, 8),
            max_speed=TYPE_MAX_SPEED[vtype],
            length=TYPE_LENGTHS[vtype],
        )
        self.next_vid += 1
        self.vehicles.append(v)

    def _is_green_for_edge(self, tl: TLState, edge: Dict) -> bool:
        # infer direction: horizontal (c changes) vs vertical (r changes)
        fr = self.nodes[edge["from"]]; to = self.nodes[edge["to"]]
        horizontal = abs(fr["y"] - to["y"]) < 1e-3
        # NS green (phase 0) serves vertical; EW green (phase 2) serves horizontal
        if tl.phase == 0:   # NS green
            return not horizontal
        if tl.phase == 2:   # EW green
            return horizontal
        return False   # amber phases block

    def set_phases(self, phase_map: Dict[str, int]):
        """Called by policy. phase_map: node_id -> desired phase (0=NS green, 1=EW green).
        We translate high-level 2-action into 4-phase state machine (NS green -> NS amber -> EW green -> EW amber)."""
        for nid, want in phase_map.items():
            tl = self.tls.get(nid)
            if tl is None:
                continue
            # current high-level color: 0 = NS active, 1 = EW active
            cur_high = 0 if tl.phase in (0, 1) else 1
            if want != cur_high and tl.phase in (0, 2):
                # move into amber
                tl.phase = 1 if tl.phase == 0 else 3
                tl.timer = 0.0

    def advance_signals(self, dt: float):
        for tl in self.tls.values():
            tl.timer += dt
            if tl.phase in (0, 2):  # green states — natural switch after dur_green (can be overridden by policy)
                if tl.timer >= tl.dur_green:
                    tl.phase = (tl.phase + 1) % 4
                    tl.timer = 0.0
            else:  # amber
                if tl.timer >= tl.dur_amber:
                    tl.phase = (tl.phase + 1) % 4
                    tl.timer = 0.0

    def step(self, dt: float = 0.4, spawn_rate: float = 0.85) -> Dict:
        self.step_id += 1
        # spawn
        spawn_budget = spawn_rate + (1.4 if self.step_id % 200 < 80 else 0.0)
        n_spawn = int(spawn_budget) + (1 if self.rng.random() < (spawn_budget - int(spawn_budget)) else 0)
        for _ in range(n_spawn):
            self._spawn_vehicle()
        self.advance_signals(dt)

        # update vehicles
        stopped = 0
        served = 0
        emissions = 0.0
        new_vehicles: List[Vehicle] = []
        for v in self.vehicles:
            edge = self.edge_map[f"{v.edge[0]}->{v.edge[1]}"]
            target_node = edge["to"]
            tl = self.tls.get(target_node)
            dist_to_end = edge["length"] - v.pos_on_edge
            should_stop = False
            if tl is not None and v.vtype != "emergency":
                green = self._is_green_for_edge(tl, edge)
                if not green and dist_to_end < 14.0:
                    should_stop = True
            # acceleration / decel
            if should_stop:
                v.speed = max(0.0, v.speed - 2.2 * dt)
            else:
                v.speed = min(v.max_speed, v.speed + 1.6 * dt)
            v.stopped = v.speed < 0.3
            if v.stopped:
                stopped += 1
            emissions += (0.2 + 0.02 * v.speed) * (1.6 if v.vtype == "truck" else 1.0)

            v.pos_on_edge += v.speed * dt
            if v.pos_on_edge >= edge["length"]:
                # hand-off to next edge
                served += 1
                next_opts = self.adjacency.get(target_node, [])
                # avoid U-turn if possible
                next_opts = [o for o in next_opts if o[0] != v.edge[0]] or next_opts
                if not next_opts:
                    continue  # drop
                nb, eid = self.rng.choice(next_opts)
                v.edge = (target_node, nb)
                v.pos_on_edge = 0.0
            new_vehicles.append(v)
        self.vehicles = new_vehicles

        metrics = {
            "step": self.step_id,
            "vehicles": len(self.vehicles),
            "stopped": stopped,
            "throughput": served,
            "emissions": emissions,
        }
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 600:
            self.metrics_history = self.metrics_history[-600:]
        return metrics

    def snapshot(self) -> Dict:
        """Compact snapshot for frontend rendering."""
        return {
            "step": self.step_id,
            "nodes": [{"id": n["id"], "x": n["x"], "y": n["y"]} for n in self.nodes.values()],
            "tls": [{"nid": t.node_id, "phase": t.phase} for t in self.tls.values()],
            "vehicles": [
                {
                    "id": v.id, "t": v.vtype,
                    "fx": self.nodes[v.edge[0]]["x"], "fy": self.nodes[v.edge[0]]["y"],
                    "tx": self.nodes[v.edge[1]]["x"], "ty": self.nodes[v.edge[1]]["y"],
                    "p": round(v.pos_on_edge / max(1.0, self.edge_map[f"{v.edge[0]}->{v.edge[1]}"]["length"]), 4),
                    "s": round(v.speed, 2), "st": v.stopped,
                } for v in self.vehicles
            ],
            "edges": [{"from": e["from"], "to": e["to"]} for e in self.edges],
        }

    def intersection_obs(self, nid: str) -> np.ndarray:
        """Build an 8-dim observation matching the DQN's expected input."""
        ns_q, ew_q, ns_w, ew_w = 0.0, 0.0, 0.0, 0.0
        incoming = [e for e in self.edges if e["to"] == nid]
        for e in incoming:
            fr, to = self.nodes[e["from"]], self.nodes[e["to"]]
            horizontal = abs(fr["y"] - to["y"]) < 1e-3
            for v in self.vehicles:
                if v.edge == (e["from"], e["to"]):
                    dist = e["length"] - v.pos_on_edge
                    if dist < 60:
                        if horizontal:
                            ew_q += 1
                            if v.stopped:
                                ew_w += 1
                        else:
                            ns_q += 1
                            if v.stopped:
                                ns_w += 1
        tl = self.tls[nid]
        phase_high = 0 if tl.phase in (0, 1) else 1
        return np.array([ns_q, ew_q, ns_w, ew_w, float(phase_high), 0.0, 0.0, 0.0], dtype=np.float32)
