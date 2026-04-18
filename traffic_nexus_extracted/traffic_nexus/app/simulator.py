from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import math
import random
import numpy as np


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
            self.ns_queue,
            self.ew_queue,
            self.ns_wait,
            self.ew_wait,
            float(self.phase),
            self.predicted_inflow_ns,
            self.predicted_inflow_ew,
            neighbor_pressure,
        ], dtype=np.float32)


class MultiIntersectionEnv:
    """A lightweight multi-agent traffic environment.

    It is not SUMO, but it is an actual executable RL environment with
    queue dynamics, stochastic arrivals, fairness-aware reward, and
    neighbor interaction through pressure propagation.
    """

    def __init__(self, rows: int = 2, cols: int = 2, seed: int = 7, max_steps: int = 120):
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
        neighbors: Dict[str, List[str]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                ids: List[str] = []
                for rr, cc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                    if 0 <= rr < self.rows and 0 <= cc < self.cols:
                        ids.append(f"I{rr * self.cols + cc}")
                neighbors[f"I{idx}"] = ids
        return neighbors

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
            neighbor_pressure = float(np.mean([
                self.states[n].ns_queue + self.states[n].ew_queue for n in self.neighbors[aid]
            ] or [0.0]))
            out[aid] = st.obs(neighbor_pressure)
        return out

    def _arrival_rate(self, idx: int) -> Tuple[float, float]:
        # time-varying demand, slightly different per intersection
        base = 3.2 + 0.6 * math.sin((self.step_count + idx * 3) / 7)
        pulse = 1.2 if 40 <= self.step_count % 120 <= 80 else 0.0
        ns = max(0.6, base + pulse + self.np_rng.normal(0, 0.35))
        ew = max(0.6, 2.8 + 0.5 * math.cos((self.step_count + idx * 5) / 9) + self.np_rng.normal(0, 0.35))
        return float(ns), float(ew)

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, float]]:
        self.step_count += 1
        rewards: Dict[str, float] = {}
        throughput = 0.0
        total_queue = 0.0
        fairness_penalties: List[float] = []

        # mild neighbor spillback coupling
        pressures = {aid: (self.states[aid].ns_queue + self.states[aid].ew_queue) for aid in self.agent_ids}
        mean_pressure = float(np.mean(list(pressures.values())))

        for idx, aid in enumerate(self.agent_ids):
            st = self.states[aid]
            action = int(actions.get(aid, st.phase))
            switched = 1 if action != st.phase else 0
            st.phase = action

            inflow_ns, inflow_ew = self._arrival_rate(idx)
            neighbor_factor = 0.08 * sum(max(0.0, pressures[n] - mean_pressure) for n in self.neighbors[aid])
            inflow_ns += neighbor_factor
            inflow_ew += neighbor_factor
            st.predicted_inflow_ns = inflow_ns
            st.predicted_inflow_ew = inflow_ew

            service_green = 5.4
            service_red = 0.8
            served_ns = min(st.ns_queue, service_green if st.phase == 0 else service_red)
            served_ew = min(st.ew_queue, service_green if st.phase == 1 else service_red)
            st.ns_queue = max(0.0, st.ns_queue - served_ns) + inflow_ns
            st.ew_queue = max(0.0, st.ew_queue - served_ew) + inflow_ew
            st.ns_wait = max(0.0, st.ns_wait + st.ns_queue * (0.16 if st.phase == 1 else 0.05))
            st.ew_wait = max(0.0, st.ew_wait + st.ew_queue * (0.16 if st.phase == 0 else 0.05))

            fairness = abs(st.ns_wait - st.ew_wait)
            fairness_penalties.append(fairness)
            queue_penalty = st.ns_queue + st.ew_queue
            wait_penalty = 0.25 * (st.ns_wait + st.ew_wait)
            switch_penalty = 0.6 * switched
            rewards[aid] = -(queue_penalty + wait_penalty + 0.7 * fairness + switch_penalty)
            throughput += served_ns + served_ew
            total_queue += st.ns_queue + st.ew_queue

            self.history.append({
                "step": self.step_count,
                "agent": aid,
                "ns_queue": st.ns_queue,
                "ew_queue": st.ew_queue,
                "ns_wait": st.ns_wait,
                "ew_wait": st.ew_wait,
                "phase": st.phase,
                "inflow_ns": inflow_ns,
                "inflow_ew": inflow_ew,
                "reward": rewards[aid],
            })

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
