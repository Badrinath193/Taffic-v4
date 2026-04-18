from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .simulator import MultiIntersectionEnv


DEVICE = torch.device("cpu")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class ForecastNet(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 48, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainResult:
    rewards: List[float]
    queues: List[float]
    fairness: List[float]
    model_path: str
    forecast_path: str
    metrics_path: str


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.data: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.idx = 0

    def push(self, s, a, r, ns, d):
        item = (np.array(s, dtype=np.float32), int(a), float(r), np.array(ns, dtype=np.float32), float(d))
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32, device=DEVICE),
            torch.tensor(a, dtype=torch.int64, device=DEVICE),
            torch.tensor(r, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.stack(ns), dtype=torch.float32, device=DEVICE),
            torch.tensor(d, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.data)


def build_forecast_dataset(env: MultiIntersectionEnv, episodes: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _ in range(episodes):
        obs = env.reset()
        prev_obs = {k: v.copy() for k, v in obs.items()}
        done = False
        while not done:
            actions = {aid: random.randint(0, 1) for aid in env.agent_ids}
            next_obs, _, done, _ = env.step(actions)
            for aid in env.agent_ids:
                po = prev_obs[aid]
                no = next_obs[aid]
                x = np.array([po[0], po[1], po[2], po[3], po[4], po[7]], dtype=np.float32)
                y = np.array([no[0], no[1]], dtype=np.float32)
                xs.append(x)
                ys.append(y)
                prev_obs[aid] = no.copy()
    return np.stack(xs), np.stack(ys)


def train_forecast_model(out_dir: str, seed: int = 7) -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = MultiIntersectionEnv(seed=seed)
    X, Y = build_forecast_dataset(env)
    model = ForecastNet().to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tx = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    for _ in range(20):
        pred = model(tx)
        loss = loss_fn(pred, ty)
        opt.zero_grad()
        loss.backward()
        opt.step()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "forecast_model.pt")
    torch.save(model.state_dict(), path)
    return path


def attach_forecast(model: ForecastNet, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    with torch.no_grad():
        for aid, obs in obs_dict.items():
            x = torch.tensor([obs[0], obs[1], obs[2], obs[3], obs[4], obs[7]], dtype=torch.float32, device=DEVICE)
            pred = model(x).cpu().numpy()
            arr = obs.copy()
            arr[5] = float(pred[0])
            arr[6] = float(pred[1])
            out[aid] = arr
    return out


def train_shared_dqn(out_dir: str, episodes: int = 20, seed: int = 7) -> TrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    forecast_path = train_forecast_model(out_dir, seed)
    forecast = ForecastNet().to(DEVICE)
    forecast.load_state_dict(torch.load(forecast_path, map_location=DEVICE))
    forecast.eval()

    env = MultiIntersectionEnv(seed=seed)
    q = QNet().to(DEVICE)
    target = QNet().to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = Adam(q.parameters(), lr=1e-3)
    buf = ReplayBuffer(12000)

    gamma = 0.97
    batch_size = 64
    epsilon = 1.0
    rewards_hist: List[float] = []
    queues_hist: List[float] = []
    fair_hist: List[float] = []

    for ep in range(episodes):
        obs = attach_forecast(forecast, env.reset())
        done = False
        ep_reward = 0.0
        while not done:
            actions = {}
            for aid, ob in obs.items():
                if random.random() < epsilon:
                    actions[aid] = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        qv = q(torch.tensor(ob, dtype=torch.float32, device=DEVICE))
                        actions[aid] = int(torch.argmax(qv).item())
            next_obs, rewards, done, info = env.step(actions)
            next_obs = attach_forecast(forecast, next_obs)
            for aid in env.agent_ids:
                buf.push(obs[aid], actions[aid], rewards[aid], next_obs[aid], float(done))
                ep_reward += rewards[aid]
            obs = next_obs

            if len(buf) >= batch_size:
                s, a, r, ns, d = buf.sample(batch_size)
                qvals = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nxt = target(ns).max(dim=1).values
                    tgt = r + gamma * (1.0 - d) * nxt
                loss = torch.nn.functional.smooth_l1_loss(qvals, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

        epsilon = max(0.08, epsilon * 0.95)
        if ep % 5 == 0:
            target.load_state_dict(q.state_dict())
        rewards_hist.append(ep_reward / env.n_agents)
        queues_hist.append(info["avg_queue"])
        fair_hist.append(info["avg_fairness_penalty"])

    model_path = os.path.join(out_dir, "shared_dqn.pt")
    torch.save(q.state_dict(), model_path)
    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "avg_reward", "avg_queue", "avg_fairness_penalty"])
        for i, (r, qv, fv) in enumerate(zip(rewards_hist, queues_hist, fair_hist), start=1):
            writer.writerow([i, r, qv, fv])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "episodes": episodes,
            "final_avg_reward": rewards_hist[-1],
            "final_avg_queue": queues_hist[-1],
            "final_avg_fairness_penalty": fair_hist[-1],
            "best_avg_reward": max(rewards_hist),
        }, f, indent=2)
    return TrainResult(rewards_hist, queues_hist, fair_hist, model_path, forecast_path, metrics_path)


def evaluate_policy(model_path: str, forecast_path: str, episodes: int = 5, seed: int = 99) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q = QNet().to(DEVICE)
    q.load_state_dict(torch.load(model_path, map_location=DEVICE))
    q.eval()
    forecast = ForecastNet().to(DEVICE)
    forecast.load_state_dict(torch.load(forecast_path, map_location=DEVICE))
    forecast.eval()

    def run(policy: str) -> Tuple[float, float, float]:
        env = MultiIntersectionEnv(seed=seed)
        total_reward = 0.0
        total_queue = 0.0
        total_fair = 0.0
        for _ in range(episodes):
            obs = attach_forecast(forecast, env.reset())
            done = False
            while not done:
                actions = {}
                for aid, ob in obs.items():
                    if policy == "fixed":
                        actions[aid] = 0
                    elif policy == "pressure":
                        actions[aid] = 0 if ob[0] + ob[5] >= ob[1] + ob[6] else 1
                    else:
                        with torch.no_grad():
                            actions[aid] = int(torch.argmax(q(torch.tensor(ob, dtype=torch.float32, device=DEVICE))).item())
                obs, rewards, done, info = env.step(actions)
                obs = attach_forecast(forecast, obs)
                total_reward += float(np.mean(list(rewards.values())))
                total_queue += info["avg_queue"]
                total_fair += info["avg_fairness_penalty"]
        denom = episodes * env.max_steps
        return total_reward / denom, total_queue / denom, total_fair / denom

    learned = run("learned")
    fixed = run("fixed")
    pressure = run("pressure")
    return {
        "learned_reward": learned[0],
        "learned_queue": learned[1],
        "learned_fairness": learned[2],
        "fixed_reward": fixed[0],
        "fixed_queue": fixed[1],
        "fixed_fairness": fixed[2],
        "pressure_reward": pressure[0],
        "pressure_queue": pressure[1],
        "pressure_fairness": pressure[2],
    }
