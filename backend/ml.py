"""
Real ML for Traffic Nexus:
  - ForecastNet: MLP that predicts next-step NS/EW queues
  - QNet: shared DQN policy (8-dim obs -> 2 actions)
  - train_shared_dqn: real training loop with replay buffer + target network
  - evaluate_policy: compares learned vs fixed vs pressure-control
All artifacts saved under ./artifacts/.
"""
from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from simulator import MultiIntersectionEnv


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
    def __init__(self, in_dim: int = 8, hidden: int = 64, n_actions: int = 2):
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
    def __init__(self, capacity: int = 20000):
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


def build_forecast_dataset(env: MultiIntersectionEnv, episodes: int = 8):
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
                po = prev_obs[aid]; no = next_obs[aid]
                x = np.array([po[0], po[1], po[2], po[3], po[4], po[7]], dtype=np.float32)
                y = np.array([no[0], no[1]], dtype=np.float32)
                xs.append(x); ys.append(y)
                prev_obs[aid] = no.copy()
    return np.stack(xs), np.stack(ys)


def train_forecast_model(out_dir: str, seed: int = 7, progress_cb: Optional[Callable[[Dict], None]] = None) -> str:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = MultiIntersectionEnv(seed=seed)
    X, Y = build_forecast_dataset(env)
    # normalize
    x_mean = X.mean(0); x_std = X.std(0) + 1e-6
    y_mean = Y.mean(0); y_std = Y.std(0) + 1e-6
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    model = ForecastNet().to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tx = torch.tensor(Xn, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(Yn, dtype=torch.float32, device=DEVICE)
    losses = []
    for epoch in range(40):
        pred = model(tx)
        loss = loss_fn(pred, ty)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if progress_cb:
            progress_cb({"kind": "forecast", "epoch": epoch + 1, "loss": float(loss.item())})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "forecast_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
        "final_loss": losses[-1],
    }, path)
    return path


def load_forecast(path: str) -> Tuple[ForecastNet, Dict]:
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ForecastNet().to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def attach_forecast(model: ForecastNet, blob: Dict, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    x_mean = np.array(blob["x_mean"]); x_std = np.array(blob["x_std"])
    y_mean = np.array(blob["y_mean"]); y_std = np.array(blob["y_std"])
    with torch.no_grad():
        for aid, obs in obs_dict.items():
            x = np.array([obs[0], obs[1], obs[2], obs[3], obs[4], obs[7]], dtype=np.float32)
            xn = (x - x_mean) / x_std
            pred_n = model(torch.tensor(xn, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            pred = pred_n * y_std + y_mean
            arr = obs.copy()
            arr[5] = float(pred[0]); arr[6] = float(pred[1])
            out[aid] = arr
    return out


def _setup_training(seed: int) -> Tuple[MultiIntersectionEnv, QNet, QNet, Adam, ReplayBuffer]:
    env = MultiIntersectionEnv(seed=seed)
    q = QNet().to(DEVICE)
    target = QNet().to(DEVICE)
    target.load_state_dict(q.state_dict())
    opt = Adam(q.parameters(), lr=1e-3)
    buf = ReplayBuffer(20000)
    return env, q, target, opt, buf


def _select_actions(obs: Dict[str, np.ndarray], q: QNet, epsilon: float) -> Dict[str, int]:
    actions = {}
    for aid, ob in obs.items():
        if random.random() < epsilon:
            actions[aid] = random.randint(0, 1)
        else:
            with torch.no_grad():
                qv = q(torch.tensor(ob, dtype=torch.float32, device=DEVICE))
                actions[aid] = int(torch.argmax(qv).item())
    return actions


def _optimize_model(q: QNet, target: QNet, opt: Adam, buf: ReplayBuffer, batch_size: int, gamma: float):
    if len(buf) >= batch_size:
        s, a, r, ns, d = buf.sample(batch_size)
        qvals = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = target(ns).max(dim=1).values
            tgt = r + gamma * (1.0 - d) * nxt
        loss = torch.nn.functional.smooth_l1_loss(qvals, tgt)
        opt.zero_grad(); loss.backward(); opt.step()


def _save_training_metrics(out_dir: str, episodes: int, rewards_hist: List[float], queues_hist: List[float], fair_hist: List[float]) -> str:
    metrics_path = os.path.join(out_dir, "training_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "avg_reward", "avg_queue", "avg_fairness_penalty"])
        for i, (r, qv, fv) in enumerate(zip(rewards_hist, queues_hist, fair_hist), start=1):
            writer.writerow([i, r, qv, fv])

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "episodes": episodes,
            "final_avg_reward": rewards_hist[-1] if rewards_hist else 0.0,
            "final_avg_queue": queues_hist[-1] if queues_hist else 0.0,
            "final_avg_fairness_penalty": fair_hist[-1] if fair_hist else 0.0,
            "best_avg_reward": max(rewards_hist) if rewards_hist else 0.0,
        }, f, indent=2)
    return metrics_path


def train_shared_dqn(
    out_dir: str,
    episodes: int = 40,
    seed: int = 7,
    progress_cb: Optional[Callable[[Dict], None]] = None,
) -> TrainResult:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    forecast_path = train_forecast_model(out_dir, seed, progress_cb=progress_cb)
    forecast, blob = load_forecast(forecast_path)

    env, q, target, opt, buf = _setup_training(seed)

    gamma = 0.97
    batch_size = 64
    epsilon = 1.0
    rewards_hist: List[float] = []
    queues_hist: List[float] = []
    fair_hist: List[float] = []

    for ep in range(episodes):
        obs = attach_forecast(forecast, blob, env.reset())
        done = False
        ep_reward = 0.0
        last_info = {}

        while not done:
            actions = _select_actions(obs, q, epsilon)
            next_obs, rewards, done, info = env.step(actions)
            last_info = info
            next_obs = attach_forecast(forecast, blob, next_obs)

            for aid in env.agent_ids:
                buf.push(obs[aid], actions[aid], rewards[aid], next_obs[aid], float(done))
                ep_reward += rewards[aid]
            obs = next_obs

            _optimize_model(q, target, opt, buf, batch_size, gamma)

        epsilon = max(0.08, epsilon * 0.93)
        if ep % 5 == 0:
            target.load_state_dict(q.state_dict())

        avg_reward = ep_reward / env.n_agents / env.max_steps
        rewards_hist.append(avg_reward)
        queues_hist.append(last_info.get("avg_queue", 0.0))
        fair_hist.append(last_info.get("avg_fairness_penalty", 0.0))

        if progress_cb:
            progress_cb({
                "kind": "dqn",
                "episode": ep + 1,
                "total": episodes,
                "avg_reward": avg_reward,
                "avg_queue": queues_hist[-1],
                "avg_fairness_penalty": fair_hist[-1],
                "epsilon": epsilon,
            })

    model_path = os.path.join(out_dir, "shared_dqn.pt")
    torch.save(q.state_dict(), model_path)
    metrics_path = _save_training_metrics(out_dir, episodes, rewards_hist, queues_hist, fair_hist)

    return TrainResult(rewards_hist, queues_hist, fair_hist, model_path, forecast_path, metrics_path)


def load_qnet(path: str) -> QNet:
    q = QNet().to(DEVICE)
    q.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    q.eval()
    return q


def evaluate_policy(model_path: str, forecast_path: str, episodes: int = 3, seed: int = 99) -> Dict[str, float]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    q = load_qnet(model_path)
    forecast, blob = load_forecast(forecast_path)

    def run(policy: str):
        env = MultiIntersectionEnv(seed=seed)
        total_r = 0.0; total_q = 0.0; total_f = 0.0
        for _ in range(episodes):
            obs = attach_forecast(forecast, blob, env.reset())
            done = False
            while not done:
                actions = {}
                for aid, ob in obs.items():
                    if policy == "fixed":
                        actions[aid] = (env.step_count // 10) % 2
                    elif policy == "pressure":
                        actions[aid] = 0 if ob[0] + ob[5] >= ob[1] + ob[6] else 1
                    else:
                        with torch.no_grad():
                            actions[aid] = int(torch.argmax(q(torch.tensor(ob, dtype=torch.float32, device=DEVICE))).item())
                obs, rewards, done, info = env.step(actions)
                obs = attach_forecast(forecast, blob, obs)
                total_r += float(np.mean(list(rewards.values())))
                total_q += info["avg_queue"]
                total_f += info["avg_fairness_penalty"]
        denom = episodes * env.max_steps
        return total_r / denom, total_q / denom, total_f / denom

    learned = run("learned")
    fixed = run("fixed")
    pressure = run("pressure")
    return {
        "learned_reward": learned[0], "learned_queue": learned[1], "learned_fairness": learned[2],
        "fixed_reward": fixed[0], "fixed_queue": fixed[1], "fixed_fairness": fixed[2],
        "pressure_reward": pressure[0], "pressure_queue": pressure[1], "pressure_fairness": pressure[2],
    }
