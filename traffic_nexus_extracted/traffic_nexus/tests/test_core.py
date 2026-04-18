from fastapi.testclient import TestClient
import numpy as np

from app.simulator import MultiIntersectionEnv
from app.main import app


def test_env_step_shapes():
    env = MultiIntersectionEnv()
    obs = env.reset()
    assert len(obs) == 4
    nxt, rewards, done, info = env.step({aid: 0 for aid in env.agent_ids})
    assert set(nxt.keys()) == set(env.agent_ids)
    assert all(v.shape == (8,) for v in nxt.values())
    assert 'avg_queue' in info
    assert not done
    assert np.isfinite(info['avg_queue'])


def test_api_synthetic_network():
    client = TestClient(app)
    res = client.get('/api/network/synthetic?rows=2&cols=2')
    assert res.status_code == 200
    data = res.json()
    assert len(data['nodes']) == 4
    assert len(data['edges']) >= 8


def test_training_endpoint_runs():
    client = TestClient(app)
    res = client.post('/api/ml/train', json={'episodes': 5, 'seed': 3})
    assert res.status_code == 200
    data = res.json()
    assert 'evaluation' in data
    assert 'learned_queue' in data['evaluation']
