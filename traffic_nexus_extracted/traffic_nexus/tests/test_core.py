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


def test_osm_import_validation():
    client = TestClient(app)

    # Negative test case: Invalid characters
    res = client.post('/api/osm/import', json={'place': 'Paris; DROP TABLE;', 'radius': 1800})
    assert res.status_code == 400
    assert "Invalid characters" in res.json()['detail']

    # Negative test case: Injection attempt
    res = client.post('/api/osm/import', json={'place': 'New York <script>alert(1)</script>', 'radius': 1800})
    assert res.status_code == 400
    assert "Invalid characters" in res.json()['detail']

    # Positive test case: Valid place name
    # We will mock requests to avoid hitting the actual API
    # Here we just want to test if it passes validation. If it passes validation, it will try to call geocode
    # which might fail with 502 since it fails to resolve or connect without a mock.
    # But it should not be 400.
    res = client.post('/api/osm/import', json={'place': 'New York, NY', 'radius': 1800})
    assert res.status_code != 400
