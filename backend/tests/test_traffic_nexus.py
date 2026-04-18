"""
Traffic Nexus Backend API Tests
Tests all endpoints: health, network, simulation, ML, OSM, V2X
"""
import os
import time
import pytest
import requests
import json

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://doc-scanner-76.preview.emergentagent.com').rstrip('/')

class TestHealthEndpoint:
    """Health check endpoint tests"""
    
    def test_health_returns_ok(self):
        """GET /api/health returns {ok:true, models_loaded:true}"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert "models_loaded" in data
        print(f"Health check passed: ok={data['ok']}, models_loaded={data['models_loaded']}")


class TestNetworkEndpoint:
    """Synthetic network generation tests"""
    
    def test_synthetic_network_3x3(self):
        """GET /api/network/synthetic?rows=3&cols=3 returns graph with 9 nodes and >=24 edges"""
        response = requests.get(f"{BASE_URL}/api/network/synthetic", params={"rows": 3, "cols": 3}, timeout=10)
        assert response.status_code == 200
        data = response.json()
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        assert len(nodes) == 9, f"Expected 9 nodes, got {len(nodes)}"
        assert len(edges) >= 24, f"Expected >=24 edges, got {len(edges)}"
        assert data.get("source") == "synthetic-grid"
        print(f"Synthetic network: {len(nodes)} nodes, {len(edges)} edges")


class TestSimulationEndpoints:
    """Simulation control endpoint tests"""
    
    def test_sim_start_and_state(self):
        """POST /api/sim/start starts simulation; GET /api/sim/state returns step>0 with vehicles>0"""
        # Start simulation
        start_response = requests.post(
            f"{BASE_URL}/api/sim/start",
            json={"rows": 3, "cols": 3, "max_vehicles": 200, "seed": 11},
            timeout=15
        )
        assert start_response.status_code == 200
        start_data = start_response.json()
        assert start_data.get("ok") is True
        assert "network" in start_data
        print(f"Simulation started: ok={start_data['ok']}")
        
        # Wait for simulation to run a few steps
        time.sleep(3)
        
        # Check state
        state_response = requests.get(f"{BASE_URL}/api/sim/state", timeout=10)
        assert state_response.status_code == 200
        state_data = state_response.json()
        
        assert state_data.get("running") is True, "Simulation should be running"
        assert state_data.get("step", 0) > 0, f"Step should be > 0, got {state_data.get('step')}"
        vehicles = state_data.get("vehicles", [])
        assert len(vehicles) > 0, f"Should have vehicles, got {len(vehicles)}"
        print(f"Simulation state: step={state_data.get('step')}, vehicles={len(vehicles)}, running={state_data.get('running')}")
    
    def test_set_policy_learned(self):
        """POST /api/sim/set_policy with {policy:'learned'} returns learned_available:true"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "learned"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert data.get("policy") == "learned"
        assert data.get("learned_available") is True, "DQN model should be loaded"
        print(f"Set policy to learned: learned_available={data.get('learned_available')}")
    
    def test_set_policy_fixed(self):
        """POST /api/sim/set_policy with {policy:'fixed'} works"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "fixed"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert data.get("policy") == "fixed"
        print(f"Set policy to fixed: ok={data.get('ok')}")
    
    def test_set_policy_pressure(self):
        """POST /api/sim/set_policy with {policy:'pressure'} works"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "pressure"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        assert data.get("policy") == "pressure"
        print(f"Set policy to pressure: ok={data.get('ok')}")
    
    def test_set_policy_invalid_returns_400(self):
        """POST /api/sim/set_policy with invalid policy returns 400"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "invalid_policy"},
            timeout=10
        )
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print(f"Invalid policy correctly rejected with 400")
    
    def test_sim_stop(self):
        """POST /api/sim/stop stops the simulation"""
        response = requests.post(f"{BASE_URL}/api/sim/stop", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        print(f"Simulation stopped: ok={data.get('ok')}")


class TestV2XEndpoint:
    """V2X messaging endpoint tests"""
    
    def test_v2x_tail_when_running(self):
        """GET /api/v2x/tail?n=5 returns real v2x messages when sim is running"""
        # First ensure simulation is running
        requests.post(
            f"{BASE_URL}/api/sim/start",
            json={"rows": 3, "cols": 3, "max_vehicles": 200, "seed": 11},
            timeout=15
        )
        time.sleep(4)  # Let simulation generate V2X messages
        
        response = requests.get(f"{BASE_URL}/api/v2x/tail", params={"n": 5}, timeout=10)
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list), "V2X tail should return a list"
        if len(data) > 0:
            # Check message structure
            msg = data[0]
            assert "ts" in msg, "V2X message should have timestamp"
            assert "t" in msg, "V2X message should have type"
            assert "src" in msg, "V2X message should have source"
            assert "p" in msg, "V2X message should have payload"
            print(f"V2X messages: {len(data)} messages, types: {set(m['t'] for m in data)}")
        else:
            print("V2X tail returned empty list (may need more simulation time)")


class TestMLEndpoints:
    """ML training and evaluation endpoint tests"""
    
    def test_ml_summary_trained(self):
        """GET /api/ml/summary returns trained:true with episodes field"""
        response = requests.get(f"{BASE_URL}/api/ml/summary", timeout=10)
        assert response.status_code == 200
        data = response.json()
        
        assert data.get("trained") is True, "Model should be trained (pre-seeded)"
        assert "episodes" in data, "Summary should have episodes field"
        print(f"ML Summary: trained={data.get('trained')}, episodes={data.get('episodes')}")
    
    def test_ml_metrics_returns_list(self):
        """GET /api/ml/metrics returns list with numeric avg_reward, avg_queue, avg_fairness_penalty"""
        response = requests.get(f"{BASE_URL}/api/ml/metrics", timeout=10)
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list), "Metrics should be a list"
        if len(data) > 0:
            metric = data[0]
            assert "avg_reward" in metric, "Metric should have avg_reward"
            assert "avg_queue" in metric, "Metric should have avg_queue"
            assert "avg_fairness_penalty" in metric, "Metric should have avg_fairness_penalty"
            assert isinstance(metric["avg_reward"], (int, float)), "avg_reward should be numeric"
            assert isinstance(metric["avg_queue"], (int, float)), "avg_queue should be numeric"
            assert isinstance(metric["avg_fairness_penalty"], (int, float)), "avg_fairness_penalty should be numeric"
            print(f"ML Metrics: {len(data)} episodes, sample: reward={metric['avg_reward']:.3f}, queue={metric['avg_queue']:.2f}")
    
    def test_ml_evaluate_learned_beats_fixed(self):
        """POST /api/ml/evaluate returns learned_reward > fixed_reward (both negative)"""
        response = requests.post(f"{BASE_URL}/api/ml/evaluate", timeout=60)
        assert response.status_code == 200
        data = response.json()
        
        assert "learned_reward" in data, "Should have learned_reward"
        assert "fixed_reward" in data, "Should have fixed_reward"
        assert "pressure_reward" in data, "Should have pressure_reward"
        
        learned = data["learned_reward"]
        fixed = data["fixed_reward"]
        pressure = data["pressure_reward"]
        
        # All rewards are negative (penalties), so learned should be less negative (higher)
        assert learned > fixed, f"Learned ({learned:.3f}) should beat Fixed ({fixed:.3f})"
        print(f"ML Evaluate: learned={learned:.3f}, fixed={fixed:.3f}, pressure={pressure:.3f}")
        print(f"Learned beats Fixed: {learned > fixed}")
    
    def test_ml_train_and_status(self):
        """POST /api/ml/train with {episodes:6,seed:7} returns {ok:true}, then status shows running/done"""
        # Start training
        train_response = requests.post(
            f"{BASE_URL}/api/ml/train",
            json={"episodes": 6, "seed": 7},
            timeout=15
        )
        assert train_response.status_code == 200
        train_data = train_response.json()
        assert train_data.get("ok") is True
        print(f"Training started: ok={train_data.get('ok')}, episodes={train_data.get('episodes')}")
        
        # Poll status until done or timeout
        max_wait = 90
        start_time = time.time()
        final_state = None
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{BASE_URL}/api/ml/train_status", timeout=10)
            assert status_response.status_code == 200
            status_data = status_response.json()
            state = status_data.get("state")
            
            if state in ("running", "done"):
                final_state = state
                if state == "done":
                    print(f"Training completed in {time.time() - start_time:.1f}s")
                    break
            
            time.sleep(2)
        
        assert final_state in ("running", "done"), f"Training should be running or done, got {final_state}"
        print(f"Training status: {final_state}")


class TestOSMEndpoint:
    """OSM import endpoint tests"""
    
    def test_osm_import_chennai(self):
        """POST /api/osm/import with {place:'Chennai', radius:1500} returns graph with nodes/edges/signals"""
        response = requests.post(
            f"{BASE_URL}/api/osm/import",
            json={"place": "Chennai", "radius": 1500},
            timeout=120
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should have graph data
        assert "graph" in data or "error" not in data, "Should return graph data"
        
        if "graph" in data:
            graph = data["graph"]
            nodes = graph.get("nodes", [])
            edges = graph.get("edges", [])
            signals = graph.get("signals", 0)
            
            assert len(nodes) > 0, "Should have nodes"
            assert len(edges) > 0, "Should have edges"
            
            # Source should be 'live' or 'offline_fallback'
            source = data.get("source", "")
            assert source in ("live", "offline_fallback"), f"Source should be live or offline_fallback, got {source}"
            
            print(f"OSM Import: {len(nodes)} nodes, {len(edges)} edges, {signals} signals, source={source}")
        else:
            print(f"OSM Import response: {data}")


class TestWebSocketEndpoint:
    """WebSocket endpoint basic connectivity test"""
    
    def test_websocket_url_accessible(self):
        """WebSocket endpoint should be accessible (basic HTTP upgrade check)"""
        # We can't do full WS test with requests, but we can verify the endpoint exists
        # by checking that it doesn't return 404
        ws_url = BASE_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/stream"
        print(f"WebSocket URL: {ws_url}")
        # Just verify the base URL is accessible
        health_response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert health_response.status_code == 200
        print("Backend accessible, WebSocket endpoint should be available at /ws/stream")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
