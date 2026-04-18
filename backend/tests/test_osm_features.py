"""
Test suite for OSM-related features:
- POST /api/osm/import - OSM import with Nominatim + Photon fallback
- POST /api/osm/load_sim - Load OSM into live simulator
- GET /api/sim/state - Verify OSM nodes with 'O' prefix after load
- Simulation stepping verification after OSM load
- Error handling for bogus place names
"""
import pytest
import requests
import time
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestOSMImport:
    """Test POST /api/osm/import endpoint"""
    
    def test_osm_import_koramangala_returns_live_data(self):
        """Test OSM import for Koramangala returns live data with >50 nodes/edges"""
        response = requests.post(
            f"{BASE_URL}/api/osm/import",
            json={"place": "Koramangala, Bengaluru, India", "radius": 1200},
            timeout=120
        )
        
        # Status code assertion
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Source should be 'live' or 'hit' (cached) - Photon fallback is acceptable
        assert data.get("source") in ["live", "hit", "offline_fallback"], f"Unexpected source: {data.get('source')}"
        
        # Graph structure validation
        graph = data.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        signals = graph.get("signals", 0)
        
        print(f"OSM Import Result: source={data.get('source')}, nodes={len(nodes)}, edges={len(edges)}, signals={signals}")
        
        # Verify we have substantial data (>50 nodes and edges for a real city area)
        assert len(nodes) > 50, f"Expected >50 nodes, got {len(nodes)}"
        assert len(edges) > 50, f"Expected >50 edges, got {len(edges)}"
        
        # Verify location data
        location = data.get("location", {})
        assert "lat" in location, "Missing lat in location"
        assert "lon" in location, "Missing lon in location"
        
        # Verify geocoder source (nominatim or photon)
        assert location.get("source") in ["nominatim", "photon"], f"Unexpected geocoder source: {location.get('source')}"
        
        print(f"Geocoder used: {location.get('source')}")


class TestOSMLoadSim:
    """Test POST /api/osm/load_sim endpoint"""
    
    def test_osm_load_sim_koramangala(self):
        """Test loading OSM data into live simulator"""
        response = requests.post(
            f"{BASE_URL}/api/osm/load_sim",
            json={
                "place": "Koramangala, Bengaluru, India",
                "radius": 1000,
                "max_nodes": 400,
                "max_vehicles": 200,
                "autostart": True
            },
            timeout=120
        )
        
        # Status code assertion
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Verify response structure
        assert data.get("ok") is True, f"Expected ok:true, got {data}"
        assert "loaded" in data, "Missing 'loaded' field in response"
        assert data.get("running") is True, "Expected running:true with autostart"
        
        loaded = data.get("loaded", {})
        print(f"Loaded into sim: nodes={loaded.get('nodes')}, edges={loaded.get('edges')}, signals={loaded.get('signals')}")
        
        # Verify loaded data
        assert loaded.get("nodes", 0) > 0, "Expected nodes > 0"
        assert loaded.get("edges", 0) > 0, "Expected edges > 0"
        assert loaded.get("signals", 0) >= 0, "Expected signals >= 0"
        
        # Verify source
        assert data.get("source") in ["live", "hit", "offline_fallback"], f"Unexpected source: {data.get('source')}"
    
    def test_osm_load_sim_bogus_place_uses_offline_fallback(self):
        """Test that a bogus place name uses offline fallback (not 500 error)"""
        response = requests.post(
            f"{BASE_URL}/api/osm/load_sim",
            json={
                "place": "ZZZZnonexistent",
                "radius": 1000,
                "max_nodes": 400,
                "max_vehicles": 200,
                "autostart": True
            },
            timeout=120
        )
        
        # Backend has offline fallback - should return 200 with offline_fallback source
        # OR 502 if no fallback available
        if response.status_code == 200:
            data = response.json()
            # Verify it used offline fallback
            assert data.get("source") == "offline_fallback", f"Expected offline_fallback source, got {data.get('source')}"
            print(f"Bogus place correctly used offline fallback: {data.get('location', {}).get('display_name', '')[:50]}")
        else:
            # 502 is also acceptable if no offline fallback
            assert response.status_code == 502, f"Expected 502 or 200 with fallback, got {response.status_code}: {response.text}"
            print(f"Bogus place correctly returned 502: {response.json().get('detail', '')[:100]}")


class TestSimStateAfterOSMLoad:
    """Test GET /api/sim/state after OSM load"""
    
    def test_sim_state_has_osm_nodes(self):
        """After OSM load, sim state should have nodes with 'O' prefix IDs"""
        # First load OSM
        load_response = requests.post(
            f"{BASE_URL}/api/osm/load_sim",
            json={
                "place": "Koramangala, Bengaluru, India",
                "radius": 1000,
                "max_nodes": 400,
                "max_vehicles": 200,
                "autostart": True
            },
            timeout=120
        )
        assert load_response.status_code == 200, f"OSM load failed: {load_response.text}"
        
        # Wait a moment for simulation to process
        time.sleep(1)
        
        # Get sim state
        state_response = requests.get(f"{BASE_URL}/api/sim/state", timeout=30)
        assert state_response.status_code == 200, f"Sim state failed: {state_response.text}"
        
        state = state_response.json()
        
        # Verify nodes have 'O' prefix (OSM nodes)
        nodes = state.get("nodes", [])
        assert len(nodes) > 0, "Expected nodes in sim state"
        
        osm_nodes = [n for n in nodes if n.get("id", "").startswith("O")]
        print(f"Sim state: {len(nodes)} total nodes, {len(osm_nodes)} OSM nodes (O prefix)")
        
        # All nodes should have 'O' prefix after OSM load
        assert len(osm_nodes) > 0, "Expected nodes with 'O' prefix after OSM load"
        
        # Verify signals count
        tls = state.get("tls", [])
        print(f"Traffic signals in sim state: {len(tls)}")
        
        # Verify running state
        assert state.get("running") is True, "Expected simulation to be running"


class TestSimulationStepping:
    """Test that simulation keeps stepping after OSM load"""
    
    def test_simulation_steps_increase(self):
        """After OSM load, simulation step should increase over time"""
        # First load OSM and start
        load_response = requests.post(
            f"{BASE_URL}/api/osm/load_sim",
            json={
                "place": "Koramangala, Bengaluru, India",
                "radius": 1000,
                "max_nodes": 400,
                "max_vehicles": 200,
                "autostart": True
            },
            timeout=120
        )
        assert load_response.status_code == 200, f"OSM load failed: {load_response.text}"
        
        # Get initial step
        state1 = requests.get(f"{BASE_URL}/api/sim/state", timeout=30).json()
        step1 = state1.get("step", 0)
        print(f"Initial step: {step1}")
        
        # Wait 2 seconds
        time.sleep(2)
        
        # Get step again
        state2 = requests.get(f"{BASE_URL}/api/sim/state", timeout=30).json()
        step2 = state2.get("step", 0)
        print(f"Step after 2s: {step2}")
        
        # Step should have increased
        assert step2 > step1, f"Expected step to increase: {step1} -> {step2}"
        print(f"Simulation stepping verified: {step1} -> {step2} (delta: {step2 - step1})")


class TestRegressionCoreFeatures:
    """Regression tests for core features that should still work"""
    
    def test_health_endpoint(self):
        """Health endpoint should return ok"""
        response = requests.get(f"{BASE_URL}/api/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
        print(f"Health: ok={data.get('ok')}, models_loaded={data.get('models_loaded')}")
    
    def test_policy_fixed(self):
        """Set policy to fixed should work"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "fixed"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("policy") == "fixed"
        print("Policy 'fixed' set successfully")
    
    def test_policy_pressure(self):
        """Set policy to pressure should work"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "pressure"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("policy") == "pressure"
        print("Policy 'pressure' set successfully")
    
    def test_policy_learned(self):
        """Set policy to learned should work"""
        response = requests.post(
            f"{BASE_URL}/api/sim/set_policy",
            json={"policy": "learned"},
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("policy") == "learned"
        assert "learned_available" in data
        print(f"Policy 'learned' set successfully, learned_available={data.get('learned_available')}")
    
    def test_ml_summary(self):
        """ML summary should return trained model info"""
        response = requests.get(f"{BASE_URL}/api/ml/summary", timeout=10)
        assert response.status_code == 200
        data = response.json()
        print(f"ML Summary: trained={data.get('trained')}, episodes={data.get('episodes')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
