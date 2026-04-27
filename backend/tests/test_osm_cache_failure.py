import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

def test_osm_import_cache_failure_handled():
    """Test that /api/osm/import handles cache get failures gracefully and proceeds with import_osm."""
    with patch("backend.server._cache_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Mongo cache read failed")

        with patch("backend.server.import_osm") as mock_import:
            mock_import.return_value = {"graph": {"nodes": [{"id": 1}], "edges": []}, "location": "TestCity"}

            with patch("backend.server._cache_put", new_callable=AsyncMock) as mock_put:
                response = client.post("/api/osm/import", json={"place": "TestCity", "radius": 100})

                assert response.status_code == 200
                data = response.json()
                assert "graph" in data
                assert data["location"] == "TestCity"

                mock_get.assert_called_once()
                mock_import.assert_called_once_with("TestCity", 100)

def test_osm_load_sim_cache_failure_handled():
    """Test that /api/osm/load_sim handles cache get failures gracefully and proceeds with import_osm."""
    with patch("backend.server._cache_get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = Exception("Mongo cache read failed")

        with patch("backend.server.import_osm") as mock_import:
            # Provide enough nodes and edges to pass `load_from_osm` validation
            mock_import.return_value = {"graph": {"nodes": [{"id": 1, "x": 0.0, "y": 0.0}, {"id": 2, "x": 10.0, "y": 10.0}], "edges": [{"from": 1, "to": 2}]}, "location": "TestCity", "source": "offline_fallback"}

            with patch("backend.server._cache_put", new_callable=AsyncMock) as mock_put:
                response = client.post("/api/osm/load_sim", json={"place": "TestCity", "radius": 100, "autostart": False})

                assert response.status_code == 200
                data = response.json()
                assert data["ok"] is True
                assert data["place"] == "TestCity"

                mock_get.assert_called_once()
                mock_import.assert_called_once_with("TestCity", 100)
