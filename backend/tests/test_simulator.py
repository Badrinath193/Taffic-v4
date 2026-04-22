import pytest
from unittest.mock import patch, MagicMock

class TestVehicleSim:
    def test_load_from_osm_empty_graph(self):
        # Mock numpy locally to prevent ModuleNotFoundError in restricted environments
        # without globally polluting sys.modules
        with patch.dict('sys.modules', {'numpy': MagicMock()}):
            from simulator import VehicleSim

            sim = VehicleSim()

            # Test with empty dict
            with pytest.raises(ValueError, match="OSM graph has no nodes or edges"):
                sim.load_from_osm({})

            # Test with dict having empty nodes and edges
            with pytest.raises(ValueError, match="OSM graph has no nodes or edges"):
                sim.load_from_osm({"nodes": [], "edges": []})

            # Test with dict having empty nodes but some edges
            with pytest.raises(ValueError, match="OSM graph has no nodes or edges"):
                sim.load_from_osm({"nodes": [], "edges": [{"from": "n1", "to": "n2"}]})

            # Test with dict having some nodes but empty edges
            with pytest.raises(ValueError, match="OSM graph has no nodes or edges"):
                sim.load_from_osm({"nodes": [{"id": "n1"}], "edges": []})
