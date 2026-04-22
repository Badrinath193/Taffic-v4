import sys
import subprocess
from unittest.mock import MagicMock

# Create a mock for requests and its attributes
class MockRequests:
    def post(self, *args, **kwargs):
        pass
    def get(self, *args, **kwargs):
        pass

sys.modules['requests'] = MockRequests()

import backend.osm as osm

# Let's test the function directly instead
def test_import_osm_cache_hit():
    cache_get = MagicMock(return_value={"data": "cached_data"})
    cache_put = MagicMock()

    result = osm.import_osm("Chennai", 1000, cache_get, cache_put)
    assert result == {"data": "cached_data", "cache": "hit"}
    print("test_import_osm_cache_hit passed!")

def test_import_osm_offline_fallback():
    # Make live data fail
    osm.geocode = MagicMock(side_effect=Exception("mocked geocode error"))

    # Mock load_offline_fallback
    osm.load_offline_fallback = MagicMock(return_value={"location": {"lat": 1, "lon": 2}, "graph": {"nodes": []}})

    result = osm.import_osm("Chennai", 1000)
    assert result["source"] == "offline_fallback"
    assert "mocked geocode error" in result["live_error"]
    print("test_import_osm_offline_fallback passed!")

def test_import_osm_live():
    # Mock geocode to return valid location
    osm.geocode = MagicMock(return_value={"lat": 13.0, "lon": 80.0})

    # Mock build_query
    osm.build_query = MagicMock(return_value="mock query")

    # Mock query_overpass
    osm.query_overpass = MagicMock(return_value={"endpoint": "mock_endpoint", "data": {}})

    # Mock parse_overpass to return valid graph with nodes
    osm.parse_overpass = MagicMock(return_value={"nodes": [{"id": 1}], "edges": [], "signals": 0, "source": "osm"})

    result = osm.import_osm("Chennai", 1000)

    assert result["source"] == "live"
    assert result["cache"] == "miss"
    assert result["graph"]["nodes"] == [{"id": 1}]
    print("test_import_osm_live passed!")

if __name__ == "__main__":
    test_import_osm_cache_hit()
    test_import_osm_offline_fallback()
    test_import_osm_live()
    print("All tests passed!")
