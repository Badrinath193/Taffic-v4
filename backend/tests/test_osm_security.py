import sys
from unittest.mock import MagicMock

# Mock requests before importing osm
sys.modules['requests'] = MagicMock()

import pytest
from backend.osm import import_osm

def test_import_osm_valid_place(monkeypatch):
    monkeypatch.setattr("backend.osm.geocode", lambda place: {"lat": 1.0, "lon": 1.0})
    monkeypatch.setattr("backend.osm.query_overpass", lambda query: {"endpoint": "mock", "data": {"elements": [{"type": "node", "id": 1, "lat": 1.0, "lon": 1.0}]}})
    monkeypatch.setattr("backend.osm.parse_overpass", lambda data, lat, lon: {"nodes": [{"id": "O1", "x": 0, "y": 0, "is_signal": False, "lat": 1.0, "lon": 1.0}], "edges": [], "signals": 0, "source": "osm"})
    monkeypatch.setattr("backend.osm.load_offline_fallback", lambda place: None)

    # Valid places
    res = import_osm("München, Germany", 100)
    assert res["place"] == "München, Germany"

def test_import_osm_invalid_place(monkeypatch):
    monkeypatch.setattr("backend.osm.load_offline_fallback", lambda place: None)
    # Invalid places with special characters or potential injection
    try:
        import_osm("place; drop table;", 100)
        assert False, "Should have raised exception"
    except RuntimeError as e:
        assert "Invalid place name format" in str(e)

    try:
        import_osm("<script>alert(1)</script>", 100)
        assert False, "Should have raised exception"
    except RuntimeError as e:
        assert "Invalid place name format" in str(e)
