"""
OSM import with robust mirror fallback, exponential backoff and MongoDB caching.
Also ships an offline fallback dataset for when all mirrors fail.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


NOMINATIM_URLS = [
    "https://nominatim.openstreetmap.org/search",
    "https://nominatim.openstreetmap.de/search",
]
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
USER_AGENT = "TrafficNexus/2.0 (research; contact=traffic-nexus@example.com)"


def _backoff_sleep(attempt: int):
    time.sleep(min(8.0, 0.5 * (2 ** attempt)))


def geocode(place: str, session: Optional[requests.Session] = None) -> Dict[str, object]:
    s = session or requests.Session()
    last_err = None
    for url in NOMINATIM_URLS:
        for attempt in range(2):
            try:
                resp = s.get(
                    url,
                    params={"q": place, "format": "json", "limit": 1, "addressdetails": 1},
                    headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                    timeout=20,
                )
                if resp.status_code == 429:
                    _backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    raise RuntimeError(f"Place not found: {place}")
                item = data[0]
                return {
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "display_name": item["display_name"],
                    "source": url,
                }
            except Exception as exc:
                last_err = exc
                _backoff_sleep(attempt)
    raise RuntimeError(f"Geocoding failed for '{place}': {last_err}")


def build_query(lat: float, lon: float, radius: int = 1800) -> str:
    return f"""[out:json][timeout:40];
(
  node["highway"="traffic_signals"](around:{radius},{lat},{lon});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified)$"](around:{radius},{lat},{lon});
);
out body;
>;
out skel qt;"""


def query_overpass(
    query: str,
    endpoints: Optional[Iterable[str]] = None,
    session: Optional[requests.Session] = None,
    max_attempts_per_endpoint: int = 2,
) -> Dict[str, object]:
    s = session or requests.Session()
    errors: List[str] = []
    for endpoint in list(endpoints or OVERPASS_ENDPOINTS):
        for attempt in range(max_attempts_per_endpoint):
            try:
                resp = s.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=60,
                )
                if resp.status_code in (429, 504, 502, 503):
                    errors.append(f"{endpoint}: HTTP {resp.status_code}")
                    _backoff_sleep(attempt + 1)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if not data.get("elements"):
                    errors.append(f"{endpoint}: empty elements")
                    continue
                return {"endpoint": endpoint, "data": data}
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
                _backoff_sleep(attempt)
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))


def parse_overpass(data: Dict, center_lat: float, center_lon: float) -> Dict:
    """Convert Overpass JSON -> simulation graph with coordinates normalized to meters."""
    nodes_raw = {el["id"]: el for el in data.get("elements", []) if el["type"] == "node"}
    ways = [el for el in data.get("elements", []) if el["type"] == "way"]
    signal_ids = {
        el["id"] for el in data.get("elements", [])
        if el["type"] == "node" and el.get("tags", {}).get("highway") == "traffic_signals"
    }

    def to_xy(lat, lon):
        # simple equirectangular projection centered at (center_lat, center_lon)
        R = 6371000.0
        x = math.radians(lon - center_lon) * math.cos(math.radians(center_lat)) * R
        y = -math.radians(lat - center_lat) * R   # flip so north is up in screen coords negative
        return x, y

    used_nodes = set()
    for w in ways:
        for nid in w.get("nodes", []):
            used_nodes.add(nid)

    out_nodes = []
    for nid in used_nodes:
        if nid not in nodes_raw:
            continue
        n = nodes_raw[nid]
        x, y = to_xy(n["lat"], n["lon"])
        out_nodes.append({
            "id": f"O{nid}", "x": x, "y": y,
            "is_signal": nid in signal_ids,
            "lat": n["lat"], "lon": n["lon"],
        })
    out_edges = []
    for w in ways:
        nds = w.get("nodes", [])
        hw = w.get("tags", {}).get("highway", "residential")
        oneway = w.get("tags", {}).get("oneway", "no") == "yes"
        for a, b in zip(nds[:-1], nds[1:]):
            if a in nodes_raw and b in nodes_raw:
                out_edges.append({"from": f"O{a}", "to": f"O{b}", "highway": hw})
                if not oneway:
                    out_edges.append({"from": f"O{b}", "to": f"O{a}", "highway": hw})
    return {
        "nodes": out_nodes,
        "edges": out_edges,
        "signals": len(signal_ids),
        "source": "osm",
    }


OFFLINE_FALLBACK_FILE = Path(__file__).parent / "offline_osm.json"


def load_offline_fallback(place_hint: str = "chennai") -> Optional[Dict]:
    if not OFFLINE_FALLBACK_FILE.exists():
        return None
    try:
        with open(OFFLINE_FALLBACK_FILE) as f:
            blob = json.load(f)
        key = place_hint.strip().lower().split(",")[0]
        for k, v in blob.items():
            if k in key or key in k:
                return v
        # default first
        return next(iter(blob.values()))
    except Exception:
        return None


def cache_key(place: str, radius: int) -> str:
    h = hashlib.sha1(f"{place.lower().strip()}|{radius}".encode()).hexdigest()[:16]
    return f"osm:{h}"


def import_osm(place: str, radius: int, cache_get=None, cache_put=None) -> Dict:
    """High-level OSM import with caching + offline fallback."""
    ck = cache_key(place, radius)
    if cache_get:
        hit = cache_get(ck)
        if hit:
            hit["cache"] = "hit"
            return hit
    try:
        loc = geocode(place)
        query = build_query(loc["lat"], loc["lon"], radius)
        raw = query_overpass(query)
        graph = parse_overpass(raw["data"], loc["lat"], loc["lon"])
        result = {
            "place": place,
            "location": loc,
            "radius": radius,
            "overpass_endpoint": raw["endpoint"],
            "graph": graph,
            "cache": "miss",
            "source": "live",
        }
        if cache_put:
            cache_put(ck, result)
        return result
    except Exception as live_err:
        fallback = load_offline_fallback(place)
        if fallback:
            return {
                "place": place,
                "location": fallback.get("location"),
                "radius": radius,
                "graph": fallback["graph"],
                "cache": "miss",
                "source": "offline_fallback",
                "live_error": str(live_err),
            }
        raise
