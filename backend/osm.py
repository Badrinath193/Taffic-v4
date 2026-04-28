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
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


NOMINATIM_URLS = [
    "https://nominatim.openstreetmap.org/search",
]
# Photon is a separate Nominatim-compatible geocoder also backed by OSM.
PHOTON_URL = "https://photon.komoot.io/api/"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
USER_AGENT = "TrafficNexusLocal/2.0 (educational demo; contact: traffic-nexus@example.invalid)"


def _backoff_sleep(attempt: int):
    time.sleep(min(8.0, 0.4 * (2 ** attempt)))


def _geocode_nominatim(s: requests.Session, place: str) -> Optional[Dict]:
    for attempt in range(2):
        try:
            resp = s.get(
                NOMINATIM_URLS[0],
                params={"q": place, "format": "json", "limit": 1, "addressdetails": 1},
                headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                timeout=20,
            )
            if resp.status_code == 429:
                _backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data:
                item = data[0]
                return {
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "display_name": item.get("display_name", place),
                    "source": "nominatim",
                }
        except Exception as exc:
            print(f"[osm] nominatim attempt {attempt} failed: {exc}")
            _backoff_sleep(attempt)
    return None


def _geocode_photon(s: requests.Session, place: str) -> Optional[Dict]:
    try:
        resp = s.get(
            PHOTON_URL,
            params={"q": place, "limit": 1},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        feats = data.get("features") or []
        if feats:
            f = feats[0]
            coords = f.get("geometry", {}).get("coordinates") or []
            if len(coords) >= 2:
                props = f.get("properties", {})
                name_parts = [p for p in [props.get("name"), props.get("city"), props.get("country")] if p]
                return {
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                    "display_name": ", ".join(name_parts) or place,
                    "source": "photon",
                }
    except Exception as exc:
        print(f"[osm] photon failed: {exc}")
    return None


def geocode(place: str, session: Optional[requests.Session] = None) -> Dict[str, object]:
    s = session or requests.Session()
    # Try Nominatim first, then Photon (Komoot) as an independent backup.
    res = _geocode_nominatim(s, place)
    if res:
        return res
    res = _geocode_photon(s, place)
    if res:
        return res
    raise RuntimeError(f"Geocoding failed for '{place}' on all providers (Nominatim, Photon).")


def build_query(lat: float, lon: float, radius: int = 1800) -> str:
    return f"""[out:json][timeout:40];
(
  node["highway"="traffic_signals"](around:{radius},{lat},{lon});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified|living_street)$"](around:{radius},{lat},{lon});
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
                errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
                _backoff_sleep(attempt)
    raise RuntimeError("All Overpass endpoints failed → " + " | ".join(errors))


def parse_overpass(data: Dict, center_lat: float, center_lon: float) -> Dict:
    """Convert Overpass JSON -> simulation graph with coordinates normalized to meters."""
    nodes_raw = {el["id"]: el for el in data.get("elements", []) if el["type"] == "node"}
    ways = [el for el in data.get("elements", []) if el["type"] == "way"]
    signal_ids = {
        el["id"] for el in data.get("elements", [])
        if el["type"] == "node" and el.get("tags", {}).get("highway") == "traffic_signals"
    }

    def to_xy(lat, lon):
        R = 6371000.0
        x = math.radians(lon - center_lon) * math.cos(math.radians(center_lat)) * R
        y = -math.radians(lat - center_lat) * R
        return x, y

    # --- pick only nodes referenced by at least one way ---
    used_nodes = set()
    for w in ways:
        for nid in w.get("nodes", []):
            used_nodes.add(nid)

    out_nodes = []
    id_to_internal: Dict[int, str] = {}
    for nid in used_nodes:
        if nid not in nodes_raw:
            continue
        n = nodes_raw[nid]
        x, y = to_xy(n["lat"], n["lon"])
        internal = f"O{nid}"
        id_to_internal[nid] = internal
        out_nodes.append({
            "id": internal, "x": x, "y": y,
            "is_signal": nid in signal_ids,
            "lat": n["lat"], "lon": n["lon"],
        })

    out_edges = []
    for w in ways:
        nds = w.get("nodes", [])
        hw = w.get("tags", {}).get("highway", "residential")
        oneway = w.get("tags", {}).get("oneway", "no") == "yes"
        for a, b in zip(nds[:-1], nds[1:]):
            if a in id_to_internal and b in id_to_internal:
                out_edges.append({"from": id_to_internal[a], "to": id_to_internal[b], "highway": hw})
                if not oneway:
                    out_edges.append({"from": id_to_internal[b], "to": id_to_internal[a], "highway": hw})

    return {
        "nodes": out_nodes,
        "edges": out_edges,
        "signals": sum(1 for n in out_nodes if n["is_signal"]),
        "source": "osm",
    }


OFFLINE_FALLBACK_FILE = Path(__file__).parent / "offline_osm.json"


def load_offline_fallback(place_hint: str = "chennai") -> Optional[Dict]:
    if not OFFLINE_FALLBACK_FILE.exists():
        return None
    try:
        with open(OFFLINE_FALLBACK_FILE) as f:
            blob = json.load(f)
        key = place_hint.strip().lower().split(",")[0].strip()
        for k, v in blob.items():
            if k in key or key in k:
                return v
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
    live_errors: List[str] = []
    try:
        if not re.match(r"^[\w\s,.\-']+$", place):
            raise ValueError("Invalid place name format. Only alphanumeric characters and basic punctuation are allowed.")
        loc = geocode(place)
    except Exception as exc:
        live_errors.append(f"geocode: {exc}")
        loc = None

    if loc is not None:
        try:
            query = build_query(loc["lat"], loc["lon"], radius)
            raw = query_overpass(query)
            graph = parse_overpass(raw["data"], loc["lat"], loc["lon"])
            if graph["nodes"]:
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
            live_errors.append("overpass returned empty graph")
        except Exception as exc:
            live_errors.append(f"overpass: {exc}")

    fallback = load_offline_fallback(place)
    if fallback:
        return {
            "place": place,
            "location": fallback.get("location"),
            "radius": radius,
            "graph": fallback["graph"],
            "cache": "miss",
            "source": "offline_fallback",
            "live_error": " | ".join(live_errors) if live_errors else None,
        }
    raise RuntimeError("OSM import failed: " + " | ".join(live_errors))
