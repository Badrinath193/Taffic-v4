from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
USER_AGENT = "TrafficNexusLocal/1.0 (educational demo)"


def geocode(place: str, session: Optional[requests.Session] = None) -> Dict[str, object]:
    s = session or requests.Session()
    resp = s.get(
        NOMINATIM_URL,
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"Place not found: {place}")
    item = data[0]
    return {
        "lat": float(item["lat"]),
        "lon": float(item["lon"]),
        "display_name": item["display_name"],
    }


def build_query(lat: float, lon: float, radius: int = 1800) -> str:
    return f'''[out:json][timeout:25];
(
  node["highway"="traffic_signals"](around:{radius},{lat},{lon});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified)$"](around:{radius},{lat},{lon});
);
out body;>;out skel qt;'''


def query_overpass(query: str, endpoints: Optional[Iterable[str]] = None, session: Optional[requests.Session] = None) -> Dict[str, object]:
    s = session or requests.Session()
    errors: List[str] = []
    for endpoint in list(endpoints or OVERPASS_ENDPOINTS):
        try:
            resp = s.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=45,
            )
            resp.raise_for_status()
            return {"endpoint": endpoint, "data": resp.json()}
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))
