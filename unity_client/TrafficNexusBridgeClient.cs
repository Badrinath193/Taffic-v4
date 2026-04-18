/*
  TrafficNexusBridgeClient.cs
  Unity client for the Traffic Nexus WebSocket bridge.

  Renders vehicles and traffic lights broadcast by the backend. Works with
  both the synthetic 3x3 grid AND real OSM networks loaded via the
  /api/osm/load_sim endpoint — whenever the backend sends a new layout
  (different node IDs), this client rebuilds its road/intersection geometry
  to match the incoming city.

  Dependencies:
    - NativeWebSocket (https://github.com/endel/NativeWebSocket) — install via UPM

  Attach this component to a GameObject in your scene and fill the prefab
  slots. Press Play. The scene will stay in sync with the Python backend.
*/

using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

[Serializable] public class TNNode { public string id; public float x; public float y; }
[Serializable] public class TNEdge { public string from; public string to; }
[Serializable] public class TNVehicle {
    public int id; public string t;
    public float fx, fy, tx, ty, p, s;
    public bool st;
}
[Serializable] public class TNTL { public string nid; public int phase; }
[Serializable] public class TNSnapshot {
    public int step;
    public List<TNNode> nodes;
    public List<TNEdge> edges;
    public List<TNTL> tls;
    public List<TNVehicle> vehicles;
}
[Serializable] public class TNWrapper {
    public string type;
    public TNSnapshot snapshot;
    public string policy;
    public bool osm_loaded;
    public string place;
}

public class TrafficNexusBridgeClient : MonoBehaviour {
    [Header("Connection")]
    public string wsUrl = "ws://localhost:8001/ws/stream";

    [Header("World")]
    public float worldScale = 0.18f;
    public GameObject vehiclePrefab;
    public GameObject trafficLightPrefab;
    public Material roadMaterial;

    private WebSocket ws;
    private readonly Dictionary<int, GameObject> vehicleObjs = new();
    private readonly Dictionary<string, GameObject> tlObjs = new();
    private readonly List<GameObject> roadObjs = new();
    private string lastLayoutKey = "";
    private Vector2 center = Vector2.zero;

    async void Start() {
        ws = new WebSocket(wsUrl);
        ws.OnOpen += () => Debug.Log("[TN] WebSocket connected");
        ws.OnError += (e) => Debug.LogError($"[TN] WS error: {e}");
        ws.OnMessage += (bytes) => HandleMessage(System.Text.Encoding.UTF8.GetString(bytes));
        await ws.Connect();
    }

    void Update() {
        #if !UNITY_WEBGL || UNITY_EDITOR
        ws?.DispatchMessageQueue();
        #endif
    }

    async void OnApplicationQuit() {
        if (ws != null) await ws.Close();
    }

    void HandleMessage(string json) {
        try {
            var msg = JsonUtility.FromJson<TNWrapper>(json);
            if (msg == null || msg.type != "snapshot" || msg.snapshot == null) return;
            if (msg.osm_loaded) Debug.Log($"[TN] OSM city loaded: {msg.place}");
            ApplySnapshot(msg.snapshot);
        } catch (Exception ex) {
            Debug.LogWarning($"[TN] parse error: {ex.Message}");
        }
    }

    Vector3 ToWorld(float x, float y) =>
        new Vector3((x - center.x) * worldScale, 0f, (y - center.y) * worldScale);

    void ApplySnapshot(TNSnapshot snap) {
        if (snap.nodes == null || snap.nodes.Count == 0) return;

        // --- detect layout change (new city loaded) ---
        var layoutKey = snap.nodes.Count + ":" + snap.nodes[0].id + ":" + (snap.edges?.Count ?? 0);
        if (layoutKey != lastLayoutKey) {
            RebuildWorld(snap);
            lastLayoutKey = layoutKey;
        }

        // --- vehicles (dynamic) ---
        var seen = new HashSet<int>();
        foreach (var v in snap.vehicles) {
            seen.Add(v.id);
            if (!vehicleObjs.TryGetValue(v.id, out var go)) {
                go = Instantiate(vehiclePrefab);
                go.name = $"Veh_{v.id}_{v.t}";
                vehicleObjs[v.id] = go;
            }
            var p0 = ToWorld(v.fx, v.fy);
            var p1 = ToWorld(v.tx, v.ty);
            go.transform.position = Vector3.Lerp(p0, p1, v.p) + new Vector3(0, 0.5f, 0);
            var dir = (p1 - p0);
            if (dir.sqrMagnitude > 1e-4f) go.transform.rotation = Quaternion.LookRotation(dir);
        }
        var stale = new List<int>();
        foreach (var kv in vehicleObjs) if (!seen.Contains(kv.Key)) stale.Add(kv.Key);
        foreach (var id in stale) { Destroy(vehicleObjs[id]); vehicleObjs.Remove(id); }

        // --- traffic light phases ---
        if (snap.tls != null) {
            foreach (var t in snap.tls) {
                if (!tlObjs.TryGetValue(t.nid, out var go)) continue;
                var r = go.GetComponentInChildren<Renderer>();
                if (r != null) {
                    bool green = (t.phase == 0 || t.phase == 2);
                    r.material.color = green ? Color.green : new Color(1f, 0.82f, 0.2f);
                }
            }
        }
    }

    void RebuildWorld(TNSnapshot snap) {
        // dispose previous geometry
        foreach (var r in roadObjs) Destroy(r);
        roadObjs.Clear();
        foreach (var kv in tlObjs) Destroy(kv.Value);
        tlObjs.Clear();

        // recenter
        float minX = float.MaxValue, maxX = float.MinValue, minY = float.MaxValue, maxY = float.MinValue;
        foreach (var n in snap.nodes) {
            if (n.x < minX) minX = n.x;
            if (n.x > maxX) maxX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.y > maxY) maxY = n.y;
        }
        center = new Vector2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);

        var byId = new Dictionary<string, TNNode>();
        foreach (var n in snap.nodes) byId[n.id] = n;

        // roads
        if (snap.edges != null) {
            foreach (var e in snap.edges) {
                if (!byId.TryGetValue(e.from, out var a)) continue;
                if (!byId.TryGetValue(e.to, out var b)) continue;
                var pa = ToWorld(a.x, a.y);
                var pb = ToWorld(b.x, b.y);
                var dir = pb - pa;
                var len = dir.magnitude;
                if (len < 0.1f) continue;
                var road = GameObject.CreatePrimitive(PrimitiveType.Cube);
                road.transform.localScale = new Vector3(len, 0.05f, 3f);
                road.transform.position = (pa + pb) * 0.5f;
                road.transform.rotation = Quaternion.LookRotation(dir) * Quaternion.Euler(0, 90, 0);
                var rend = road.GetComponent<Renderer>();
                if (roadMaterial != null) rend.material = roadMaterial;
                else rend.material.color = new Color(0.1f, 0.13f, 0.22f);
                roadObjs.Add(road);
            }
        }

        // traffic lights at intersections
        if (snap.tls != null) {
            foreach (var t in snap.tls) {
                if (!byId.TryGetValue(t.nid, out var n)) continue;
                var go = Instantiate(trafficLightPrefab);
                go.transform.position = ToWorld(n.x, n.y);
                go.name = $"TL_{t.nid}";
                tlObjs[t.nid] = go;
            }
        }

        Debug.Log($"[TN] Rebuilt world: {snap.nodes.Count} nodes, {(snap.edges?.Count ?? 0)} edges, {(snap.tls?.Count ?? 0)} signals");
    }

    // ---- Client -> Server commands ----
    public async void SetPolicy(string policy) {
        if (ws != null && ws.State == WebSocketState.Open)
            await ws.SendText($"{{\"type\":\"cmd\",\"action\":\"set_policy\",\"value\":\"{policy}\"}}");
    }
    public async void Pause() {
        if (ws != null && ws.State == WebSocketState.Open)
            await ws.SendText("{\"type\":\"cmd\",\"action\":\"pause\"}");
    }
    public async void Resume() {
        if (ws != null && ws.State == WebSocketState.Open)
            await ws.SendText("{\"type\":\"cmd\",\"action\":\"resume\"}");
    }
}
