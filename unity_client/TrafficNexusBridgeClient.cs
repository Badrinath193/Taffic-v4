/*
  TrafficNexusBridgeClient.cs
  Sample Unity client for the Traffic Nexus WebSocket protocol.

  Attach this MonoBehaviour to a GameObject. Provide the WebSocket URL
  (e.g. ws://localhost:8001/ws/stream) in the inspector. The script
  maintains a pool of vehicle prefabs and updates their positions every
  time a "snapshot" message arrives from the backend.

  Dependencies:
    - NativeWebSocket (https://github.com/endel/NativeWebSocket) -- add via UPM.

  Protocol (matches backend unity_bridge.py):
    Server -> Client:
      { "type": "hello", "msg": "..." }
      { "type": "snapshot", "snapshot": {...}, "v2x": [...], "metrics": {...}, "policy": "..." }
    Client -> Server:
      { "type": "cmd", "action": "pause" | "resume" | "set_policy", "value": "learned" }
*/

using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;   // UPM: com.endel.nativewebsocket

[Serializable]
public class TNVehicle {
    public int id;
    public string t;       // type: car|truck|bus|motorcycle|emergency
    public float fx, fy, tx, ty;
    public float p;        // progress along edge [0..1]
    public float s;
    public bool st;
}

[Serializable]
public class TNTL {
    public string nid;
    public int phase;      // 0..3
}

[Serializable]
public class TNSnapshot {
    public int step;
    public List<TNVehicle> vehicles;
    public List<TNTL> tls;
    public List<Dictionary<string, object>> nodes;
    public List<Dictionary<string, object>> edges;
}

[Serializable]
public class TNMessage {
    public string type;
    public TNSnapshot snapshot;
}

public class TrafficNexusBridgeClient : MonoBehaviour {
    public string wsUrl = "ws://localhost:8001/ws/stream";
    public GameObject vehiclePrefab;
    public GameObject trafficLightPrefab;
    public float worldScale = 0.2f;

    private WebSocket ws;
    private readonly Dictionary<int, GameObject> vehicleObjs = new();
    private readonly Dictionary<string, GameObject> tlObjs = new();

    async void Start() {
        ws = new WebSocket(wsUrl);
        ws.OnOpen += () => Debug.Log("[TN] WebSocket connected");
        ws.OnError += (e) => Debug.LogError($"[TN] WS error: {e}");
        ws.OnClose += (c) => Debug.Log($"[TN] WS closed: {c}");
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
        var msg = JsonUtility.FromJson<TNMessage>(json);
        if (msg == null || msg.type != "snapshot" || msg.snapshot == null) return;
        ApplySnapshot(msg.snapshot);
    }

    void ApplySnapshot(TNSnapshot snap) {
        // Vehicles
        var seen = new HashSet<int>();
        foreach (var v in snap.vehicles) {
            seen.Add(v.id);
            if (!vehicleObjs.TryGetValue(v.id, out var go)) {
                go = Instantiate(vehiclePrefab);
                vehicleObjs[v.id] = go;
            }
            float x = (v.fx + (v.tx - v.fx) * v.p) * worldScale;
            float z = (v.fy + (v.ty - v.fy) * v.p) * worldScale;
            go.transform.position = new Vector3(x, 0.5f, z);
            go.transform.rotation = Quaternion.LookRotation(new Vector3(v.tx - v.fx, 0, v.ty - v.fy));
        }
        // remove stale
        var remove = new List<int>();
        foreach (var kv in vehicleObjs)
            if (!seen.Contains(kv.Key)) remove.Add(kv.Key);
        foreach (var id in remove) { Destroy(vehicleObjs[id]); vehicleObjs.Remove(id); }

        // Traffic lights
        foreach (var t in snap.tls) {
            if (!tlObjs.TryGetValue(t.nid, out var go)) {
                go = Instantiate(trafficLightPrefab);
                tlObjs[t.nid] = go;
            }
            var r = go.GetComponentInChildren<Renderer>();
            if (r != null) {
                Color c = (t.phase == 0 || t.phase == 2) ? Color.green : new Color(1f, 0.82f, 0.2f);
                r.material.color = c;
            }
        }
    }

    public async void SetPolicy(string policy) {
        if (ws != null && ws.State == WebSocketState.Open) {
            string json = $"{{\"type\":\"cmd\",\"action\":\"set_policy\",\"value\":\"{policy}\"}}";
            await ws.SendText(json);
        }
    }
}
