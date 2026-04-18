import React, { useEffect, useState, useRef, useCallback } from "react";
import { api, WS_URL } from "./lib/api";
import SimCanvas from "./components/SimCanvas";
import ThreeViewer from "./components/ThreeViewer";
import "./App.css";

const ARCH_LAYERS = [
  { num: "LAYER 01", name: "OSM Ingestion", desc: "Server-side Overpass with 5 mirrors, Nominatim fallback, MongoDB cache, and offline dataset for resilience.", real: "real" },
  { num: "LAYER 02", name: "Vehicle Engine", desc: "Per-vehicle kinematics. 5 types (car/truck/bus/motorcycle/emergency) with distinct lengths and max speeds.", real: "real" },
  { num: "LAYER 03", name: "MARL Policy", desc: "Shared DQN (PyTorch) across N intersections. 8-dim obs, 2 actions, replay buffer, target net, epsilon-greedy.", real: "real" },
  { num: "LAYER 04", name: "V2X Protocol", desc: "Real messages: PHASE_CHANGE, QUEUE_UPDATE, ROUTE_INTENT, EMERGENCY_PREEMPT — generated from actual state.", real: "real" },
  { num: "LAYER 05", name: "Signal Control", desc: "4-phase state machine (NS/EW green & amber) driven by policy output with amber transitions.", real: "real" },
  { num: "LAYER 06", name: "Unity Bridge", desc: "WebSocket protocol. Three.js 3D viewer consumes it live; a Unity C# client sample is included.", real: "real" },
  { num: "LAYER 07", name: "Analytics", desc: "Live throughput, queue, emissions + historical training metrics (CSV + summary.json).", real: "real" },
  { num: "LAYER 08", name: "Forecast Net", desc: "MLP trained on simulator rollouts to predict next-step NS/EW queues. Feeds into DQN observation.", real: "real" },
];

function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (typeof n !== "number") return n;
  return n.toFixed(d);
}

export default function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [running, setRunning] = useState(false);
  const [policy, setPolicy] = useState("fixed");
  const [health, setHealth] = useState(null);
  const [v2x, setV2x] = useState([]);
  const [trainStatus, setTrainStatus] = useState({ state: "idle", events: [] });
  const [summary, setSummary] = useState(null);
  const [osmResult, setOsmResult] = useState(null);
  const [osmPlace, setOsmPlace] = useState("Chennai, India");
  const [osmRadius, setOsmRadius] = useState(1500);
  const [osmLoading, setOsmLoading] = useState(false);
  const [simCfg, setSimCfg] = useState({ rows: 3, cols: 3, max_vehicles: 200 });
  const [episodes, setEpisodes] = useState(25);
  const [metrics, setMetrics] = useState([]);
  const wsRef = useRef(null);

  const connectWS = useCallback(() => {
    if (wsRef.current) return;
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => console.log("[ws] open");
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "snapshot") {
            const snap = { ...msg.snapshot, _policy: msg.policy };
            setSnapshot(snap);
            if (Array.isArray(msg.v2x) && msg.v2x.length) {
              // Keep a rolling window of the most recent messages.
              // Each broadcast carries the last ~10 real events from the backend bus.
              setV2x((prev) => {
                const keyOf = (m) => `${m.ts}|${m.t}|${m.src}|${JSON.stringify(m.p)}`;
                const seen = new Set(prev.map(keyOf));
                const fresh = msg.v2x.filter((m) => !seen.has(keyOf(m)));
                return [...fresh.reverse(), ...prev].slice(0, 60);
              });
            }
          }
        } catch (e) {
          console.warn("[ws parse]", e);
        }
      };
      ws.onclose = () => {
        wsRef.current = null;
        setTimeout(connectWS, 1400);
      };
      ws.onerror = () => ws.close();
    } catch (e) {
      console.warn(e);
    }
  }, []);

  useEffect(() => {
    connectWS();
    api.health().then(setHealth).catch(() => {});
    api.mlSummary().then((s) => s.trained && setSummary(s)).catch(() => {});
    api.mlMetrics().then(setMetrics).catch(() => {});
    api.simState().then((s) => { if (s && s.running) setRunning(true); setSnapshot(s); }).catch(() => {});
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, [connectWS]);

  // poll train status if running
  useEffect(() => {
    if (trainStatus.state !== "running") return;
    const t = setInterval(() => {
      api.mlStatus().then((s) => {
        setTrainStatus(s);
        if (s.state === "done") {
          api.mlSummary().then(setSummary).catch(() => {});
          api.mlMetrics().then(setMetrics).catch(() => {});
        }
      });
    }, 1200);
    return () => clearInterval(t);
  }, [trainStatus.state]);

  // V2X fallback polling — ensures the panel always shows live messages
  // even if the WebSocket frames don't include the v2x array.
  useEffect(() => {
    const t = setInterval(() => {
      if (!running) return;
      api.v2xTail(30).then((msgs) => {
        if (Array.isArray(msgs) && msgs.length) {
          setV2x((prev) => {
            const keyOf = (m) => `${m.ts}|${m.t}|${m.src}|${JSON.stringify(m.p)}`;
            const seen = new Set(prev.map(keyOf));
            const fresh = msgs.filter((m) => !seen.has(keyOf(m)));
            return [...fresh.reverse(), ...prev].slice(0, 60);
          });
        }
      }).catch(() => {});
    }, 1500);
    return () => clearInterval(t);
  }, [running]);

  const onStart = async () => {
    await api.simStart(simCfg);
    setRunning(true);
  };
  const onStop = async () => {
    await api.simStop();
    setRunning(false);
  };
  const onReset = async () => {
    await api.simReset(simCfg);
  };
  const onPolicy = async (p) => {
    const r = await api.setPolicy(p);
    setPolicy(r.policy);
  };

  const onTrain = async () => {
    try {
      await api.mlTrain(Number(episodes), 7);
      setTrainStatus({ state: "running", events: [] });
    } catch (e) {
      alert(e?.response?.data?.detail || "Training failed to start");
    }
  };

  const onImportOSM = async () => {
    setOsmLoading(true);
    try {
      const r = await api.osmImport(osmPlace, Number(osmRadius));
      setOsmResult(r);
    } catch (e) {
      setOsmResult({ error: e?.response?.data?.detail || e.message || "OSM import failed" });
    } finally {
      setOsmLoading(false);
    }
  };

  const liveMetrics = snapshot?.metrics || {};
  const latestTrain = trainStatus.events?.slice(-1)[0];

  return (
    <>
      {/* NAV */}
      <nav className="tn-nav">
        <div className="tn-brand">
          <div className="tn-brand-logo">TN</div>
          <span>TRAFFIC NEXUS</span>
        </div>
        <div className="tn-nav-links">
          <a href="#simulation" data-testid="nav-sim">Simulation</a>
          <a href="#ml" data-testid="nav-ml">ML Training</a>
          <a href="#three" data-testid="nav-3d">3D Bridge</a>
          <a href="#osm" data-testid="nav-osm">OSM</a>
          <a href="#arch" data-testid="nav-arch">Architecture</a>
        </div>
        <div className="tn-status">
          <span className={`dot ${running ? "on" : ""}`} /> {running ? "RUNNING" : "IDLE"}
          <span className="tag" data-testid="policy-tag">POLICY: {policy.toUpperCase()}</span>
        </div>
      </nav>

      {/* HERO */}
      <section className="tn-hero tn-container">
        <span className="tn-badge" data-testid="hero-badge">
          <span className="dot on" /> MARL DQN · Live V2X · Real 3D
        </span>
        <h1 className="tn-title">
          Adaptive Traffic Control<br />
          <span className="line2">Powered by Real MARL</span>
        </h1>
        <p className="tn-sub">
          A full-stack metropolitan traffic platform: PyTorch shared-DQN, per-vehicle kinematics, real OSM ingestion
          with multi-mirror fallback, live V2X messaging, and Three.js + Unity-compatible 3D visualization — all wired
          through a real FastAPI + WebSocket backbone.
        </p>
        <div className="tn-cta">
          <button className="btn btn-primary" onClick={onStart} data-testid="hero-start-btn">Start Simulation</button>
          <a href="#ml" className="btn btn-ghost" data-testid="hero-train-link">Train ML Model →</a>
          <a href="#arch" className="btn btn-ghost" data-testid="hero-arch-link">Architecture</a>
        </div>

        <div className="tn-kpi">
          <div className="kpi"><div className="kpi-val" data-testid="kpi-vehicles">{snapshot?.vehicles?.length ?? 0}</div><div className="kpi-lbl">Live Vehicles</div></div>
          <div className="kpi"><div className="kpi-val" data-testid="kpi-signals">{snapshot?.tls?.length ?? 0}</div><div className="kpi-lbl">Traffic Signals</div></div>
          <div className="kpi"><div className="kpi-val" data-testid="kpi-step">{snapshot?.step ?? 0}</div><div className="kpi-lbl">Sim Step</div></div>
          <div className="kpi"><div className="kpi-val" data-testid="kpi-throughput">{liveMetrics.throughput ?? 0}</div><div className="kpi-lbl">Throughput / Step</div></div>
          <div className="kpi"><div className="kpi-val" data-testid="kpi-models">{health?.models_loaded ? "YES" : "NO"}</div><div className="kpi-lbl">DQN Loaded</div></div>
          <div className="kpi"><div className="kpi-val" data-testid="kpi-best">{summary?.best_avg_reward ? fmt(summary.best_avg_reward, 1) : "—"}</div><div className="kpi-lbl">Best Avg Reward</div></div>
        </div>
      </section>

      {/* SIMULATION */}
      <section id="simulation" className="tn-section tn-container">
        <div className="sec-head">
          <h2 className="sec-title">Live Simulation</h2>
          <p className="sec-sub">Per-vehicle kinematics over a multi-intersection grid. Pick a policy — the DQN reacts in real time.</p>
        </div>
        <div className="grid-3">
          {/* Controls */}
          <div className="card">
            <div className="card-head"><h3 className="card-title">Controls</h3></div>
            <label className="field-lbl">Rows × Cols</label>
            <div className="ctrl-row">
              <input className="input" type="number" min="2" max="6" value={simCfg.rows} onChange={(e) => setSimCfg({ ...simCfg, rows: Number(e.target.value) })} data-testid="ctrl-rows" />
              <input className="input" type="number" min="2" max="6" value={simCfg.cols} onChange={(e) => setSimCfg({ ...simCfg, cols: Number(e.target.value) })} data-testid="ctrl-cols" />
            </div>
            <label className="field-lbl">Max Vehicles</label>
            <input className="input" type="number" min="20" max="800" value={simCfg.max_vehicles} onChange={(e) => setSimCfg({ ...simCfg, max_vehicles: Number(e.target.value) })} data-testid="ctrl-max" />

            <div style={{ height: 14 }} />
            <div className="ctrl-row">
              {!running ? (
                <button className="btn btn-primary" onClick={onStart} data-testid="sim-start-btn">Start</button>
              ) : (
                <button className="btn btn-danger" onClick={onStop} data-testid="sim-stop-btn">Stop</button>
              )}
              <button className="btn btn-ghost" onClick={onReset} data-testid="sim-reset-btn">Reset</button>
            </div>

            <div style={{ height: 20 }} />
            <label className="field-lbl">Signal Policy</label>
            <div className="ctrl-row">
              {["fixed", "pressure", "learned"].map((p) => (
                <button
                  key={p}
                  className={`btn ${policy === p ? "btn-primary" : "btn-ghost"}`}
                  onClick={() => onPolicy(p)}
                  data-testid={`policy-${p}`}
                  style={{ padding: "10px 8px", fontSize: "0.72rem" }}
                >
                  {p}
                </button>
              ))}
            </div>
            {policy === "learned" && !health?.models_loaded && (
              <p className="muted" style={{ fontSize: "0.7rem", marginTop: 10 }}>
                ⚠ Learned policy selected but no DQN artifact is loaded — train first.
              </p>
            )}

            <div style={{ height: 20 }} />
            <div className="metric-row"><span>Vehicles</span><span data-testid="metric-vehicles">{liveMetrics.vehicles ?? 0}</span></div>
            <div className="metric-row"><span>Stopped</span><span data-testid="metric-stopped">{liveMetrics.stopped ?? 0}</span></div>
            <div className="metric-row"><span>Throughput</span><span>{liveMetrics.throughput ?? 0}</span></div>
            <div className="metric-row"><span>Emissions</span><span>{fmt(liveMetrics.emissions, 1)}</span></div>
          </div>

          {/* Canvas */}
          <div className="card">
            <div className="card-head">
              <h3 className="card-title">2D Live View</h3>
              <span className="tag on">LIVE · WebSocket</span>
            </div>
            <SimCanvas snapshot={snapshot} />
          </div>

          {/* V2X log */}
          <div className="card">
            <div className="card-head">
              <h3 className="card-title">V2X Messages (real events)</h3>
            </div>
            <div className="v2x-log" data-testid="v2x-log">
              {v2x.length === 0 && <div className="muted" style={{ fontSize: "0.75rem" }}>Start the simulation to see real V2X traffic.</div>}
              {v2x.map((m, i) => (
                <div className="v2x-row" key={`${m.ts}-${m.src}-${i}`}>
                  <span className={`v2x-type v2x-${m.t}`}>{m.t}</span>
                  <span className="v2x-src">{m.src}</span>
                  <span className="v2x-payload">{JSON.stringify(m.p)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ML Section */}
      <section id="ml" className="tn-section tn-container">
        <div className="sec-head">
          <h2 className="sec-title">Real ML Training</h2>
          <p className="sec-sub">Shared-DQN over the multi-intersection env. PyTorch. Real replay buffer, target network, epsilon-greedy.</p>
        </div>
        <div className="grid-2">
          <div className="card">
            <div className="card-head">
              <h3 className="card-title">Launch Training</h3>
              <span className={`train-status ${trainStatus.state}`} data-testid="train-status">{trainStatus.state}</span>
            </div>
            <label className="field-lbl">Episodes</label>
            <input className="input" type="number" min="5" max="200" value={episodes} onChange={(e) => setEpisodes(e.target.value)} data-testid="train-episodes" />
            <div style={{ height: 10 }} />
            <button className="btn btn-primary" onClick={onTrain} disabled={trainStatus.state === "running"} data-testid="train-btn">
              {trainStatus.state === "running" ? "Training…" : "Train Shared DQN"}
            </button>

            <div style={{ height: 18 }} />
            <label className="field-lbl">Progress</label>
            <div className="event-log" data-testid="train-log">
              {trainStatus.events?.length === 0 && <div className="muted">No events yet.</div>}
              {trainStatus.events?.slice(-18).map((ev, i) => (
                <div className="event-row" key={i}>
                  {ev.kind === "forecast"
                    ? `· forecast epoch ${ev.epoch} loss=${fmt(ev.loss, 4)}`
                    : `· episode ${ev.episode}/${ev.total} · reward=${fmt(ev.avg_reward, 2)} · q=${fmt(ev.avg_queue, 1)} · f=${fmt(ev.avg_fairness_penalty, 2)} · ε=${fmt(ev.epsilon, 2)}`}
                </div>
              ))}
            </div>

            {trainStatus.state === "done" && trainStatus.result && (
              <>
                <div style={{ height: 14 }} />
                <label className="field-lbl">Policy Comparison (learned vs baselines)</label>
                <div className="metric-row"><span>Learned reward</span><span data-testid="eval-learned-reward">{fmt(trainStatus.result.evaluation.learned_reward, 2)}</span></div>
                <div className="metric-row"><span>Pressure reward</span><span>{fmt(trainStatus.result.evaluation.pressure_reward, 2)}</span></div>
                <div className="metric-row"><span>Fixed reward</span><span>{fmt(trainStatus.result.evaluation.fixed_reward, 2)}</span></div>
                <div className="metric-row"><span>Learned queue</span><span>{fmt(trainStatus.result.evaluation.learned_queue, 2)}</span></div>
                <div className="metric-row"><span>Learned fairness</span><span>{fmt(trainStatus.result.evaluation.learned_fairness, 2)}</span></div>
              </>
            )}
          </div>

          <div className="card">
            <div className="card-head">
              <h3 className="card-title">Training History</h3>
              <span className="tag">{metrics.length} episodes</span>
            </div>
            <MetricsChart metrics={metrics} />
            {summary && (
              <>
                <div style={{ height: 10 }} />
                <div className="metric-row"><span>Episodes</span><span>{summary.episodes}</span></div>
                <div className="metric-row"><span>Best avg reward</span><span data-testid="summary-best">{fmt(summary.best_avg_reward, 2)}</span></div>
                <div className="metric-row"><span>Final avg reward</span><span>{fmt(summary.final_avg_reward, 2)}</span></div>
                <div className="metric-row"><span>Final avg queue</span><span>{fmt(summary.final_avg_queue, 2)}</span></div>
                <div className="metric-row"><span>Final fairness penalty</span><span>{fmt(summary.final_avg_fairness_penalty, 2)}</span></div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* 3D Bridge */}
      <section id="three" className="tn-section tn-container">
        <div className="sec-head">
          <h2 className="sec-title">Unity-Compatible 3D Bridge</h2>
          <p className="sec-sub">
            Real WebGL 3D via Three.js, driven by the same WebSocket protocol a Unity client would consume.
            Orbit, zoom, and pan with the mouse.
          </p>
        </div>
        <div className="card">
          <div className="card-head">
            <h3 className="card-title">Real 3D Viewer</h3>
            <span className="tag on">WEBSOCKET · SAME PROTOCOL AS UNITY</span>
          </div>
          <ThreeViewer snapshot={snapshot} />
          <p className="muted" style={{ fontSize: "0.7rem", marginTop: 12 }}>
            A Unity C# client sample is in <code>/app/unity_client/</code> — connect it to{" "}
            <code>{WS_URL}</code> and it will render the same world.
          </p>
        </div>
      </section>

      {/* OSM */}
      <section id="osm" className="tn-section tn-container">
        <div className="sec-head">
          <h2 className="sec-title">Real OSM Ingestion</h2>
          <p className="sec-sub">5 Overpass mirrors + exponential backoff + MongoDB cache + offline fallback for when public servers are rate-limited.</p>
        </div>
        <div className="grid-2">
          <div className="card">
            <div className="card-head"><h3 className="card-title">Import</h3></div>
            <label className="field-lbl">Place</label>
            <input className="input" value={osmPlace} onChange={(e) => setOsmPlace(e.target.value)} data-testid="osm-place" />
            <div style={{ height: 10 }} />
            <label className="field-lbl">Radius (m)</label>
            <input className="input" type="number" value={osmRadius} onChange={(e) => setOsmRadius(e.target.value)} data-testid="osm-radius" />
            <div style={{ height: 12 }} />
            <button className="btn btn-primary" onClick={onImportOSM} disabled={osmLoading} data-testid="osm-import-btn">
              {osmLoading ? "Fetching…" : "Import with Fallback"}
            </button>
            <p className="muted" style={{ fontSize: "0.7rem", marginTop: 12 }}>
              Tries 5 Overpass mirrors. On total failure, serves offline Chennai / Bengaluru snapshots so your demo never breaks.
            </p>
          </div>
          <div className="card">
            <div className="card-head">
              <h3 className="card-title">Result</h3>
              {osmResult && <span className={`tag ${osmResult.source === "live" ? "on" : ""}`}>{osmResult.source || osmResult.cache || "—"}</span>}
            </div>
            {!osmResult && <div className="muted">Run an import to see parsed nodes, edges, and signal counts.</div>}
            {osmResult?.error && <div style={{ color: "var(--accent)", fontSize: "0.8rem" }}>{osmResult.error}</div>}
            {osmResult?.graph && (
              <>
                <div className="metric-row"><span>Place</span><span>{osmResult.location?.display_name || osmResult.place}</span></div>
                <div className="metric-row"><span>Nodes</span><span data-testid="osm-nodes">{osmResult.graph.nodes?.length}</span></div>
                <div className="metric-row"><span>Edges</span><span data-testid="osm-edges">{osmResult.graph.edges?.length}</span></div>
                <div className="metric-row"><span>Traffic signals</span><span data-testid="osm-signals">{osmResult.graph.signals ?? osmResult.graph.nodes?.filter((n) => n.is_signal).length}</span></div>
                <div className="metric-row"><span>Source</span><span>{osmResult.source}</span></div>
                {osmResult.overpass_endpoint && <div className="metric-row"><span>Overpass</span><span style={{ fontSize: "0.65rem" }}>{osmResult.overpass_endpoint}</span></div>}
                {osmResult.live_error && <div className="muted" style={{ fontSize: "0.7rem", marginTop: 8, color: "var(--warn)" }}>live failed → {osmResult.live_error}</div>}
              </>
            )}
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section id="arch" className="tn-section tn-container">
        <div className="sec-head">
          <h2 className="sec-title">Architecture</h2>
          <p className="sec-sub">Every layer here corresponds to real, runnable code in the repository — nothing is mocked.</p>
        </div>
        <div className="arch-grid">
          {ARCH_LAYERS.map((l) => (
            <div className="arch-card" key={l.num}>
              <div className="arch-num">{l.num}</div>
              <div className="arch-name">{l.name}</div>
              <div className="arch-desc">{l.desc}</div>
              <span className={`arch-real ${l.real}`}>{l.real === "real" ? "✓ REAL" : "~ PARTIAL"}</span>
            </div>
          ))}
        </div>
      </section>

      <footer className="tn-footer">
        TRAFFIC NEXUS v2.0 · Real MARL / V2X / OSM / 3D · Built with FastAPI + PyTorch + React + Three.js · <a href="/api/health">/api/health</a>
      </footer>
    </>
  );
}

function MetricsChart({ metrics }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const c = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);
    c.fillStyle = "#060813"; c.fillRect(0, 0, W, H);
    if (!metrics || metrics.length === 0) {
      c.fillStyle = "#8ea0bd"; c.font = "12px JetBrains Mono";
      c.fillText("No training metrics yet. Train to populate.", 20, H / 2);
      return;
    }
    const rewards = metrics.map((m) => m.avg_reward);
    const queues = metrics.map((m) => m.avg_queue);
    const rMin = Math.min(...rewards), rMax = Math.max(...rewards);
    const qMax = Math.max(...queues);
    const pad = 32;
    const plot = (vals, max, min, color, label, yOffset) => {
      c.strokeStyle = color; c.lineWidth = 2; c.beginPath();
      vals.forEach((v, i) => {
        const x = pad + (i / Math.max(1, vals.length - 1)) * (W - pad * 2);
        const y = H - pad - ((v - min) / Math.max(0.0001, max - min)) * (H - pad * 2);
        if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      });
      c.stroke();
      c.fillStyle = color; c.font = "10px JetBrains Mono"; c.fillText(label, pad + 4, 18 + yOffset);
    };
    plot(rewards, rMax, rMin, "#5fe3a1", "avg_reward", 0);
    plot(queues, qMax, 0, "#00f0ff", "avg_queue", 14);
    c.strokeStyle = "rgba(255,255,255,0.08)";
    c.strokeRect(pad, pad, W - pad * 2, H - pad * 2);
  }, [metrics]);
  return <canvas ref={ref} width={640} height={240} style={{ width: "100%", height: 240 }} data-testid="metrics-chart" />;
}
