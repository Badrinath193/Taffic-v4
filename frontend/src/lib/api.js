import axios from "axios";

const BACKEND = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND}/api`;

export const WS_URL = `${BACKEND.replace(/^http/, "ws")}/ws/stream`;

const http = axios.create({ baseURL: API, timeout: 60000 });

export const api = {
  health: () => http.get("/health").then((r) => r.data),
  syntheticNet: (rows = 3, cols = 3) =>
    http.get("/network/synthetic", { params: { rows, cols } }).then((r) => r.data),

  simStart: (cfg) => http.post("/sim/start", cfg).then((r) => r.data),
  simStop: () => http.post("/sim/stop").then((r) => r.data),
  simReset: (cfg) => http.post("/sim/reset", cfg).then((r) => r.data),
  simState: () => http.get("/sim/state").then((r) => r.data),
  simMetrics: (limit = 200) => http.get("/sim/metrics", { params: { limit } }).then((r) => r.data),
  setPolicy: (policy) => http.post("/sim/set_policy", { policy }).then((r) => r.data),

  v2xTail: (n = 50) => http.get("/v2x/tail", { params: { n } }).then((r) => r.data),

  mlTrain: (episodes = 25, seed = 7) => http.post("/ml/train", { episodes, seed }).then((r) => r.data),
  mlStatus: () => http.get("/ml/train_status").then((r) => r.data),
  mlMetrics: () => http.get("/ml/metrics").then((r) => r.data),
  mlSummary: () => http.get("/ml/summary").then((r) => r.data),
  mlEvaluate: () => http.post("/ml/evaluate").then((r) => r.data),

  osmImport: (place, radius = 1500) =>
    http.post("/osm/import", { place, radius }, { timeout: 120000 }).then((r) => r.data),
  osmCached: () => http.get("/osm/cached").then((r) => r.data),
};
