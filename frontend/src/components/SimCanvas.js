import React, { useEffect, useRef } from "react";

const TYPE_COLORS = {
  car: "#5fe3a1",
  truck: "#ffb84d",
  bus: "#00f0ff",
  motorcycle: "#ff4d9d",
  emergency: "#ff3b3b",
};

const PHASE_COLORS = ["#3fff80", "#ffd23d", "#3fff80", "#ffd23d"]; // 0=NS green, 1=NS amber, 2=EW green, 3=EW amber
const PHASE_LABEL = ["NS GREEN", "NS AMBER", "EW GREEN", "EW AMBER"];

export default function SimCanvas({ snapshot }) {
  const ref = useRef(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !snapshot) return;
    const c = cv.getContext("2d");
    const W = cv.width;
    const H = cv.height;
    c.clearRect(0, 0, W, H);

    // background grid
    c.fillStyle = "#060813";
    c.fillRect(0, 0, W, H);
    c.strokeStyle = "rgba(120,180,255,0.05)";
    c.lineWidth = 1;
    for (let x = 0; x < W; x += 40) {
      c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
    }
    for (let y = 0; y < H; y += 40) {
      c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke();
    }

    const nodes = snapshot.nodes || [];
    if (!nodes.length) return;
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 60;
    const sx = (W - pad * 2) / Math.max(1, maxX - minX);
    const sy = (H - pad * 2) / Math.max(1, maxY - minY);
    const s = Math.min(sx, sy);
    const toX = (x) => pad + (x - minX) * s;
    const toY = (y) => pad + (y - minY) * s;

    const byId = {};
    nodes.forEach((n) => (byId[n.id] = n));

    // roads - fat, bright, clearly visible at any zoom
    const edges = snapshot.edges || [];
    const roadWidth = Math.max(6, Math.min(14, 10 * s));
    // outer glow
    c.strokeStyle = "rgba(0, 240, 255, 0.08)";
    c.lineWidth = roadWidth + 8;
    c.lineCap = "round";
    edges.forEach((e) => {
      const a = byId[e.from], b = byId[e.to];
      if (!a || !b) return;
      c.beginPath();
      c.moveTo(toX(a.x), toY(a.y));
      c.lineTo(toX(b.x), toY(b.y));
      c.stroke();
    });
    // main asphalt
    c.strokeStyle = "rgba(176, 195, 230, 0.78)";
    c.lineWidth = roadWidth;
    edges.forEach((e) => {
      const a = byId[e.from], b = byId[e.to];
      if (!a || !b) return;
      c.beginPath();
      c.moveTo(toX(a.x), toY(a.y));
      c.lineTo(toX(b.x), toY(b.y));
      c.stroke();
    });
    // lane dashes - darker for contrast on the brighter roads
    c.strokeStyle = "rgba(15, 20, 36, 0.65)";
    c.lineWidth = Math.max(1, roadWidth * 0.14);
    c.setLineDash([8, 8]);
    edges.forEach((e) => {
      const a = byId[e.from], b = byId[e.to];
      if (!a || !b) return;
      c.beginPath();
      c.moveTo(toX(a.x), toY(a.y));
      c.lineTo(toX(b.x), toY(b.y));
      c.stroke();
    });
    c.setLineDash([]);

    // traffic lights
    const tls = snapshot.tls || [];
    const tlByNode = {};
    tls.forEach((t) => (tlByNode[t.nid] = t));
    nodes.forEach((n) => {
      const tl = tlByNode[n.id];
      const px = toX(n.x), py = toY(n.y);
      if (tl !== undefined) {
        const col = PHASE_COLORS[tl.phase] || "#666";
        c.fillStyle = col;
        c.shadowColor = col;
        c.shadowBlur = 14;
        c.beginPath();
        c.arc(px, py, 7, 0, Math.PI * 2);
        c.fill();
        c.shadowBlur = 0;
      } else {
        c.fillStyle = "#334";
        c.beginPath(); c.arc(px, py, 4, 0, Math.PI * 2); c.fill();
      }
    });

    // vehicles
    const vehicles = snapshot.vehicles || [];
    vehicles.forEach((v) => {
      const fx = toX(v.fx), fy = toY(v.fy);
      const tx = toX(v.tx), ty = toY(v.ty);
      const x = fx + (tx - fx) * v.p;
      const y = fy + (ty - fy) * v.p;
      const ang = Math.atan2(ty - fy, tx - fx);
      const col = TYPE_COLORS[v.t] || "#ccc";
      c.save();
      c.translate(x, y);
      c.rotate(ang);
      const len = v.t === "truck" || v.t === "bus" ? 13 : v.t === "motorcycle" ? 6 : 9;
      c.fillStyle = col;
      if (v.t === "emergency") {
        const blink = Math.floor(Date.now() / 140) % 2 === 0;
        c.fillStyle = blink ? "#ff3b3b" : "#ffffff";
        c.shadowColor = "#ff3b3b";
        c.shadowBlur = 10;
      }
      c.fillRect(-len / 2, -3, len, 6);
      c.shadowBlur = 0;
      c.restore();
    });

    // legend
    c.fillStyle = "rgba(0,0,0,0.5)";
    c.fillRect(W - 150, 10, 140, 92);
    c.strokeStyle = "rgba(255,255,255,0.12)";
    c.strokeRect(W - 150, 10, 140, 92);
    c.font = "10px JetBrains Mono";
    c.fillStyle = "#8ea0bd";
    c.fillText(`STEP: ${snapshot.step || 0}`, W - 140, 26);
    c.fillText(`VEHICLES: ${vehicles.length}`, W - 140, 40);
    c.fillText(`SIGNALS: ${tls.length}`, W - 140, 54);
    const phases = tls.length ? PHASE_LABEL[tls[0].phase] : "-";
    c.fillText(`TL[0]: ${phases}`, W - 140, 68);
    c.fillText(`POLICY: ${(snapshot._policy || "fixed").toUpperCase()}`, W - 140, 82);
  }, [snapshot]);

  return (
    <div className="sim-wrap">
      <div className="sim-overlay">
        {snapshot ? `LIVE · ${snapshot.vehicles?.length || 0} vehicles · step ${snapshot.step || 0}` : "idle"}
      </div>
      <canvas
        ref={ref}
        width={960}
        height={520}
        style={{ width: "100%", height: "100%", display: "block" }}
        data-testid="sim-canvas"
      />
    </div>
  );
}
