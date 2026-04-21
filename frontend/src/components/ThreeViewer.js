import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const TYPE_COLORS = {
  car: 0x5fe3a1,
  truck: 0xffb84d,
  bus: 0x00f0ff,
  motorcycle: 0xff4d9d,
  emergency: 0xff3b3b,
};
const PHASE_COLORS = [0x3fff80, 0xffd23d, 0x3fff80, 0xffd23d];

export default function ThreeViewer({ snapshot }) {
  const mountRef = useRef(null);
  const stateRef = useRef({});
  const snapRef = useRef(snapshot);

  useEffect(() => {
    snapRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const width = mount.clientWidth;
    const height = mount.clientHeight || 520;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x03040a);
    scene.fog = new THREE.Fog(0x03040a, 120, 360);

    const camera = new THREE.PerspectiveCamera(48, width / height, 0.5, 800);
    camera.position.set(55, 70, 85);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.62));
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(30, 40, 20);
    scene.add(dir);
    const pt = new THREE.PointLight(0x00f0ff, 0.7, 500);
    pt.position.set(0, 30, 0);
    scene.add(pt);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(500, 500),
      new THREE.MeshStandardMaterial({ color: 0x070a18 })
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const worldGroup = new THREE.Group();
    scene.add(worldGroup);

    const roadMat = new THREE.MeshStandardMaterial({ color: 0x3b4770, roughness: 0.8 });
    const roadHaloMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.15 });
    const tlPoleMat = new THREE.MeshStandardMaterial({ color: 0x2a2f40 });

    let roadMeshes = [];
    let tlGroups = new Map();
    let vehicleMeshes = new Map();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    stateRef.current = { scene, camera, renderer, worldGroup, roadMat, roadHaloMat, tlPoleMat, controls };

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight || 520;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    let rafId;
    let lastLayoutKey = "";

    function buildStatic(snap) {
      // dispose previous
      roadMeshes.forEach((m) => {
        worldGroup.remove(m);
        m.geometry?.dispose();
      });
      roadMeshes = [];
      tlGroups.forEach((g) => worldGroup.remove(g.group));
      tlGroups.clear();

      const nodes = snap.nodes || [];
      const edges = snap.edges || [];
      if (!nodes.length) return;
      const xs = nodes.map((n) => n.x);
      const ys = nodes.map((n) => n.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
      const spanW = Math.max(1, maxX - minX);
      const spanH = Math.max(1, maxY - minY);

      // Fit the world into a ~110-unit canvas irrespective of whether input is
      // a synthetic grid (range ~400) or a large OSM city (range ~2000).
      const scale = Math.min(110 / spanW, 110 / spanH);
      const byId = {};
      nodes.forEach((n) => { byId[n.id] = { x: (n.x - cx) * scale, z: (n.y - cy) * scale }; });
      stateRef.current.byId = byId;
      stateRef.current.center = { cx, cy, scale };

      // Road width scales inversely with the world scale so it stays visually readable.
      const roadWidth = Math.max(2.4, Math.min(6.0, 260 * scale));

      edges.forEach((e) => {
        const a = byId[e.from];
        const b = byId[e.to];
        if (!a || !b) return;
        const dx = b.x - a.x, dz = b.z - a.z;
        const len = Math.hypot(dx, dz);
        if (len < 0.3) return;
        const ang = Math.atan2(dz, dx);

        // glow halo
        const halo = new THREE.Mesh(new THREE.PlaneGeometry(len, roadWidth + 1.6), roadHaloMat);
        halo.rotation.x = -Math.PI / 2;
        halo.rotation.z = -ang;
        halo.position.set((a.x + b.x) / 2, 0.015, (a.z + b.z) / 2);
        worldGroup.add(halo);
        roadMeshes.push(halo);

        // asphalt surface
        const road = new THREE.Mesh(new THREE.PlaneGeometry(len, roadWidth), roadMat);
        road.rotation.x = -Math.PI / 2;
        road.rotation.z = -ang;
        road.position.set((a.x + b.x) / 2, 0.03, (a.z + b.z) / 2);
        worldGroup.add(road);
        roadMeshes.push(road);
      });

      const tls = snap.tls || [];
      tls.forEach((t) => {
        const p = byId[t.nid];
        if (!p) return;
        const g = new THREE.Group();
        g.position.set(p.x, 0, p.z);
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 5, 8), tlPoleMat);
        pole.position.y = 2.5;
        g.add(pole);
        const bulbMat = new THREE.MeshStandardMaterial({
          color: PHASE_COLORS[t.phase] || 0x666666,
          emissive: PHASE_COLORS[t.phase] || 0x666666,
          emissiveIntensity: 1.2,
        });
        const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.7, 14, 14), bulbMat);
        bulb.position.y = 5.2;
        g.add(bulb);
        worldGroup.add(g);
        tlGroups.set(t.nid, { group: g, bulbMat });
      });

      // auto-fit camera to bounds
      const worldHalf = Math.max(spanW, spanH) * scale * 0.5;
      const dist = Math.max(40, worldHalf * 1.85);
      camera.position.set(dist * 0.55, dist * 0.95, dist * 0.85);
      camera.far = Math.max(500, dist * 6);
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.minDistance = Math.max(14, dist * 0.18);
      controls.maxDistance = dist * 4;
      controls.update();

      stateRef.current.roadMeshes = roadMeshes;
      stateRef.current.tlGroups = tlGroups;
    }

    function updateDynamic(snap) {
      const ctr = stateRef.current.center;
      if (!ctr) return;
      const { cx, cy, scale } = ctr;

      const tls = snap.tls || [];
      tls.forEach((t) => {
        const g = tlGroups.get(t.nid);
        if (!g) return;
        const col = PHASE_COLORS[t.phase] || 0x666666;
        g.bulbMat.color.setHex(col);
        g.bulbMat.emissive.setHex(col);
      });

      const vScale = Math.max(0.7, Math.min(1.8, 50 * scale));
      const vehicles = snap.vehicles || [];
      const seen = new Set();
      vehicles.forEach((v) => {
        seen.add(v.id);
        let mesh = vehicleMeshes.get(v.id);
        if (!mesh) {
          const col = TYPE_COLORS[v.t] || 0xcccccc;
          const base = v.t === "truck" || v.t === "bus" ? [3, 1.1, 1.1] : v.t === "motorcycle" ? [1.3, 0.5, 0.5] : [2, 0.9, 0.9];
          const sz = base.map((b) => b * vScale);
          const geom = new THREE.BoxGeometry(sz[0], sz[1], sz[2]);
          const mat = new THREE.MeshStandardMaterial({
            color: col, emissive: col,
            emissiveIntensity: v.t === "emergency" ? 0.75 : 0.32,
          });
          mesh = new THREE.Mesh(geom, mat);
          worldGroup.add(mesh);
          vehicleMeshes.set(v.id, mesh);
        }
        const sx0 = (v.fx - cx) * scale;
        const sz0 = (v.fy - cy) * scale;
        const sx1 = (v.tx - cx) * scale;
        const sz1 = (v.ty - cy) * scale;
        const x = sx0 + (sx1 - sx0) * v.p;
        const z = sz0 + (sz1 - sz0) * v.p;
        const rot = -Math.atan2(sz1 - sz0, sx1 - sx0);
        mesh.position.set(x, 0.55 * vScale, z);
        mesh.rotation.y = rot;
      });
      vehicleMeshes.forEach((mesh, id) => {
        if (!seen.has(id)) {
          worldGroup.remove(mesh);
          mesh.geometry?.dispose();
          mesh.material?.dispose();
          vehicleMeshes.delete(id);
        }
      });
    }

    function tick() {
      const snap = snapRef.current;
      if (snap && snap.nodes && snap.nodes.length) {
        const key = snap.nodes.length + ":" + (snap.nodes[0]?.id || "") + "|" + (snap.edges?.length ?? 0);
        if (key !== lastLayoutKey) {
          lastLayoutKey = key;
          buildStatic(snap);
        }
        updateDynamic(snap);
      }
      worldGroup.rotation.y += 0.0004;
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(tick);
    }
    tick();

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
      scene.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose?.();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose?.());
          else obj.material.dispose?.();
        }
      });
    };
  }, []);

  return (
    <div className="three-wrap">
      <div className="three-info">
        REAL 3D · THREE.JS r172 · {snapshot?.vehicles?.length || 0} VEH
      </div>
      <div ref={mountRef} style={{ width: "100%", height: 520 }} data-testid="three-canvas" />
      {!snapshot?.nodes?.length && (
        <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", pointerEvents: "none" }}>
          <span style={{ color: "#8ea0bd", fontSize: "0.8rem", letterSpacing: "0.2em" }}>
            START SIMULATION TO SEE 3D VIEW
          </span>
        </div>
      )}
    </div>
  );
}
