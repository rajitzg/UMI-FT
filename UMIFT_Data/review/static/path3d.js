// three.js tool path. Geometry is built once per episode; the start/end range uses
// setDrawRange (no rebuild). Renders on demand only (dirty flag).
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const STOPS = [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]];
function viridis(u) {
  u = Math.min(1, Math.max(0, u)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(u)), f = u - i, a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function lowerBound(arr, v) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

function interp(tSrc, ySrc, tDst) {
  const out = new Float32Array(tDst.length);
  let j = 0;
  for (let i = 0; i < tDst.length; i++) {
    const t = tDst[i];
    while (j < tSrc.length - 2 && tSrc[j + 1] < t) j++;
    const t0 = tSrc[j], t1 = tSrc[j + 1] ?? t0;
    const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    out[i] = ySrc[j] + ((ySrc[j + 1] ?? ySrc[j]) - ySrc[j]) * f;
  }
  return out;
}

export function createPath3d(container, arrays, header) {
  container.innerHTML = "";
  const n = arrays.pose_t.length;
  const tArr = arrays.pose_t;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0xffffff);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.001, 100);
  camera.up.set(0, -1, 0); // umift_std_v2: +Y down
  const controls = new OrbitControls(camera, renderer.domElement);

  const pos = new Float32Array(n * 3);
  const c = new THREE.Vector3();
  for (let i = 0; i < n; i++) { pos[3 * i] = arrays.px[i]; pos[3 * i + 1] = arrays.py[i]; pos[3 * i + 2] = arrays.pz[i]; }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.computeBoundingSphere();
  const bs = geom.boundingSphere;
  c.copy(bs.center);
  const colors = new Float32Array(n * 3);
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const line = new THREE.Line(geom, new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(line);

  const histGeom = new THREE.BufferGeometry(); // shares the position buffer, own draw range
  histGeom.setAttribute("position", geom.attributes.position);
  histGeom.setDrawRange(0, 0);
  const hist = new THREE.Points(histGeom, new THREE.PointsMaterial({ color: 0x1f77b4, size: 4, sizeAttenuation: false }));
  scene.add(hist);

  const r = Math.max(bs.radius, 0.01);
  const marker = new THREE.Mesh(new THREE.SphereGeometry(r * 0.025, 12, 8), new THREE.MeshBasicMaterial({ color: 0x000000 }));
  scene.add(marker);
  const triad = new THREE.AxesHelper(r * 0.15);
  scene.add(triad);
  const grid = new THREE.AxesHelper(r * 0.3);
  grid.position.copy(c);
  grid.material.transparent = true; grid.material.opacity = 0.25;
  scene.add(grid);

  controls.target.copy(c);
  camera.position.set(c.x, c.y - r * 0.8, c.z + r * 2.6); // operator side (+Z), slightly above
  controls.update();

  let fmagAtPose = null;
  let dirty = true;
  function setColorBy(mode) {
    let vals, lo, hi;
    if (mode === "force") {
      fmagAtPose ??= interp(arrays.wrench_t, arrays.fmag, tArr);
      vals = fmagAtPose;
      const sorted = Float32Array.from(vals).sort();
      lo = sorted[Math.floor(0.02 * (n - 1))]; hi = sorted[Math.floor(0.98 * (n - 1))];
    } else { vals = tArr; lo = tArr[0]; hi = tArr[n - 1]; }
    const span = hi > lo ? hi - lo : 1;
    for (let i = 0; i < n; i++) { const [rr, g, b] = viridis((vals[i] - lo) / span); colors[3 * i] = rr; colors[3 * i + 1] = g; colors[3 * i + 2] = b; }
    geom.attributes.color.needsUpdate = true;
    dirty = true;
  }
  setColorBy("time");

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();
  controls.addEventListener("change", () => { dirty = true; });

  let alive = true;
  (function loop() {
    if (!alive) return;
    if (dirty) { renderer.render(scene, camera); dirty = false; }
    requestAnimationFrame(loop);
  })();

  const q = new THREE.Quaternion();
  return {
    setRange(a, b) {
      const i0 = lowerBound(tArr, a), i1 = Math.max(i0 + 1, lowerBound(tArr, b));
      geom.setDrawRange(i0, i1 - i0); dirty = true;
    },
    setCursor(t, histFrom) {
      const i = Math.min(n - 1, lowerBound(tArr, t));
      marker.position.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
      triad.position.copy(marker.position);
      q.set(arrays.qx[i], arrays.qy[i], arrays.qz[i], arrays.qw[i]);
      triad.quaternion.copy(q);
      if (histFrom != null) { const j = lowerBound(tArr, histFrom); hist.geometry.setDrawRange(j, Math.max(0, i - j + 1)); }
      else hist.geometry.setDrawRange(0, 0);
      dirty = true;
    },
    setColorBy,
    destroy() { alive = false; ro.disconnect(); controls.dispose(); renderer.dispose(); geom.dispose(); container.innerHTML = ""; },
  };
}
