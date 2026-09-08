// Runs the EXACT SAME geometry-construction code Canvas3D.tsx's useMultiViewFit
// runs client-side (imported directly from src/lib/gnm/*, not reimplemented),
// against the live Next.js (:3000) + ai-engine (:8001) servers, for patient
// 0e9e1d90-2478-4d49-872f-c80772c4bc4f. Prints the REAL THREE.BufferGeometry
// runtime values (position/index/color/uv counts, groups, per-vertex data for
// suspect eye/forehead vertices) — not a Python re-simulation, not a guess.
import * as THREE from "three";

const BASE = "http://localhost:3000";
const realFetch = globalThis.fetch;
globalThis.fetch = ((input: any, init?: any) => {
  if (typeof input === "string" && input.startsWith("/")) {
    return realFetch(BASE + input, init);
  }
  return realFetch(input, init);
}) as typeof fetch;

const { fetchMultiViewAtlas } = await import("../src/lib/gnm/camera-pose");
const { buildGnmGeometryFromPositions } = await import("../src/lib/gnm/geometry");
const { loadGnmHeadRaw, loadGnmRenderMask } = await import("../src/lib/gnm/loader");
const { maskAtlasGeometryByFaceRegion } = await import("../src/lib/gnm/face-oval-mask");
const { extractZipEntries } = await import("../src/lib/gnm/zip");
const { parseNpy } = await import("../src/lib/gnm/npy");

const PATIENT_ID = "0e9e1d90-2478-4d49-872f-c80772c4bc4f";

console.log("Fetching /api/fit-multiview-atlas ...");
const fit = await fetchMultiViewAtlas(PATIENT_ID);
console.log("viewsUsed:", fit.viewsUsed, "warnings:", fit.warnings);
console.log("atlas present:", !!fit.atlas, "regions:", fit.atlas ? Object.keys(fit.atlas.regions).length : 0);

const raw = await loadGnmHeadRaw();
const renderMask = await loadGnmRenderMask();

const positions = Float32Array.from(fit.fittedPositions);
const built = buildGnmGeometryFromPositions(positions, raw);

const colors = new Float32Array(fit.vertexColors.length);
for (let i = 0; i < colors.length; i++) {
  const c = fit.vertexColors[i] / 255;
  colors[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
built.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

let atlasMaterialsOrdered: { region: string }[] = [];
if (fit.atlas) {
  const extraSourceVids = fit.atlas.threejs.extraSourceVids;
  const baseVertexCount = positions.length / 3;
  let finalPositions = positions;
  let finalColors = colors;
  if (extraSourceVids.length > 0) {
    finalPositions = new Float32Array((baseVertexCount + extraSourceVids.length) * 3);
    finalPositions.set(positions);
    finalColors = new Float32Array((baseVertexCount + extraSourceVids.length) * 3);
    finalColors.set(colors);
    extraSourceVids.forEach((srcVid, i) => {
      const dst = baseVertexCount + i;
      finalPositions.set(positions.subarray(srcVid * 3, srcVid * 3 + 3), dst * 3);
      finalColors.set(colors.subarray(srcVid * 3, srcVid * 3 + 3), dst * 3);
    });
    built.geometry.setAttribute("position", new THREE.BufferAttribute(finalPositions, 3));
    built.geometry.setAttribute("color", new THREE.BufferAttribute(finalColors, 3));
  }

  const indices = new Uint16Array(fit.atlas.threejs.index.length);
  indices.set(fit.atlas.threejs.index);
  built.geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  built.geometry.setAttribute("uv", new THREE.BufferAttribute(Float32Array.from(fit.atlas.threejs.uv), 2));
  if (extraSourceVids.length > 0) built.geometry.computeVertexNormals();

  built.geometry.clearGroups();
  for (const group of fit.atlas.threejs.groups) {
    built.geometry.addGroup(group.start, group.count, group.materialIndex);
  }
  atlasMaterialsOrdered = fit.atlas.threejs.groups;
}

// D-halfface masking, exactly as Canvas3D.tsx does for the atlas branch
if (fit.atlas) {
  const currentIndex = built.geometry.index!.array;
  const extraSourceVids = fit.atlas.threejs.extraSourceVids;
  let effectiveMembership = renderMask.membership;
  if (extraSourceVids.length > 0) {
    effectiveMembership = new Float32Array(renderMask.membership.length + extraSourceVids.length);
    effectiveMembership.set(renderMask.membership);
    extraSourceVids.forEach((srcVid, i) => {
      effectiveMembership[renderMask.membership.length + i] = renderMask.membership[srcVid];
    });
  }
  const { index: maskedIndex, groups: maskedGroups } = maskAtlasGeometryByFaceRegion(
    currentIndex as Uint16Array,
    built.geometry.groups as { start: number; count: number; materialIndex: number }[],
    effectiveMembership
  );
  built.geometry.setIndex(new THREE.BufferAttribute(maskedIndex, 1));
  built.geometry.clearGroups();
  for (const g of maskedGroups) built.geometry.addGroup(g.start, g.count, g.materialIndex);
}
built.geometry.computeVertexNormals();

const geo = built.geometry;
console.log("\n=== REAL RUNTIME GEOMETRY (as Canvas3D would render it) ===");
console.log("position.count:", geo.attributes.position.count);
console.log("index.count:", geo.index!.count, "-> triangles:", geo.index!.count / 3);
console.log("color.count:", geo.attributes.color.count);
console.log("uv.count:", geo.attributes.uv ? geo.attributes.uv.count : "(no uv attribute)");
console.log("groups.length:", geo.groups.length);
const byMat: Record<number, { region?: string; tris: number; groups: number }> = {};
for (const g of geo.groups as { start: number; count: number; materialIndex: number }[]) {
  const region = atlasMaterialsOrdered.find((a: any) => a.materialIndex === g.materialIndex)?.region;
  if (!byMat[g.materialIndex]) byMat[g.materialIndex] = { region, tris: 0, groups: 0 };
  byMat[g.materialIndex].tris += g.count / 3;
  byMat[g.materialIndex].groups += 1;
}
for (const mi of Object.keys(byMat).map(Number).sort((a, b) => a - b)) {
  const e = byMat[mi];
  console.log(`  materialIndex=${mi} region=${e.region ?? "_base_vertex_color"} groups=${e.groups} triangles=${e.tris}`);
}

// Check a vertex index appearing in TWO different materialIndex groups (would be a real bug)
const vidToMaterials = new Map<number, Set<number>>();
const idxArr = geo.index!.array;
for (const g of geo.groups as { start: number; count: number; materialIndex: number }[]) {
  for (let i = g.start; i < g.start + g.count; i++) {
    const vid = idxArr[i];
    if (!vidToMaterials.has(vid)) vidToMaterials.set(vid, new Set());
    vidToMaterials.get(vid)!.add(g.materialIndex);
  }
}
let multiMatVids = 0;
for (const [, mats] of vidToMaterials) if (mats.size > 1) multiMatVids++;
console.log(`\nvertices referenced by 2+ DIFFERENT materialIndex groups: ${multiMatVids} (expected: >0, real UV-chart-boundary duplicates handled via extraSourceVids -- NOT a bug by itself)`);

// Eye vertex groups: fetch vertex_group_names/vertex_groups directly (same
// helpers loader.ts itself uses) to look at sclera/iris/pupil vertices.
const npzRes = await fetch("/models/gnm/gnm_head_v3.npz");
const npzBuf = await npzRes.arrayBuffer();
const entries = await extractZipEntries(npzBuf, ["vertex_groups.npy"]);
const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
const [groupCount, vCount] = groupsNpy.shape;
const groupsData = groupsNpy.data as Float32Array;
function rowMask(row: number): boolean[] {
  const out: boolean[] = new Array(vCount);
  for (let i = 0; i < vCount; i++) out[i] = groupsData[row * vCount + i] > 0.5;
  return out;
}
const SCLERA_ROW = 19, IRIS_ROW = 20, PUPIL_ROW = 21, LEFT_EYE_ROW = 14, RIGHT_EYE_ROW = 15;
const sclera = rowMask(SCLERA_ROW), iris = rowMask(IRIS_ROW), pupil = rowMask(PUPIL_ROW);
const leftEye = rowMask(LEFT_EYE_ROW), rightEye = rowMask(RIGHT_EYE_ROW);

const posArr = geo.attributes.position.array;
const colArr = geo.attributes.color.array;
const normArr = geo.attributes.normal.array;

function srgbEncode(linear: number): number {
  const c = linear <= 0.0031308 ? linear * 12.92 : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
  return Math.round(c * 255);
}

function dumpEyeSide(label: string, sideMask: boolean[]) {
  console.log(`\n--- ${label} sclera vertices (real runtime position/color/normal) ---`);
  let printed = 0;
  for (let vid = 0; vid < vCount && printed < 8; vid++) {
    if (sclera[vid] && sideMask[vid]) {
      const px = posArr[vid * 3], py = posArr[vid * 3 + 1], pz = posArr[vid * 3 + 2];
      const cr = srgbEncode(colArr[vid * 3]), cg = srgbEncode(colArr[vid * 3 + 1]), cb = srgbEncode(colArr[vid * 3 + 2]);
      const nx = normArr[vid * 3], ny = normArr[vid * 3 + 1], nz = normArr[vid * 3 + 2];
      const mats = Array.from(vidToMaterials.get(vid) ?? []);
      console.log(`  vid=${vid} pos=(${px.toFixed(4)},${py.toFixed(4)},${pz.toFixed(4)}) color(sRGB)=(${cr},${cg},${cb}) normal=(${nx.toFixed(2)},${ny.toFixed(2)},${nz.toFixed(2)}) materialIndex(es)=${JSON.stringify(mats)}`);
      printed++;
    }
  }
}
dumpEyeSide("LEFT_EYE", leftEye);
dumpEyeSide("RIGHT_EYE", rightEye);

// Aggregate sRGB-encoded (i.e. what the GPU actually samples) luminance per side
function aggSide(label: string, mask: boolean[], groupMask: boolean[]) {
  let sum = 0, n = 0, maxLum = 0;
  for (let vid = 0; vid < vCount; vid++) {
    if (mask[vid] && groupMask[vid]) {
      const r = srgbEncode(colArr[vid * 3]), g = srgbEncode(colArr[vid * 3 + 1]), b = srgbEncode(colArr[vid * 3 + 2]);
      const lum = (r + g + b) / 3;
      sum += lum; n++;
      if (lum > maxLum) maxLum = lum;
    }
  }
  console.log(`${label}: n=${n} meanLum(sRGB)=${n ? (sum / n).toFixed(1) : "-"} maxLum(sRGB)=${maxLum}`);
}
console.log("\n=== sRGB-encoded (actual GPU-sampled) luminance, sclera by side ===");
aggSide("left_sclera", sclera, leftEye);
aggSide("right_sclera", sclera, rightEye);

console.log("\nDONE");

// Refine: material 0 (_base_vertex_color) never samples UV -- only
// vertices shared between TWO+ DIFFERENT NON-ZERO (texture-sampling)
// materialIndex groups can actually manifest a wrong-texture-sample bug,
// since each such group reads the SAME shared uv01 value into a
// DIFFERENT independent atlas chart's own coordinate system.
let realBugCandidates = 0;
const examples: number[] = [];
for (const [vid, mats] of vidToMaterials) {
  const nonZero = Array.from(mats).filter((m) => m !== 0);
  if (nonZero.length >= 2) {
    realBugCandidates++;
    if (examples.length < 15) examples.push(vid);
  }
}
console.log(`\nvertices shared between 2+ DIFFERENT NON-ZERO (texture) materialIndex groups: ${realBugCandidates}`);
console.log("example vids:", examples);
for (const vid of examples.slice(0, 5)) {
  const mats = Array.from(vidToMaterials.get(vid)!);
  const u = geo.attributes.uv!.array[vid * 2], v = geo.attributes.uv!.array[vid * 2 + 1];
  const regions = mats.filter((m) => m !== 0).map((m) => atlasMaterialsOrdered.find((a: any) => a.materialIndex === m)?.region);
  console.log(`  vid=${vid} materialIndexes=${JSON.stringify(mats)} regions=${JSON.stringify(regions)} SHARED_uv=(${u.toFixed(4)},${v.toFixed(4)})`);
}

// #7 requested check: normal discontinuity across real topology edges, in
// the SAME forehead/cheek/orbital/nose/mouth/chin regions the user reports
// patches in -- using the REAL post-mask runtime index (not the raw template).
const edgeSet = new Set<string>();
const idxArr2 = geo.index!.array;
for (let t = 0; t < idxArr2.length / 3; t++) {
  const a = idxArr2[t * 3], b = idxArr2[t * 3 + 1], c = idxArr2[t * 3 + 2];
  for (const [x, y] of [[a, b], [b, c], [a, c]]) {
    const key = x < y ? `${x}_${y}` : `${y}_${x}`;
    edgeSet.add(key);
  }
}
let sumAngle = 0, nEdge = 0, hardEdges = 0;
const dot3 = (i: number, j: number) => {
  const ax = normArr[i * 3], ay = normArr[i * 3 + 1], az = normArr[i * 3 + 2];
  const bx = normArr[j * 3], by = normArr[j * 3 + 1], bz = normArr[j * 3 + 2];
  return ax * bx + ay * by + az * bz;
};
for (const key of edgeSet) {
  const [xs, ys] = key.split("_");
  const x = Number(xs), y = Number(ys);
  const d = Math.max(-1, Math.min(1, dot3(x, y)));
  const angleDeg = (Math.acos(d) * 180) / Math.PI;
  sumAngle += angleDeg;
  nEdge++;
  if (angleDeg > 45) hardEdges++;
}
console.log(`\n=== NORMAL discontinuity across ${nEdge} real topology edges (post-mask) ===`);
console.log(`mean angle: ${(sumAngle / nEdge).toFixed(2)} deg, edges with normal-angle > 45deg: ${hardEdges} (${(100 * hardEdges / nEdge).toFixed(2)}%)`);

// Export final post-mask runtime geometry for an independent SOFTWARE
// rasterization test (no WebGL/browser needed) -- lets us actually SEE
// whether polygon artifacts align with materialIndex boundaries, using a
// completely separate code path from Canvas3D's own WebGL pipeline (any
// artifact reproduced here rules out a WebGL/driver-specific cause and
// confirms it's already present in the DATA/geometry itself).
import { writeFileSync } from "fs";
const exportObj = {
  position: Array.from(geo.attributes.position.array),
  index: Array.from(geo.index!.array),
  colorSrgb: Array.from(colArr).map((c: number) => srgbEncode(c)),
  normal: Array.from(normArr),
  groups: (geo.groups as { start: number; count: number; materialIndex: number }[]).map((g) => ({
    start: g.start, count: g.count, materialIndex: g.materialIndex,
    region: atlasMaterialsOrdered.find((a: any) => a.materialIndex === g.materialIndex)?.region ?? null,
  })),
  uv: geo.attributes.uv ? Array.from(geo.attributes.uv.array) : null,
};
writeFileSync("scratchpad/runtime_geometry_export.json", JSON.stringify(exportObj));
console.log("\nExported scratchpad/runtime_geometry_export.json");
