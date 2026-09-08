import * as THREE from "three";
const BASE = "http://localhost:3000";
const realFetch = globalThis.fetch;
globalThis.fetch = ((input: any, init?: any) => {
  if (typeof input === "string" && input.startsWith("/")) return realFetch(BASE + input, init);
  return realFetch(input, init);
}) as typeof fetch;

const { fetchMultiViewAtlas } = await import("../src/lib/gnm/camera-pose");
const { buildGnmGeometryFromPositions } = await import("../src/lib/gnm/geometry");
const { loadGnmHeadRaw } = await import("../src/lib/gnm/loader");
const { extractZipEntries } = await import("../src/lib/gnm/zip");
const { parseNpy } = await import("../src/lib/gnm/npy");

const fit = await fetchMultiViewAtlas("0e9e1d90-2478-4d49-872f-c80772c4bc4f");
const raw = await loadGnmHeadRaw();
const positions = Float32Array.from(fit.fittedPositions);
const built = buildGnmGeometryFromPositions(positions, raw);
const colorsLin = new Float32Array(fit.vertexColors.length);
for (let i = 0; i < colorsLin.length; i++) {
  const c = fit.vertexColors[i] / 255;
  colorsLin[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}
built.geometry.setAttribute("color", new THREE.BufferAttribute(colorsLin, 3));
built.geometry.computeVertexNormals();

const npzRes = await fetch("/models/gnm/gnm_head_v3.npz");
const npzBuf = await npzRes.arrayBuffer();
const entries = await extractZipEntries(npzBuf, ["vertex_groups.npy"]);
const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
const [, vCount] = groupsNpy.shape;
const groupsData = groupsNpy.data as Float32Array;
const rowMask = (row: number) => { const o: boolean[] = new Array(vCount); for (let i=0;i<vCount;i++) o[i]=groupsData[row*vCount+i]>0.5; return o; };
const sclera = rowMask(19), leftEye = rowMask(14), rightEye = rowMask(15);

const colArr = built.geometry.attributes.color.array;
const normArr = built.geometry.attributes.normal.array;
const posArr = built.geometry.attributes.position.array;
const SCLERA_FALLBACK_LIN = [232,227,220].map(c => { c/=255; return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4); });

// "front-facing at 0deg render" == normal.z close to +1 (camera looks down -Z or +Z depending on convention;
// use the same convention buildGnmGeometryFromPositions/raw data uses -- just rank by normal.z descending
// and look at the TOP-front hemisphere, sign-agnostic by checking both).
function analyze(label: string, mask: boolean[]) {
  const vids: number[] = [];
  for (let i=0;i<vCount;i++) if (sclera[i] && mask[i]) vids.push(i);
  // z axis: determine sign by using whichever gives LARGER max (i.e. camera-facing side)
  const zs = vids.map(v => normArr[v*3+2]);
  const useSign = (zs.reduce((a,b)=>a+b,0) >= 0) ? 1 : -1;
  const frontVids = vids.filter(v => normArr[v*3+2]*useSign > 0.3);
  let fallbackCount = 0;
  for (const v of frontVids) {
    const isFallback = Math.abs(colArr[v*3]-SCLERA_FALLBACK_LIN[0])<0.001 && Math.abs(colArr[v*3+1]-SCLERA_FALLBACK_LIN[1])<0.001;
    if (isFallback) fallbackCount++;
  }
  console.log(`${label}: total_sclera=${vids.length} front-facing(|normal.z|>0.3)=${frontVids.length} fallback_among_front=${fallbackCount} (${frontVids.length?(100*fallbackCount/frontVids.length).toFixed(1):0}%)`);
}
analyze("left_sclera", leftEye);
analyze("right_sclera", rightEye);
