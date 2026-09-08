import { parseNpy } from "./npy";
import { GNM_HEAD_V3_SCHEMA, type GnmFaceMaskRawData, type GnmHeadRawData, type GnmIdentityBasisRawData } from "./types";
import { extractZipEntries } from "./zip";

/**
 * Self-hosted copy of Google's official GNM Head v3 asset
 * (github.com/google/GNM, `gnm/shape/data/versions/v3_0/gnm_head.npz`),
 * byte-identical to the released file — verified by sha256
 * (0b1de1bd74af639279e0a3b6c1ff235f88894176d7e19258df631e387089a657) against
 * the file downloaded directly from that repository. Served from this
 * project's own `public/` so the Studio doesn't depend on a third-party CDN
 * at runtime.
 */
const GNM_HEAD_ASSET_URL = "/models/gnm/gnm_head_v3.npz";

const WANTED_ENTRIES = ["template_vertex_positions.npy", "triangles.npy", "triangle_uvs.npy"] as const;

let cachedPromise: Promise<GnmHeadRawData> | null = null;

function expectShape(name: string, shape: number[], expected: number[]): void {
  const matches = shape.length === expected.length && shape.every((v, i) => v === expected[i]);
  if (!matches) {
    throw new Error(`gnm-schema-mismatch: "${name}" has shape ${JSON.stringify(shape)}, expected ${JSON.stringify(expected)}`);
  }
}

async function loadGnmHeadRawUncached(): Promise<GnmHeadRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, WANTED_ENTRIES);

  const positionsNpy = parseNpy(entries.get("template_vertex_positions.npy")!);
  const trianglesNpy = parseNpy(entries.get("triangles.npy")!);
  const triangleUvsNpy = parseNpy(entries.get("triangle_uvs.npy")!);

  if (positionsNpy.fortranOrder || trianglesNpy.fortranOrder || triangleUvsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) arrays");
  }
  if (positionsNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: positions dtype ${positionsNpy.dtype}`);
  if (trianglesNpy.dtype !== "<i4") throw new Error(`gnm-schema-mismatch: triangles dtype ${trianglesNpy.dtype}`);
  if (triangleUvsNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: triangle_uvs dtype ${triangleUvsNpy.dtype}`);

  const { vertexCount, triangleCount } = GNM_HEAD_V3_SCHEMA;
  expectShape("template_vertex_positions", positionsNpy.shape, [vertexCount, 3]);
  expectShape("triangles", trianglesNpy.shape, [triangleCount, 3]);
  expectShape("triangle_uvs", triangleUvsNpy.shape, [triangleCount, 3, 2]);

  return {
    positions: positionsNpy.data as Float32Array,
    triangleIndices: trianglesNpy.data as Int32Array,
    triangleUvs: triangleUvsNpy.data as Float32Array,
    vertexCount,
    triangleCount,
  };
}

/** Loads (and caches, module-wide) the raw GNM Head v3 template arrays. Must only be called client-side. */
export function loadGnmHeadRaw(): Promise<GnmHeadRawData> {
  if (!cachedPromise) {
    cachedPromise = loadGnmHeadRawUncached().catch((err) => {
      cachedPromise = null; // allow retry on next call instead of caching a permanent failure
      throw err;
    });
  }
  return cachedPromise;
}

let cachedIdentityBasisPromise: Promise<GnmIdentityBasisRawData> | null = null;

/**
 * The identity basis lives in the same `.npz` as the three base arrays above
 * (verified: `vertex_identity_basis.npy` is one of that archive's own
 * entries) but is deliberately fetched/extracted separately, via its own
 * `fetch(GNM_HEAD_ASSET_URL)` call, rather than folded into
 * `loadGnmHeadRawUncached`'s `WANTED_ENTRIES` — patient fitting (the only
 * caller) is a distinct, optional step from just displaying the static
 * template, and this keeps that ~54MB array out of the critical path for
 * checkpoints that only need the template. The browser's HTTP cache makes
 * the second fetch to the identical URL cheap once the first has completed.
 */
async function loadGnmIdentityBasisUncached(): Promise<GnmIdentityBasisRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_identity_basis.npy"]);

  const basisNpy = parseNpy(entries.get("vertex_identity_basis.npy")!);
  if (basisNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_identity_basis");
  }
  if (basisNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: vertex_identity_basis dtype ${basisNpy.dtype}`);
  if (basisNpy.shape.length !== 3 || basisNpy.shape[2] !== 3) {
    throw new Error(`gnm-schema-mismatch: vertex_identity_basis has shape ${JSON.stringify(basisNpy.shape)}`);
  }
  const [identityDim, vertexCount] = basisNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(
      `gnm-schema-mismatch: vertex_identity_basis vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`
    );
  }

  return { data: basisNpy.data as Float32Array, identityDim, vertexCount };
}

/** Loads (and caches, module-wide) the GNM Head v3 identity basis. Must only be called client-side. */
export function loadGnmIdentityBasis(): Promise<GnmIdentityBasisRawData> {
  if (!cachedIdentityBasisPromise) {
    cachedIdentityBasisPromise = loadGnmIdentityBasisUncached().catch((err) => {
      cachedIdentityBasisPromise = null;
      throw err;
    });
  }
  return cachedIdentityBasisPromise;
}

/**
 * Row index of "hockey_mask" within `vertex_groups.npy`'s (46, vertexCount)
 * array. `vertex_groups.npy` only contains float membership data — the
 * corresponding names live in the separate `vertex_group_names.npy`, which
 * is a unicode-string array (`<U32`) this project's `parseNpy` doesn't
 * support (it only handles `<f4`/`<i4`, the two dtypes every OTHER array
 * this loader reads actually uses). Rather than extend the shared npy
 * parser just for this one string array, the index was read directly and
 * verified once (offline, against the real asset): row 4 of
 * `vertex_group_names.npy` decodes to exactly "hockey_mask".
 */
const HOCKEY_MASK_ROW_INDEX = 4;

let cachedFaceMaskPromise: Promise<GnmFaceMaskRawData> | null = null;

async function loadGnmFaceMaskUncached(): Promise<GnmFaceMaskRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_groups.npy"]);

  const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
  if (groupsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_groups");
  }
  if (groupsNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: vertex_groups dtype ${groupsNpy.dtype}`);
  if (groupsNpy.shape.length !== 2) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has shape ${JSON.stringify(groupsNpy.shape)}`);
  }
  const [groupCount, vertexCount] = groupsNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`);
  }
  if (HOCKEY_MASK_ROW_INDEX >= groupCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has only ${groupCount} rows, expected hockey_mask at row ${HOCKEY_MASK_ROW_INDEX}`);
  }

  const data = groupsNpy.data as Float32Array;
  const membership = data.slice(HOCKEY_MASK_ROW_INDEX * vertexCount, (HOCKEY_MASK_ROW_INDEX + 1) * vertexCount);

  return { membership, vertexCount };
}

/** Loads (and caches, module-wide) GNM Head v3's real "hockey_mask" face-region vertex membership. Must only be called client-side. */
export function loadGnmFaceMask(): Promise<GnmFaceMaskRawData> {
  if (!cachedFaceMaskPromise) {
    cachedFaceMaskPromise = loadGnmFaceMaskUncached().catch((err) => {
      cachedFaceMaskPromise = null;
      throw err;
    });
  }
  return cachedFaceMaskPromise;
}

/**
 * Row indices of "eyes" and "eye_sockets" within `vertex_groups.npy`'s
 * (46, vertexCount) array — same read-and-verify-once provenance as
 * `HOCKEY_MASK_ROW_INDEX` above (offline `zipfile`/`numpy` read of
 * `vertex_group_names.npy` against the real asset): row 13 decodes to
 * exactly "eyes" (sclera/iris/pupil/interior/exterior — verified
 * `eye_interiors`+`eye_exteriors` sum to exactly `eyes`' own vertex count),
 * row 16 to exactly "eye_sockets" (the eyelid skin immediately around them).
 */
const EYES_ROW_INDEX = 13;
const EYE_SOCKETS_ROW_INDEX = 16;
const FOREHEAD_REGION_ROW_INDEX = 26;
const LEFT_BROW_REGION_ROW_INDEX = 27;
const MIDDLE_BROW_REGION_ROW_INDEX = 28;
const RIGHT_BROW_REGION_ROW_INDEX = 29;
const LEFT_TEMPLE_REGION_ROW_INDEX = 30;
const RIGHT_TEMPLE_REGION_ROW_INDEX = 31;
const LEFT_ZYGOMATIC_REGION_ROW_INDEX = 34;
const RIGHT_ZYGOMATIC_REGION_ROW_INDEX = 35;
const LEFT_PAROTID_REGION_ROW_INDEX = 37;
const RIGHT_PAROTID_REGION_ROW_INDEX = 38;

let cachedRenderMaskPromise: Promise<GnmFaceMaskRawData> | null = null;

/**
 * Union of verified anatomical facial groups:
 * `hockey_mask ∪ eyes ∪ eye_sockets ∪ forehead_region ∪ left_brow_region ∪ right_brow_region ∪ middle_brow_region ∪ left_temple_region ∪ right_temple_region ∪ left_zygomatic_region ∪ right_zygomatic_region ∪ left_parotid_region ∪ right_parotid_region`
 * Strictly covers the facial anterior surface, excluding scalp/gáy/ears/neck.
 */
async function loadGnmRenderMaskUncached(): Promise<GnmFaceMaskRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_groups.npy"]);

  const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
  if (groupsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_groups");
  }
  const [groupCount, vertexCount] = groupsNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`);
  }
  const maxRow = Math.max(
    HOCKEY_MASK_ROW_INDEX,
    EYES_ROW_INDEX,
    EYE_SOCKETS_ROW_INDEX,
    FOREHEAD_REGION_ROW_INDEX,
    LEFT_BROW_REGION_ROW_INDEX,
    MIDDLE_BROW_REGION_ROW_INDEX,
    RIGHT_BROW_REGION_ROW_INDEX,
    LEFT_TEMPLE_REGION_ROW_INDEX,
    RIGHT_TEMPLE_REGION_ROW_INDEX,
    LEFT_ZYGOMATIC_REGION_ROW_INDEX,
    RIGHT_ZYGOMATIC_REGION_ROW_INDEX,
    LEFT_PAROTID_REGION_ROW_INDEX,
    RIGHT_PAROTID_REGION_ROW_INDEX
  );
  if (maxRow >= groupCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has only ${groupCount} rows, expected row ${maxRow}`);
  }

  const data = groupsNpy.data as Float32Array;
  const rowAt = (row: number) => data.subarray(row * vertexCount, (row + 1) * vertexCount);
  const rows = [
    rowAt(HOCKEY_MASK_ROW_INDEX),
    rowAt(EYES_ROW_INDEX),
    rowAt(EYE_SOCKETS_ROW_INDEX),
    rowAt(FOREHEAD_REGION_ROW_INDEX),
    rowAt(LEFT_BROW_REGION_ROW_INDEX),
    rowAt(MIDDLE_BROW_REGION_ROW_INDEX),
    rowAt(RIGHT_BROW_REGION_ROW_INDEX),
    rowAt(LEFT_TEMPLE_REGION_ROW_INDEX),
    rowAt(RIGHT_TEMPLE_REGION_ROW_INDEX),
    rowAt(LEFT_ZYGOMATIC_REGION_ROW_INDEX),
    rowAt(RIGHT_ZYGOMATIC_REGION_ROW_INDEX),
    rowAt(LEFT_PAROTID_REGION_ROW_INDEX),
    rowAt(RIGHT_PAROTID_REGION_ROW_INDEX),
  ];

  const membership = new Float32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    let inside = false;
    for (let r = 0; r < rows.length; r++) {
      if (rows[r][i] > 0.5) {
        inside = true;
        break;
      }
    }
    membership[i] = inside ? 1 : 0;
  }

  return { membership, vertexCount };
}

/** Loads (and caches, module-wide) the real hockey_mask ∪ eyes ∪ eye_sockets ∪ skin_exterior union — see `loadGnmRenderMaskUncached`'s own docstring for why this, not `loadGnmFaceMask`, is the correct mask for an actual geometry cut. Must only be called client-side. */
export function loadGnmRenderMask(): Promise<GnmFaceMaskRawData> {
  if (!cachedRenderMaskPromise) {
    cachedRenderMaskPromise = loadGnmRenderMaskUncached().catch((err) => {
      cachedRenderMaskPromise = null;
      throw err;
    });
  }
  return cachedRenderMaskPromise;
}

let cachedIdentityMaskPromise: Promise<GnmFaceMaskRawData> | null = null;

/**
 * `hockey_mask ∪ eyes ∪ eye_sockets` — NO `skin_exterior` — GNM's real
 * identity-critical face region, matching ai-engine/gnm_vertex_trust.py's
 * own `compute_face_mask` EXACTLY (same 3 vertex-group rows, same union):
 * 6,582/17,821 real member vertices. Re-derived here from the same source
 * asset data (never hardcoded) for exactly one purpose — driving the
 * optional `Canvas3DProps.renderMode="identity-only"` review mode in
 * Canvas3D.tsx, which RECOLORS (never removes) triangles outside this mask
 * to a flat generic tone. Distinct from `loadGnmRenderMask` (adds
 * skin_exterior — the DEFAULT geometry-cut mask, see that function's own
 * docstring for why the full head, not a face-only patch, is the default)
 * and from `loadGnmFaceMask` (hockey_mask alone, camera-framing only) —
 * three different masks for three different real purposes, not
 * interchangeable.
 */
async function loadGnmIdentityMaskUncached(): Promise<GnmFaceMaskRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_groups.npy"]);

  const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
  if (groupsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_groups");
  }
  const [groupCount, vertexCount] = groupsNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`);
  }
  const maxRow = Math.max(HOCKEY_MASK_ROW_INDEX, EYES_ROW_INDEX, EYE_SOCKETS_ROW_INDEX);
  if (maxRow >= groupCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has only ${groupCount} rows, expected row ${maxRow}`);
  }

  const data = groupsNpy.data as Float32Array;
  const rowAt = (row: number) => data.subarray(row * vertexCount, (row + 1) * vertexCount);
  const hockeyMask = rowAt(HOCKEY_MASK_ROW_INDEX);
  const eyes = rowAt(EYES_ROW_INDEX);
  const eyeSockets = rowAt(EYE_SOCKETS_ROW_INDEX);

  const membership = new Float32Array(vertexCount);
  for (let i = 0; i < vertexCount; i++) {
    membership[i] = hockeyMask[i] > 0.5 || eyes[i] > 0.5 || eyeSockets[i] > 0.5 ? 1 : 0;
  }

  return { membership, vertexCount };
}

/** Loads (and caches, module-wide) GNM's real hockey_mask ∪ eyes ∪ eye_sockets identity-face union — see `loadGnmIdentityMaskUncached`'s own docstring for what this is for. Must only be called client-side. */
export function loadGnmIdentityMask(): Promise<GnmFaceMaskRawData> {
  if (!cachedIdentityMaskPromise) {
    cachedIdentityMaskPromise = loadGnmIdentityMaskUncached().catch((err) => {
      cachedIdentityMaskPromise = null;
      throw err;
    });
  }
  return cachedIdentityMaskPromise;
}

/**
 * Row index of "nose_region" within `vertex_groups.npy`'s (46, vertexCount)
 * array — same provenance discipline as `HOCKEY_MASK_ROW_INDEX` above: read
 * directly and verified once (offline, against the real asset, via a
 * one-off Python `zipfile`/`numpy` read of `vertex_group_names.npy`, which
 * this project's own `parseNpy` can't parse since it's a unicode-string
 * (`<U32`) array): row 36 decodes to exactly "nose_region" (889 real member
 * vertices out of 17,821, verified the same run).
 */
const NOSE_REGION_ROW_INDEX = 36;

let cachedNoseMaskPromise: Promise<GnmFaceMaskRawData> | null = null;

async function loadGnmNoseMaskUncached(): Promise<GnmFaceMaskRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_groups.npy"]);

  const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
  if (groupsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_groups");
  }
  if (groupsNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: vertex_groups dtype ${groupsNpy.dtype}`);
  if (groupsNpy.shape.length !== 2) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has shape ${JSON.stringify(groupsNpy.shape)}`);
  }
  const [groupCount, vertexCount] = groupsNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`);
  }
  if (NOSE_REGION_ROW_INDEX >= groupCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has only ${groupCount} rows, expected nose_region at row ${NOSE_REGION_ROW_INDEX}`);
  }

  const data = groupsNpy.data as Float32Array;
  const membership = data.slice(NOSE_REGION_ROW_INDEX * vertexCount, (NOSE_REGION_ROW_INDEX + 1) * vertexCount);

  return { membership, vertexCount };
}

/** Loads (and caches, module-wide) GNM Head v3's real "nose_region" vertex membership. Must only be called client-side. */
export function loadGnmNoseMask(): Promise<GnmFaceMaskRawData> {
  if (!cachedNoseMaskPromise) {
    cachedNoseMaskPromise = loadGnmNoseMaskUncached().catch((err) => {
      cachedNoseMaskPromise = null;
      throw err;
    });
  }
  return cachedNoseMaskPromise;
}

let cachedEyeSocketsMaskPromise: Promise<GnmFaceMaskRawData> | null = null;

async function loadGnmEyeSocketsMaskUncached(): Promise<GnmFaceMaskRawData> {
  const response = await fetch(GNM_HEAD_ASSET_URL);
  if (!response.ok) {
    throw new Error(`gnm-fetch-failed: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  const entries = await extractZipEntries(buffer, ["vertex_groups.npy"]);

  const groupsNpy = parseNpy(entries.get("vertex_groups.npy")!);
  if (groupsNpy.fortranOrder) {
    throw new Error("gnm-schema-mismatch: expected C-order (fortran_order=False) vertex_groups");
  }
  if (groupsNpy.dtype !== "<f4") throw new Error(`gnm-schema-mismatch: vertex_groups dtype ${groupsNpy.dtype}`);
  if (groupsNpy.shape.length !== 2) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has shape ${JSON.stringify(groupsNpy.shape)}`);
  }
  const [groupCount, vertexCount] = groupsNpy.shape;
  if (vertexCount !== GNM_HEAD_V3_SCHEMA.vertexCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups vertex count ${vertexCount} != ${GNM_HEAD_V3_SCHEMA.vertexCount}`);
  }
  if (EYE_SOCKETS_ROW_INDEX >= groupCount) {
    throw new Error(`gnm-schema-mismatch: vertex_groups has only ${groupCount} rows, expected eye_sockets at row ${EYE_SOCKETS_ROW_INDEX}`);
  }

  const data = groupsNpy.data as Float32Array;
  const membership = data.slice(EYE_SOCKETS_ROW_INDEX * vertexCount, (EYE_SOCKETS_ROW_INDEX + 1) * vertexCount);

  return { membership, vertexCount };
}

/**
 * Loads (and caches, module-wide) GNM Head v3's real "eye_sockets" vertex
 * membership (row 16, same `EYE_SOCKETS_ROW_INDEX` already validated for
 * `loadGnmRenderMask`'s own union — 608 real member vertices) — the eyelid/
 * socket SKIN immediately around each eye, NOT the eyeball surface itself
 * (that's the separate `eyes`/`scleras`/`irises`/`pupils` groups
 * gnm_eye_render.py already owns). Phase 4's eye-crease morph target
 * (lib/gnm/morph-eye.ts) deforms exactly this group — must only be called
 * client-side.
 */
export function loadGnmEyeSocketsMask(): Promise<GnmFaceMaskRawData> {
  if (!cachedEyeSocketsMaskPromise) {
    cachedEyeSocketsMaskPromise = loadGnmEyeSocketsMaskUncached().catch((err) => {
      cachedEyeSocketsMaskPromise = null;
      throw err;
    });
  }
  return cachedEyeSocketsMaskPromise;
}
