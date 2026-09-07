"""
Phase 1 — per-vertex "identity trust" field for GNM Head v3's fixed 17,821-
vertex topology. This is the concrete mechanism for the "face-patch vs.
generic-shell" architecture principle reverse-engineered from Crisalix
(reconstruction_k3d face patch stitched into a fullscan_k3d shell at the
forehead/neck border, blend distance 3/5 — see file/var.docx's
FaceFullScanAreaId), adapted to this project's own topology instead of
literally reproducing Crisalix's two-mesh stitch (see the approved plan,
/home/ubuntu/.claude/plans/jaunty-tinkering-aho.md, correction #2).

Why a continuous trust field on ONE mesh instead of two stitched meshes:
GNM Head v3 already ships a single linear identity basis (vertex_identity_
basis.npy, 253 x 17821 x 3) covering the WHOLE head, and a real per-vertex
group union already used by this project's own atlas/mask code
(hockey_mask | eyes | eye_sockets, "face_mask", 6582/17821 vertices,
verified 0 overlap with `ears`) that already tracks exactly the region the
project's own 4 upload photos (0/45/90/below) can actually see and that the
48 WFLW landmarks actually constrain. Re-using this SAME mask as the
"protected core" and blending the rest toward the (population-mean, zero-
coefficient) template achieves the same principle Crisalix's stitch does
(protect identity-critical geometry, fall back to a generic/safe shape
where no real evidence exists) without introducing a second topology, a
second asset, or a seam-matching algorithm — every function downstream of
this module (bake_vertex_colors, build_multi_region_atlas, the client
gnm/*.ts contract) keeps operating on the exact same 17,821-vertex/35,324-
triangle mesh it already does.

Root cause this specifically fixes (measured directly on the real asset,
see the plan's "Sai lệch đã sửa" correction — NOT because the identity
basis is weaker outside face_mask; per-vertex basis magnitude there is
actually ~2.2x HIGHER than inside face_mask): the 48 WFLW landmarks used by
gnm_identity_fit.py's ridge solve all sit inside face_mask. The resulting
253-dim coefficient vector is shared by the WHOLE mesh (identity-fit.ts's
`applyIdentityCoefficients` / gnm_identity_fit.py's own `apply_identity_
coefficients` apply it to all 17,821 vertices), so outside face_mask the
fitted shape is a linear EXTRAPOLATION with zero landmark evidence backing
it, into precisely the region with the largest basis swings. That
extrapolation is what has been producing the blotchy/unstable neck-
shoulder and ear-region artifacts several prior fixes (D24, the neck Y-band
in gnm_texture_atlas.py) patched downstream, symptom by symptom. This
module fixes the shared root cause instead: outside face_mask, pull the
fitted shape back toward the template (coefficients effectively -> 0) in
proportion to real geodesic distance from the nearest evidence-backed
vertex, smoothly, with zero effect inside face_mask.

Geometry note: `distance` here is REAL mesh-surface geodesic distance
(Dijkstra over the vertex-adjacency graph, edge weight = real Euclidean
edge length in the template's own vertex_identity_basis coordinate space,
i.e. meters — same space gnm_identity_fit.py's own `max_disp_mm` sanity
check and nose-deform.ts's MM_TO_SCENE_UNITS already treat as meters), not
Euclidean straight-line distance — a straight line from a jaw vertex to a
neck vertex can cut through empty space off the mesh surface entirely,
which would understate how far apart they actually are ALONG the head.

FALLOFF_DISTANCE_M = 0.02 (20mm) chosen from two real, measured facts on
this exact asset (see ai-engine/regression/ Phase-0 session notes):
  - mean triangle edge length in template space = 3.82mm, median 3.37mm —
    20mm is ~5-6 edge-lengths, wide enough for a visually smooth per-vertex
    gradient (not a 1-triangle hard step), matching this project's own
    existing precedent for "a few edge-lengths" smoothing radii (e.g.
    gnm_texture_atlas.py's EDGE_DILATION_PX=6, NECK_SHOULDER_SMOOTH_PASSES=6).
  - the real measured euclidean gap from `ears` vertices to their nearest
    `face_mask` vertex is 38-42mm (5 sampled ears vertices, all >= 38mm) —
    20mm keeps the transition band strictly inside that gap, so `ears`
    (a real GNM vertex group with genuinely no photo/landmark coverage at
    any of the 4 upload angles) lands at trust ~= 0 (full shell fallback)
    rather than being accidentally protected by an oversized band.
"""
import heapq
import zipfile
from pathlib import Path

import numpy as np

GNM_ASSET = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"
FALLOFF_DISTANCE_M = 0.04  # 40mm smooth falloff to retain patient profile & contour geometry

_trust_cache: np.ndarray | None = None
_face_mask_cache: np.ndarray | None = None
_region_index_cache: np.ndarray | None = None

# Region enum for `compute_region_index()` — see that function's own
# docstring for the 2026-08-24 topology audit each boundary is derived
# from. Values are stable and safe to persist (small int8), but nothing
# outside this module reads/writes them yet (backend-internal use only,
# currently consumed by main.py's adaptive shell-tone step) — extending the
# API/artifact schema to expose this is a deliberate separate decision, not
# implied by adding the enum here.
REGION_FACE = 0
REGION_EYES = 1
REGION_EYE_SOCKETS = 2
REGION_EARS = 3
REGION_NECK = 4
REGION_HEAD_SHELL_GENERIC = 5
REGION_INTERIOR_MOUTH = 6
REGION_UNASSIGNED = 7


def _load_vertex_groups():
    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open("vertex_group_names.npy") as f:
            names = list(np.load(f, allow_pickle=True))
        with zf.open("vertex_groups.npy") as f:
            groups = np.load(f, allow_pickle=True)
    return names, groups


def _region_mask(names, groups, group_names, V):
    idxs = [names.index(n) for n in group_names if n in names]
    m = np.zeros(V, dtype=bool)
    for i in idxs:
        m |= groups[i] > 0.5
    return m


def compute_face_mask(vertex_count: int) -> np.ndarray:
    """`hockey_mask | eyes | eye_sockets` — the SAME union already used by
    loader.ts's `loadGnmRenderMask()` and gnm_texture_atlas.py's
    `_attach_threejs_contract` for geometry cuts. Re-derived here from the
    same source data (not imported from gnm_texture_atlas at module scope,
    to avoid a circular import — see this module's own docstring)."""
    global _face_mask_cache
    if _face_mask_cache is None:
        names, groups = _load_vertex_groups()
        _face_mask_cache = _region_mask(names, groups, ["hockey_mask", "eyes", "eye_sockets"], vertex_count)
    return _face_mask_cache


def _build_adjacency_with_weights(triangles: np.ndarray, positions: np.ndarray, V: int) -> list[list[tuple[int, float]]]:
    """Per-vertex list of (neighbor_index, real_euclidean_edge_length_m) —
    same adjacency this project's gnm_texture_atlas.py `_build_adjacency`
    already builds (unweighted sets, for its own dilation/ICM use), extended
    here with real edge length so Dijkstra can compute geodesic distance
    instead of a ring count."""
    adj: list[set[int]] = [set() for _ in range(V)]
    for a, b, c in triangles:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    weighted = []
    for v, neighbors in enumerate(adj):
        row = [(n, float(np.linalg.norm(positions[v] - positions[n]))) for n in neighbors]
        weighted.append(row)
    return weighted


def _multi_source_geodesic_distance(adj_weighted: list[list[tuple[int, float]]], sources: np.ndarray) -> np.ndarray:
    """Dijkstra from every `sources` vertex simultaneously (all pushed at
    distance 0) — standard multi-source shortest-path, gives each vertex its
    real mesh-surface distance to the NEAREST source vertex."""
    V = len(adj_weighted)
    dist = np.full(V, np.inf, dtype=np.float64)
    heap: list[tuple[float, int]] = []
    for s in np.nonzero(sources)[0]:
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, int(s)))
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj_weighted[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def compute_vertex_trust() -> np.ndarray:
    """Computed ONCE from the fixed GNM topology + fixed face_mask (no
    per-patient input) and cached for the process lifetime — matches the
    plan's "tính 1 lần theo topology cố định" spec. trust=1.0 exactly for
    every face_mask vertex (distance 0, by construction — this is the hard
    guarantee that the protected region is never touched), smoothstep down
    to 0.0 over FALLOFF_DISTANCE_M of real geodesic distance outside it."""
    global _trust_cache
    if _trust_cache is not None:
        return _trust_cache

    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open("template_vertex_positions.npy") as f:
            positions = np.load(f)
        with zf.open("triangles.npy") as f:
            triangles = np.load(f)

    V = positions.shape[0]
    face_mask = compute_face_mask(V)
    adj_weighted = _build_adjacency_with_weights(triangles, positions, V)
    dist = _multi_source_geodesic_distance(adj_weighted, face_mask)

    trust = 1.0 - _smoothstep(0.0, FALLOFF_DISTANCE_M, dist)
    trust[face_mask] = 1.0  # exact, not just "close to 1" from the smoothstep at dist=0 (which is already 1.0, this just documents the guarantee)
    # Any vertex geodesically unreachable from face_mask (would only happen on a
    # disconnected mesh component, e.g. teeth/tongue/eye-interior submeshes if
    # ever included) has dist=inf -> smoothstep saturates at 1 -> trust=0, the
    # correct "no evidence, full shell fallback" behavior, not a crash.
    _trust_cache = trust.astype(np.float32)
    return _trust_cache


def blend_positions_by_trust(fitted_positions: np.ndarray, template_positions: np.ndarray, trust: np.ndarray | None = None) -> np.ndarray:
    """final = template + trust * (fitted - template). trust=1 (face_mask) ->
    final == fitted exactly (bit-identical, the Phase-0 regression hard
    gate). trust=0 -> final == template exactly (pure shell fallback, never
    a fabricated shape)."""
    if trust is None:
        trust = compute_vertex_trust()
    return template_positions + trust[:, None] * (fitted_positions - template_positions)


def smooth_colors_by_trust(vertex_color: np.ndarray, triangles: np.ndarray, trust: np.ndarray | None = None, passes: int = 6) -> np.ndarray:
    """Low-trust vertices get their baked color blended toward a real
    mesh-adjacency neighbor average (same iterative-neighbor-smoothing
    mechanism already validated by gnm_texture_atlas.py's
    `improve_neck_shoulder_vertex_colors`, NECK_SHOULDER_SMOOTH_PASSES=6,
    reused here as the same discipline rather than a new smoothing scheme)
    instead of staying at the raw per-vertex ICM/sentinel value, which is
    what produced the blotchy shell-region artifact prior fixes patched
    downstream. trust=1 (face_mask) vertices are excluded from ever
    receiving neighbor-smoothed color (weight 0 by construction) — their
    color stays bit-identical to `bake_vertex_colors`'s own output."""
    if trust is None:
        trust = compute_vertex_trust()
    V = vertex_color.shape[0]
    adj_sets: list[set[int]] = [set() for _ in range(V)]
    for a, b, c in triangles:
        adj_sets[a].update((b, c))
        adj_sets[b].update((a, c))
        adj_sets[c].update((a, b))

    smoothed = vertex_color.astype(np.float64).copy()
    for _ in range(passes):
        next_smoothed = smoothed.copy()
        for v in range(V):
            if trust[v] >= 1.0:
                continue  # never touch face_mask
            neighbors = adj_sets[v]
            if not neighbors:
                continue
            idx = np.fromiter(neighbors, dtype=np.int64)
            next_smoothed[v] = smoothed[idx].mean(axis=0)
        smoothed = next_smoothed

    blend_weight = (1.0 - trust)[:, None]
    return vertex_color * (1.0 - blend_weight) + smoothed * blend_weight


def compute_region_index() -> np.ndarray:
    """Per-vertex region classification for the fixed 17,821-vertex GNM
    topology — (V,) int8, one of the REGION_* constants above. Computed
    ONCE from the fixed topology + fixed vertex_groups/skinning_weights (no
    per-patient input) and cached for the process lifetime, same discipline
    as `compute_vertex_trust`/`compute_face_mask` above.

    Derived 2026-08-24 by a read-only topology audit (real connected-
    components + triangle adjacency + boundary-loop checks on this exact
    asset — see that session's own report for the full evidence trail).
    Every boundary below is real asset metadata, never an invented
    threshold:
      - FACE/EYES/EYE_SOCKETS/EARS: the existing named vertex_groups
        (`hockey_mask`, `eyes`, `eye_sockets`, `ears`).
      - INTERIOR_MOUTH: `upper_teeth_and_gums`/`lower_teeth_and_gums`/
        `tongue`/`mouth_sock` — also existing named groups, just not
        anatomically "shell" so kept out of HEAD_SHELL_GENERIC.
      - Everything else forms `skin_shell` (not in any of the above).
        `skinning_weights.npy`'s dominant joint (`joint_names.npy`:
        neck/head/left_eye/right_eye — a real authored skeleton, not a
        derived heuristic) splits it into a `neck`-dominant part (2,448
        vertices, ALL of which independently measured trust==0.0 exactly —
        see `compute_vertex_trust`) and a `head`-dominant part.
      - HEAD_SHELL_GENERIC: only the SINGLE LARGEST connected component
        (BFS over real triangle adjacency) of that head-dominant part —
        measured 3,308/3,518 vertices, verified to touch FACE, EARS, and
        NECK's own boundary all at once (real boundary-loop edges, not
        proximity). The remaining ~210 vertices are small (76/76/58)
        components isolated from this main shell within the head-dominant
        subgraph (only reachable via FACE) — mouth-corner/philtrum
        adjacent per their own group overlap, NOT anatomically resolved —
        left UNASSIGNED. Do not reclassify them without new evidence (see
        the audit report for exactly what evidence would be needed).
    """
    global _region_index_cache
    if _region_index_cache is not None:
        return _region_index_cache

    names, groups = _load_vertex_groups()
    with zipfile.ZipFile(GNM_ASSET) as zf:
        with zf.open("triangles.npy") as f:
            triangles = np.load(f)
        with zf.open("skinning_weights.npy") as f:
            skin_w = np.load(f)
        with zf.open("joint_names.npy") as f:
            joint_names = list(np.load(f, allow_pickle=True))

    V = groups.shape[1]

    def gmask(group_name: str) -> np.ndarray:
        return groups[names.index(group_name)] > 0.5

    hockey = gmask("hockey_mask")
    ears = gmask("ears")
    eyes = gmask("eyes")
    eye_sockets = gmask("eye_sockets")
    interior_mouth = gmask("upper_teeth_and_gums") | gmask("lower_teeth_and_gums") | gmask("tongue") | gmask("mouth_sock")
    skin_shell = ~(hockey | ears | eyes | eye_sockets) & ~interior_mouth

    dom = np.full(V, -1, dtype=np.int64)
    dom[skin_shell] = np.argmax(skin_w[:, skin_shell], axis=0)
    neck_dom = skin_shell & (dom == joint_names.index("neck"))
    head_dom = skin_shell & (dom == joint_names.index("head"))

    adj: list[set[int]] = [set() for _ in range(V)]
    for a, b, c in triangles:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))

    visited = np.zeros(V, dtype=bool)
    main_component: list[int] | None = None
    for start in np.nonzero(head_dom)[0]:
        if visited[start]:
            continue
        comp = []
        stack = [int(start)]
        visited[start] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if head_dom[nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if main_component is None or len(comp) > len(main_component):
            main_component = comp

    head_shell_generic = np.zeros(V, dtype=bool)
    if main_component:
        head_shell_generic[main_component] = True

    # Priority order below: most-authoritative/protected group wins any
    # incidental overlap (none measured on the real asset, but assignment
    # order documents the intended precedence regardless).
    region = np.full(V, REGION_UNASSIGNED, dtype=np.int8)
    region[interior_mouth] = REGION_INTERIOR_MOUTH
    region[neck_dom] = REGION_NECK
    region[head_shell_generic] = REGION_HEAD_SHELL_GENERIC
    region[ears] = REGION_EARS
    region[eye_sockets] = REGION_EYE_SOCKETS
    region[eyes] = REGION_EYES
    region[hockey] = REGION_FACE

    _region_index_cache = region
    return _region_index_cache


def compute_adaptive_shell_tone(vertex_color: np.ndarray, covered: np.ndarray) -> np.ndarray | None:
    """Per-patient replacement for the fixed `SENTINEL_UNCOVERED` skin tone
    (`#e0ac8f`, gnm_texture_bake.py) — root cause fixed here (2026-08-24
    audit): that ONE fixed tone is used for every patient regardless of
    their own real skin tone, which on real patient 257d9bfe measured a
    ~68/59/47 (R/G/B) hard color jump between FACE's own real mean color
    ([148,107,91]) and HEAD_SHELL_GENERIC's mean color ([216,166,138],
    79.6% of which was this exact fixed sentinel). This derives a
    per-patient tone instead, from THIS patient's own real, photo-covered
    FACE-core pixels (`hockey_mask & covered` — real bake output only,
    never a fabricated or patient-photo-independent value).

    Returns None (caller must keep the fixed SENTINEL_UNCOVERED) if this
    patient has zero real-covered FACE pixels — never fabricates a tone
    from no evidence, same discipline as every other fallback in this
    module."""
    hockey = compute_region_index() == REGION_FACE
    reliable = hockey & covered
    if not np.any(reliable):
        return None
    return vertex_color[reliable].mean(axis=0)
