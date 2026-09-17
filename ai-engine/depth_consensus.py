"""Conservative correction of measured ARFace geometry from independent depths.

Regularizes the displacement field, never the patient's underlying face shape.
Unsupported vertices retain the original measured ARFace geometry exactly.
"""
import hashlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def consensus_displacement(normals, faces, samples, sample_weights):
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(sample_weights, dtype=float)
    valid = np.isfinite(samples) & (weights > 0)
    # Vertex-wise robust center without warning on never-observed vertices.
    center = np.zeros(samples.shape[1])
    for i in np.flatnonzero(valid.any(axis=0)):
        center[i] = np.median(samples[valid[:, i], i])
    inliers = valid & (np.abs(samples - center) <= .003)
    support = inliers.sum(axis=0)
    good = support >= 2
    w = np.where(inliers, weights, 0.)
    strength = np.minimum(w.sum(axis=0), 4.) * good
    target = np.divide(np.sum(np.where(inliers, samples, 0.) * w, axis=0),
                       w.sum(axis=0), out=np.zeros_like(center), where=w.sum(axis=0) > 0)
    # Refuse large inferred corrections; raw sensor samples remain untouched.
    good &= np.abs(target) <= .010
    strength *= good
    n = len(center)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = edges.T
    adjacency = sparse.coo_matrix((np.ones(2*len(edges)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    ids = np.flatnonzero(good)
    displacement = np.zeros(n)
    if len(ids):
        # Dirichlet boundary at unsupported vertices; prior anchors all supported
        # corrections. No smoothing/replacement of ARFace vertices themselves.
        system = sparse.diags(1. + strength[ids]) + 2. * laplacian[ids][:, ids]
        displacement[ids] = spsolve(system, (strength * target)[ids])
    return normals * displacement[:, None], {
        'method': 'independent-depth-normal-consensus',
        'minimumIndependentSupport': 2,
        'consensusToleranceMm': 3.,
        'supportedVertices': int(good.sum()),
        'unchangedVertices': int((~good).sum()),
        'maxCorrectionMm': float(np.max(np.abs(displacement))*1000),
    }


def independent_depth_frames(frames):
    seen = set()
    result = []
    for frame in frames:
        depth = np.ascontiguousarray(frame['depth_f32'], dtype='<f4')
        identity = (depth.shape, hashlib.sha256(depth.tobytes()).digest())
        if identity not in seen:
            seen.add(identity)
            result.append(frame)
    return result
