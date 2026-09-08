import sys, os, glob
sys.path.insert(0, "/home/ubuntu/dr-vantruong-3d-studio/ai-engine")
os.chdir("/home/ubuntu/dr-vantruong-3d-studio")
import numpy as np
import cv2

import gnm_identity_fit as gif
from gnm_identity_fit import evaluate_basis, evaluate_points, solve_epnp, backproject_pseudo3d, solve_procrustes, apply_rigid
from gnm_correspondence import WFLW_INDICES, VERTEX_INDICES, WEIGHTS, point_mask
from detect_pose import detect_pose

gif._ensure_loaded()
TEMPLATE = gif._TEMPLATE_POSITIONS
BASIS = gif._IDENTITY_BASIS
identity_dim = BASIS.shape[0]
REG = 1e-5

PID = "35e1ffef-d7e5-405d-a7ec-dae152bf883b"
PHOTO_DIR = f".data/patients/{PID}/photos"

template_landmarks_full = evaluate_points(TEMPLATE)
landmark_basis_full = evaluate_basis()

views = []
for slot in ["angle1", "angle2", "angle3", "angle4"]:
    matches = glob.glob(f"{PHOTO_DIR}/{slot}.*")
    if not matches:
        continue
    img = cv2.imread(matches[0])
    is_profile = (slot == "angle3")
    det = detect_pose(img, is_profile_view=is_profile)
    if not det["has_face"] or det["camera_pose"] is None:
        print(f"{slot}: skipped"); continue
    cp = det["camera_pose"]
    fx = cp["focal_length"]; cx, cy = cp["principal_point"]
    cam = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    mask = np.array(point_mask(is_profile), dtype=bool)
    kept_wflw = [wf for wf, keep in zip(WFLW_INDICES, mask) if keep]
    points_2d = np.array([det["landmarks_98"][i] for i in kept_wflw], dtype=np.float64)
    template_masked = template_landmarks_full[mask]
    pose = solve_epnp(template_masked, points_2d, cam)
    if pose is None:
        print(f"{slot}: EPNP failed"); continue
    Rp, tp = pose
    observed = backproject_pseudo3d(points_2d, template_masked, Rp, tp, cam)
    Rproc, sproc, tproc = solve_procrustes(observed, template_masked)
    aligned = apply_rigid(observed, Rproc, sproc, tproc)
    offset = aligned - template_masked
    views.append(dict(slot=slot, mask=mask, offset=offset, Rp=Rp, tp=tp, cam=cam,
                       landmarks_98=np.array(det["landmarks_98"]), img_shape=img.shape))
    print(f"{slot}: real pose reproj err = {cp['mean_reprojection_error_px']:.2f}px")

np.save("scratchpad/jaw_fix/views_cache.npy", np.array(views, dtype=object), allow_pickle=True)


def solve_ridge(jaw_weight):
    weights = np.array([1.0] * 7 + [10.0] + [1.0] * 20 + [1.0] * 10 + [1.0] * 10)
    weights[0:7] = jaw_weight
    M = np.zeros((identity_dim, identity_dim))
    rhs = np.zeros(identity_dim)
    for v in views:
        B_v = landmark_basis_full[:, v["mask"], :].reshape(identity_dim, -1)
        w_ = np.repeat(weights[v["mask"]], 3)
        Bw = B_v * w_[None, :]
        M += Bw @ B_v.T
        rhs += Bw @ v["offset"].reshape(-1)
    M += REG * np.eye(identity_dim)
    coeff = np.linalg.solve(M, rhs)
    fitted = TEMPLATE + np.einsum("d,dvc->vc", coeff, BASIS)
    return fitted, coeff


def measure_face_width_ratio(fitted):
    """Reproject the two outermost jaw/cheek WFLW points (0 and 32) through
    the REAL frontal-view pose and compare against the REAL detected 2D
    pixel distance for those same points -- a camera-consistent, real
    measurement of how much wider/narrower the fit is than the actual photo."""
    frontal = next(v for v in views if v["slot"] == "angle1")
    row0 = WFLW_INDICES.index(0)
    row32 = WFLW_INDICES.index(32)
    p0_3d = sum(w * fitted[vid] for vid, w in zip(VERTEX_INDICES[row0], WEIGHTS[row0]))
    p32_3d = sum(w * fitted[vid] for vid, w in zip(VERTEX_INDICES[row32], WEIGHTS[row32]))
    rvec, _ = cv2.Rodrigues(frontal["Rp"])
    proj, _ = cv2.projectPoints(np.array([p0_3d, p32_3d]), rvec, frontal["tp"], frontal["cam"], np.zeros((4, 1)))
    proj = proj.reshape(2, 2)
    fitted_width_px = np.linalg.norm(proj[0] - proj[1])
    real_width_px = np.linalg.norm(frontal["landmarks_98"][0] - frontal["landmarks_98"][32])
    return fitted_width_px, real_width_px, fitted_width_px / real_width_px


print("\n=== jaw weight sweep: fitted/real face-width ratio (frontal view) ===")
for jw in [1.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]:
    fitted, coeff = solve_ridge(jw)
    fw, rw, ratio = measure_face_width_ratio(fitted)
    maxcoeff = float(np.abs(coeff).max())
    print(f"jaw_weight={jw:5.1f}  fitted_width_px={fw:6.1f}  real_width_px={rw:6.1f}  ratio={ratio:.3f}  maxCoeff={maxcoeff:.2f}")
    np.save(f"scratchpad/jaw_fix/fitted_jw{int(jw)}.npy", fitted)
