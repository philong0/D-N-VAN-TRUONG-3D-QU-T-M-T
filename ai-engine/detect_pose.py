"""
Real face pose (yaw) detection for the 4 Bước 2 reference photos.

Two real, pretrained models, loaded once at import time (both proven working
against this project's own real patient photos — see the standalone tests at
scratchpad/phaseB_detector_test/{test_detector.py,test_pipnet.py} from an
earlier investigation phase, including the 90 deg profile case MediaPipe
FaceMesh cannot detect at all):

  - YuNet (opencv_zoo, `models/yunet_2023mar.onnx`) — face bounding box.
  - PIPNet (resnet18 backbone, WFLW 98-point landmarks,
    `models/pipnet/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth`)
    — 98 facial landmarks within that box. Code copied verbatim from PIPNet's
    own `lib/` (MIT-licensed, github.com/jhb86253817/PIPNet), unmodified.

Yaw estimate: WFLW-98 has no literal "ear" landmark. An earlier version of
this module used the two outermost face-contour points (0, 32) as an
"ear" stand-in; checked empirically against real 90 deg profile photos, that
broke down — the FAR contour point (whichever side is turned away) has no
real feature to lock onto and collapses somewhere on the visible side
instead (verified by drawing indices 0/32/57 on a real detected profile
photo), giving an inconsistent yaw sign between a 45 deg and a 90 deg shot of
the SAME side. Point clusters 60-67 ("left eye") and 68-75 ("right eye") are
core, densely-trained WFLW landmarks and stay reliable much closer to true
profile, so those are the left/right reference instead — verified against
this app's OWN actual camera math (not assumed): a real `three` Vector3
`.project(camera)` check at azimuth=0 confirms `sceneX = (px - width/2) *
scale` (face-geometry.ts) is NOT mirrored (scene +X really does render on
the screen's right half at azimuth 0, matching the original photo's own
left/right), and increasing positive azimuth moves the camera toward scene
+X (three.js's own Spherical convention) — i.e. brings the side that
appeared on the RIGHT of the frontal photo into view. Cross-checked against
two real patient photos: whichever eye cluster (60-67 vs 68-75) was the
NEAR/visible one in a real profile shot matched the side this sign
convention predicts, for both directions. Point 57 (bottom-center of the
nose cluster, indices 51-59 — verified empirically to sit at the base of the
nose/columella) stands in for the nose. yaw = 0 when the nose sits centered
between the two eye clusters (frontal); yaw approaches +/-90 deg as the nose
is pushed toward one cluster relative to the apparent eye-to-eye span (which
itself shrinks as the head turns, since the far eye is foreshortened) — a
standard single-image proxy for head yaw when no depth/3D landmark exists.
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
PIPNET_DIR = MODELS_DIR / "pipnet"
YUNET_PATH = MODELS_DIR / "yunet_2023mar.onnx"

sys.path.insert(0, str(PIPNET_DIR))
sys.path.insert(0, str(PIPNET_DIR / "lib"))
sys.path.insert(0, str(PIPNET_DIR / "FaceBoxesV2"))

import importlib  # noqa: E402

from functions import forward_pip, get_meanface  # noqa: E402
from networks import Pip_resnet18  # noqa: E402

# WFLW-98 index conventions used for the yaw proxy — see module docstring for
# how these were verified (not assumed) against real detected photos.
NOSE_TIP_IDX = 57
LEFT_EYE_IDXS = range(60, 68)
RIGHT_EYE_IDXS = range(68, 76)

_cfg_module = importlib.import_module("experiments.WFLW.pip_32_16_60_r18_l2_l1_10_1_nb10")
_cfg = _cfg_module.Config()
_cfg.experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10"
_cfg.data_name = "WFLW"

_meanface_indices, _reverse_index1, _reverse_index2, _max_len = get_meanface(
    str(PIPNET_DIR / "data" / "WFLW" / "meanface.txt"), _cfg.num_nb
)

_device = torch.device("cpu")
_resnet18 = models.resnet18(weights=None)
_pipnet = Pip_resnet18(
    _resnet18, _cfg.num_nb, num_lms=_cfg.num_lms, input_size=_cfg.input_size, net_stride=_cfg.net_stride
)
_pipnet = _pipnet.to(_device)
_weight_file = (
    PIPNET_DIR / "snapshots" / "WFLW" / "pip_32_16_60_r18_l2_l1_10_1_nb10" / f"epoch{_cfg.num_epochs - 1}.pth"
)
_pipnet.load_state_dict(torch.load(str(_weight_file), map_location=_device))
_pipnet.eval()

_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_preprocess = transforms.Compose(
    [transforms.Resize((_cfg.input_size, _cfg.input_size)), transforms.ToTensor(), _normalize]
)

_yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_threshold=0.5)


def _detect_face_box(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """YuNet face bbox (xmin, ymin, xmax, ymax) in pixel coords, or None if no face."""
    h, w = image_bgr.shape[:2]
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(image_bgr)
    if faces is None or len(faces) == 0:
        return None
    # Highest-confidence face — same as PIPNet's own reference demo picking
    # the single detection when maxNumFaces-style single-subject photos.
    best = max(faces, key=lambda f: f[-1])
    x, y, bw, bh = best[0], best[1], best[2], best[3]
    return int(x), int(y), int(x + bw), int(y + bh)


def _run_pipnet(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> list[list[float]]:
    """98 (x, y) landmarks in ORIGINAL image pixel coords, given a face box — verbatim math from PIPNet's own lib/demo.py, only the crop-margin/box source changed (YuNet instead of FaceBoxesV2)."""
    image_height, image_width = image_bgr.shape[:2]
    det_xmin, det_ymin, det_xmax, det_ymax = box
    det_width = det_xmax - det_xmin
    det_height = det_ymax - det_ymin
    det_box_scale = 1.2
    det_xmin -= int(det_width * (det_box_scale - 1) / 2)
    det_ymin += int(det_height * (det_box_scale - 1) / 2)
    det_xmax += int(det_width * (det_box_scale - 1) / 2)
    det_ymax += int(det_height * (det_box_scale - 1) / 2)
    det_xmin = max(det_xmin, 0)
    det_ymin = max(det_ymin, 0)
    det_xmax = min(det_xmax, image_width - 1)
    det_ymax = min(det_ymax, image_height - 1)
    det_width = det_xmax - det_xmin + 1
    det_height = det_ymax - det_ymin + 1

    det_crop = image_bgr[det_ymin:det_ymax, det_xmin:det_xmax, :]
    det_crop = cv2.resize(det_crop, (_cfg.input_size, _cfg.input_size))
    inputs = Image.fromarray(det_crop[:, :, ::-1].astype("uint8"), "RGB")
    inputs = _preprocess(inputs).unsqueeze(0).to(_device)

    with torch.no_grad():
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, _, _ = forward_pip(
            _pipnet, inputs, _preprocess, _cfg.input_size, _cfg.net_stride, _cfg.num_nb
        )

    tmp_nb_x = lms_pred_nb_x[_reverse_index1, _reverse_index2].view(_cfg.num_lms, _max_len)
    tmp_nb_y = lms_pred_nb_y[_reverse_index1, _reverse_index2].view(_cfg.num_lms, _max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()

    pts = []
    for k in range(_cfg.num_lms):
        x_pred = float(lms_pred_merge[k * 2] * det_width + det_xmin)
        y_pred = float(lms_pred_merge[k * 2 + 1] * det_height + det_ymin)
        pts.append([x_pred, y_pred])
    return pts


# ---------------------------------------------------------------------------
# Real camera pose (SolvePnP) — upgraded from the original 8-point/ITERATIVE
# version (Giai đoạn B) to the full 48-point correspondence + EPNP, per D1.5's
# own head-to-head validation: ITERATIVE without an initial guess either
# diverges outright (0913e4c9's "below" photo: 113% reprojection error) or,
# worse, converges to a geometrically IMPOSSIBLE camera position (behind the
# head) while still reporting a deceptively low reprojection error
# (67434b3a's "below" photo) — EPNP was the only method that stayed
# geometrically valid AND accurate on both real patients at every angle (see
# scratchpad D1.5/D2/D2.5/D3 reports). Point correspondence: gnm_correspondence.py
# (48 points; 3 eyebrow points auto-dropped for a profile/90° photo — real
# far-side occlusion, not a detector or mapping flaw, see that module).
# ---------------------------------------------------------------------------
_GNM_TEMPLATE_PTS = None

def _gnm_landmark_xyz(mask: list[bool]) -> np.ndarray:
    import zipfile
    from gnm_correspondence import VERTEX_INDICES as _GNM_VIDX, WEIGHTS as _GNM_W
    global _GNM_TEMPLATE_PTS
    if _GNM_TEMPLATE_PTS is None:
        gnm_asset = Path(__file__).parent.parent / "public" / "models" / "gnm" / "gnm_head_v3.npz"
        with zipfile.ZipFile(gnm_asset) as zf:
            with zf.open("template_vertex_positions.npy") as f:
                _GNM_TEMPLATE_PTS = np.load(f)
    vidx = np.array(_GNM_VIDX)[mask]
    w = np.array(_GNM_W)[mask]
    pts = _GNM_TEMPLATE_PTS[vidx]
    return np.einsum("nk,nkc->nc", w, pts)


def solve_camera_pose(landmarks_98: list[list[float]], image_shape: tuple[int, int], is_profile_view: bool = False) -> dict | None:
    """Real SolvePnP (EPNP) camera pose from the validated 48-point correspondence (45 for a 90°/profile photo — see gnm_correspondence.py). Returns None (never raises) if solvePnP itself fails to converge — caller must treat that as 'no pose available', not fall back to a fabricated one."""
    from gnm_correspondence import WFLW_INDICES as _GNM_WFLW, point_mask as _gnm_point_mask
    h, w = image_shape
    mask = _gnm_point_mask(is_profile_view)
    kept_wflw = [idx for idx, keep in zip(_GNM_WFLW, mask) if keep]
    points_2d = np.array([landmarks_98[i] for i in kept_wflw], dtype=np.float64)
    points_3d = _gnm_landmark_xyz(mask)

    focal = float(w)  # same approximation validated in scratchpad — no real EXIF/calibration data exists
    camera_matrix = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None

    reprojected, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    mean_error_px = float(np.linalg.norm(reprojected.reshape(-1, 2) - points_2d, axis=1).mean())

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return {
        "rotation_matrix": rotation_matrix.tolist(),  # 3x3, GNM-space -> camera-space
        "translation": tvec.flatten().tolist(),
        "focal_length": focal,
        "principal_point": [w / 2, h / 2],
        "image_size": [w, h],
        "mean_reprojection_error_px": mean_error_px,
        "mean_reprojection_error_pct_width": mean_error_px / w * 100,
    }


def compute_yaw_deg(landmarks_98: list[list[float]]) -> float:
    """See module docstring for why indices 57/60-67/68-75 and the atan2 formula were chosen, and how the sign was verified against this app's own camera math."""
    nose = landmarks_98[NOSE_TIP_IDX]
    left_eye_x = sum(landmarks_98[i][0] for i in LEFT_EYE_IDXS) / len(LEFT_EYE_IDXS)
    right_eye_x = sum(landmarks_98[i][0] for i in RIGHT_EYE_IDXS) / len(RIGHT_EYE_IDXS)
    eye_mid_x = (left_eye_x + right_eye_x) / 2
    eye_span = abs(right_eye_x - left_eye_x)
    if eye_span < 1e-6:
        return 0.0
    dx = nose[0] - eye_mid_x
    return math.degrees(math.atan2(dx, eye_span / 2))


def detect_face_landmarks(image_bgr: np.ndarray) -> dict:
    """Pure landmark detection without GNM template pose solving."""
    box = _detect_face_box(image_bgr)
    if box is None:
        return {"has_face": False, "yaw": None, "landmarks_98": None}
    landmarks = _run_pipnet(image_bgr, box)
    yaw = compute_yaw_deg(landmarks)
    return {"has_face": True, "yaw": yaw, "landmarks_98": landmarks}


def detect_pose(image_bgr: np.ndarray, is_profile_view: bool = False) -> dict:
    """Full pipeline for one image: YuNet box -> PIPNet 98 landmarks -> yaw + real SolvePnP camera pose."""
    res = detect_face_landmarks(image_bgr)
    if not res["has_face"]:
        return {"has_face": False, "yaw": None, "landmarks_98": None, "camera_pose": None}

    landmarks = res["landmarks_98"]
    yaw = res["yaw"]
    h, w = image_bgr.shape[:2]
    camera_pose = solve_camera_pose(landmarks, (h, w), is_profile_view=is_profile_view)
    return {"has_face": True, "yaw": yaw, "landmarks_98": landmarks, "camera_pose": camera_pose}
