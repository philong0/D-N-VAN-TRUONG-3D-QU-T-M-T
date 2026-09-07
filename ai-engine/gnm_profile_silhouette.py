"""
gnm_profile_silhouette.py

Silhouette-Guided Profile Deformation for GNM Head v3.
Reconstructs authentic patient side profile geometry (nose bridge, tip projection,
nasolabial angle, labial projection, pogonion/chin projection) from 90° profile photos
and multi-view landmarks, replacing generic template depth borrowing.
"""

from pathlib import Path
import cv2
import numpy as np

# Anatomical Profile Midline Anchors in GNM Head v3 (17,821 vertices)
# Mapped to specific anatomical landmark zones
PROFILE_MIDLINE_VERTICES = {
    "glabella": 12321,      # Glabella / Nasion root (y ~ 0.302)
    "nasal_bridge_upper": 12326, # Upper dorsal bridge (y ~ 0.295)
    "nasal_bridge_mid": 12315,   # Rhinion / Mid dorsal bridge (y ~ 0.291)
    "nasal_bridge_lower": 12311, # Supratip (y ~ 0.284)
    "pronasale_tip": 12308,      # Pronasale / Nose Tip (y ~ 0.274, apex Z)
    "infratip": 12290,           # Infratip lobule (y ~ 0.260)
    "columella": 12301,          # Columella / Subnasale (y ~ 0.258)
    "subnasale": 12299,          # Subnasale junction (y ~ 0.256)
    "philtrum": 12291,           # Philtrum column (y ~ 0.255)
    "labrale_superius": 12285,   # Upper Lip (y ~ 0.235)
    "stomion": 12274,            # Mouth cleft (y ~ 0.247)
    "labrale_inferius": 12270,   # Lower Lip (y ~ 0.223)
    "supramentale": 12262,       # Mentolabial sulcus (y ~ 0.211)
    "pogonion_chin": 12258,      # Pogonion / Chin Apex (y ~ 0.192)
    "menton": 12342,             # Menton / Submental base (y ~ 0.182)
}

# Corresponding WFLW-98 Landmark Indices for 90° Profile
PROFILE_LANDMARK_MAP = {
    "glabella": 51,
    "nasal_bridge_upper": 51,
    "nasal_bridge_mid": 52,
    "nasal_bridge_lower": 53,
    "pronasale_tip": 54,
    "infratip": 55,
    "columella": 56,
    "subnasale": 57,
    "philtrum": 57,
    "labrale_superius": 89,
    "stomion": 90,
    "labrale_inferius": 94,
    "supramentale": 16,
    "pogonion_chin": 16,
    "menton": 18,
}


def extract_profile_contour_offsets(
    mesh_positions: np.ndarray,
    profile_image: np.ndarray,
    landmarks_98: np.ndarray,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    K_cam: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Computes 3D displacement vectors (dx, dy, dz) for profile anchor vertices
    by comparing projected 3D landmarks with detected 2D landmarks and edge silhouette.
    """
    h_img, w_img = profile_image.shape[:2]
    fx = K_cam[0, 0]
    fy = K_cam[1, 1]
    cx = K_cam[0, 2]
    cy = K_cam[1, 2]

    offsets = {}

    for zone_name, vid in PROFILE_MIDLINE_VERTICES.items():
        if zone_name not in PROFILE_LANDMARK_MAP:
            continue
        lms_idx = PROFILE_LANDMARK_MAP[zone_name]
        if lms_idx >= len(landmarks_98):
            continue

        p_2d_target = landmarks_98[lms_idx]  # (u, v) in image space
        v_3d = mesh_positions[vid]           # (X, Y, Z) in model space

        # Project 3D vertex to camera space
        v_cam = R_cam @ v_3d + t_cam
        z_c = v_cam[2]
        if z_c <= 0.01:
            continue

        u_proj = fx * (v_cam[0] / z_c) + cx
        v_proj = fy * (v_cam[1] / z_c) + cy

        # Target 2D offset (du, dv)
        du = p_2d_target[0] - u_proj
        dv = p_2d_target[1] - v_proj

        # Check for profile edge boundary along the horizontal line
        v_int = int(np.clip(p_2d_target[1], 0, h_img - 1))
        u_start = int(np.clip(p_2d_target[0] - 25, 0, w_img - 1))
        u_end = int(np.clip(p_2d_target[0] + 25, 0, w_img - 1))

        if u_end > u_start + 10:
            scanline = profile_image[v_int, u_start:u_end]
            gray = cv2.cvtColor(scanline[None, :, :], cv2.COLOR_BGR2GRAY).flatten()
            grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).flatten()
            best_edge_local = int(np.argmax(grad))
            if grad[best_edge_local] > 20:
                refined_u = u_start + best_edge_local
                if abs(refined_u - p_2d_target[0]) <= 12:
                    du = refined_u - u_proj

        # Convert 2D pixel displacement to 3D camera coordinate displacement:
        d_cam = np.array([du * z_c / fx, dv * z_c / fy, 0.0], dtype=np.float64)

        # Transform displacement vector back to model space:
        d_model = R_cam.T @ d_cam

        # Cap maximum displacement for anatomical safety (max 18mm = 0.018m)
        disp_mag = float(np.linalg.norm(d_model))
        if disp_mag > 0.018:
            d_model = d_model * (0.018 / disp_mag)

        offsets[vid] = d_model

    return offsets


def apply_profile_silhouette_deformation(
    mesh_positions: np.ndarray,
    profile_image: np.ndarray,
    landmarks_98: np.ndarray,
    R_cam: np.ndarray,
    t_cam: np.ndarray,
    K_cam: np.ndarray,
    profile_slot: str = "angle3",
) -> tuple[np.ndarray, str]:
    """
    Deforms mesh_positions using smooth radial Gaussian / RBF fields anchored
    at the profile silhouette anchor vertices.
    """
    anchor_offsets = extract_profile_contour_offsets(
        mesh_positions, profile_image, landmarks_98, R_cam, t_cam, K_cam
    )

    if not anchor_offsets:
        return mesh_positions, "profile-silhouette: no valid anchors"

    deformed_positions = mesh_positions.copy()
    N = len(mesh_positions)

    zone_radii = {
        12321: 0.025, # glabella
        12326: 0.022, # nasal bridge upper
        12315: 0.020, # rhinion
        12311: 0.020, # supratip
        12308: 0.022, # pronasale tip
        12290: 0.018, # infratip
        12301: 0.018, # columella
        12299: 0.018, # subnasale
        12291: 0.018, # philtrum
        12285: 0.016, # upper lip
        12274: 0.016, # stomion
        12270: 0.016, # lower lip
        12262: 0.025, # supramentale
        12258: 0.035, # pogonion chin
        12342: 0.030, # menton
    }

    total_disp = np.zeros_like(mesh_positions, dtype=np.float64)
    total_weights = np.zeros(N, dtype=np.float64)

    for vid, offset in anchor_offsets.items():
        anchor_pos = mesh_positions[vid]
        sigma = zone_radii.get(vid, 0.022)

        # Euclidean distance to anchor
        dists = np.linalg.norm(mesh_positions - anchor_pos, axis=1)

        # Gaussian radial falloff
        w = np.exp(-0.5 * (dists / sigma) ** 2)

        # Cutoff at 2.5 sigma for strict locality
        w[dists > (2.5 * sigma)] = 0.0

        # Midline symmetry preservation: dampen displacement for lateral vertices (|X| > 0.045)
        x_damp = np.clip(1.0 - (np.abs(mesh_positions[:, 0]) / 0.045), 0.0, 1.0)
        w = w * x_damp

        total_disp += w[:, None] * offset[None, :]
        total_weights += w

    has_w = total_weights > 1e-4
    effective_disp = np.zeros_like(total_disp)
    effective_disp[has_w] = total_disp[has_w] / np.clip(total_weights[has_w, None], 1.0, None)

    # Apply direct anatomical 3D displacement:
    # effective_disp contains accurate dZ (depth projection) and dY (vertical alignment)
    # in model coordinates computed directly from 2D silhouette landmarks & edge gradients.
    # Midline lateral shift (X) is zeroed to guarantee perfect bilateral symmetry.
    effective_disp[:, 0] = 0.0

    deformed_positions = mesh_positions + effective_disp

    max_shift_mm = float(np.max(np.linalg.norm(deformed_positions - mesh_positions, axis=1)) * 1000.0)
    note = f"profile-silhouette: deformed {int(has_w.sum())} vertices, max_shift={max_shift_mm:.2f}mm"
    print(f"[{profile_slot}] {note}", flush=True)

    return deformed_positions, note
