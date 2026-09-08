import json, zipfile
import numpy as np
import cv2

d = json.load(open("scratchpad/runtime_geometry_export.json"))
positions = np.array(d["position"], dtype=np.float64).reshape(-1, 3)
index = np.array(d["index"], dtype=np.int64)
triangles = index.reshape(-1, 3)
n_verts_real = positions.shape[0]

with zipfile.ZipFile("public/models/gnm/gnm_head_v3.npz") as zf:
    with zf.open("vertex_groups.npy") as f:
        groups_arr = np.load(f)

def mask(row):
    m = groups_arr[row] > 0.5
    if len(m) < n_verts_real:
        m = np.concatenate([m, np.zeros(n_verts_real - len(m), dtype=bool)])
    return m[:n_verts_real]

SCLERA, IRIS, PUPIL, LEFT_EYE, RIGHT_EYE, EYE_SOCKETS = 19, 20, 21, 14, 15, 16
sclera, iris, pupil = mask(SCLERA), mask(IRIS), mask(PUPIL)
left_eye, right_eye, socket = mask(LEFT_EYE), mask(RIGHT_EYE), mask(EYE_SOCKETS)

# per-vertex debug color, solid (geometry-only test, no photo color at all)
DEBUG = np.zeros((n_verts_real, 3))
DEBUG[socket] = [90, 60, 40]       # eyelid/socket skin = brown
DEBUG[sclera] = [255, 255, 255]    # sclera = pure white
DEBUG[iris] = [30, 90, 200]        # iris = blue
DEBUG[pupil] = [0, 0, 0]           # pupil = black
# tint left vs right for clarity
is_left = left_eye
is_right = right_eye
DEBUG_L = DEBUG.copy()
DEBUG_R = DEBUG.copy()

W, H, FOV_DEG = 900, 1000, 35.0
centroid = positions.mean(axis=0)
radius = np.linalg.norm(positions - centroid, axis=1).max()
fov_rad = np.radians(FOV_DEG)
dist = (radius / np.sin(fov_rad / 2)) * 1.25
eye = centroid + np.array([0.0, 0.02, dist])
forward = centroid - eye; forward /= np.linalg.norm(forward)
right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
true_up = np.cross(right, forward)
rel = positions - eye[None,:]
cx, cy, cz = rel@right, rel@true_up, rel@forward
focal = (H/2)/np.tan(fov_rad/2)
px = focal*cx/np.clip(cz,1e-6,None) + W/2
py = -focal*cy/np.clip(cz,1e-6,None) + H/2

img = np.zeros((H,W,3))
depth = np.full((H,W), np.inf)
eye_tri_mask = (sclera[triangles] | iris[triangles] | pupil[triangles] | socket[triangles]).any(axis=1)
tri_cz = cz[triangles]
valid = eye_tri_mask & (tri_cz > 0.01).all(axis=1)

for ti in np.nonzero(valid)[0]:
    a,b,c = triangles[ti]
    pu=np.array([px[a],px[b],px[c]]); pv=np.array([py[a],py[b],py[c]]); pz=np.array([cz[a],cz[b],cz[c]])
    x_min=max(int(np.floor(pu.min())),0); x_max=min(int(np.ceil(pu.max())),W-1)
    y_min=max(int(np.floor(pv.min())),0); y_max=min(int(np.ceil(pv.max())),H-1)
    if x_min>x_max or y_min>y_max: continue
    xs,ys=np.meshgrid(np.arange(x_min,x_max+1)+0.5, np.arange(y_min,y_max+1)+0.5)
    x0,y0,x1,y1,x2,y2=pu[0],pv[0],pu[1],pv[1],pu[2],pv[2]
    denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
    if abs(denom)<1e-9: continue
    w0=((y1-y2)*(xs-x2)+(x2-x1)*(ys-y2))/denom
    w1=((y2-y0)*(xs-x2)+(x0-x2)*(ys-y2))/denom
    w2=1.0-w0-w1
    inside=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)
    if not inside.any(): continue
    interp_z=w0*pz[0]+w1*pz[1]+w2*pz[2]
    region=depth[y_min:y_max+1,x_min:x_max+1]
    update=inside&(interp_z<region)
    if not update.any(): continue
    region[update]=interp_z[update]
    col = w0[...,None]*DEBUG[a] + w1[...,None]*DEBUG[b] + w2[...,None]*DEBUG[c]
    img[y_min:y_max+1,x_min:x_max+1][update] = col[update]

out = np.clip(img,0,255).astype(np.uint8)
cv2.imwrite("scratchpad/test_eye_parts_geometry_debug.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print("saved. left_eye sclera verts:", int((sclera&left_eye).sum()), "right_eye sclera verts:", int((sclera&right_eye).sum()))
