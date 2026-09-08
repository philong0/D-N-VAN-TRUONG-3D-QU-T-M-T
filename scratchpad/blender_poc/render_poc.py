"""Blender POC -- run via: blender -b --python render_poc.py
Imports the real production GLB (patient_mesh.glb, exported from a fresh,
unmodified /fit-multiview-atlas call), sets camera framing using the SAME
math Canvas3D.tsx uses (facial-centroid target, computeSafeFitDistance
formula, DEFAULT_ZOOM_MARGIN=1.2, fov=35), renders 2 material profiles x 4
angles with Cycles. Does not modify any project file."""
import bpy, json, math, time, sys

BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"

with open(f"{BPOC}/scene_meta.json") as f:
    meta = json.load(f)
center = meta["center"]  # [x, y, z] -- same GNM coordinate convention as Canvas3D

# ---- clean scene ----
# Environment finding: this sandbox's apt-packaged Blender 3.0.1 (arm64) has
# a broken/nonfunctional OpenImageDenoise + adaptive-sampling combination --
# verified by direct raw-EXR-buffer inspection (uniformly [0,0,0,1]
# regardless of scene content/lighting/camera) -- render output was silently
# ZEROED, not an error. Disabling both fixes it (verified: a simple lit red
# sphere test went from mean=0 to a correctly exposed, visibly lit render
# once both were off). Also: the Cycles CPU device is registered but not
# enabled by default in this environment -- must call get_devices() + set
# d.use=True explicitly, done below.
bpy.ops.wm.read_factory_settings(use_empty=True)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices()
for d in prefs.devices:
    d.use = True
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 64
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 700
scene.render.resolution_y = 850
scene.render.film_transparent = False
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0, 0, 0, 1)
bg.inputs[1].default_value = 0.0
scene.view_settings.view_transform = "Standard"  # NoToneMapping equivalent -- no Filmic curve

# ---- import production mesh ----
t0 = time.time()
bpy.ops.import_scene.gltf(filepath=f"{BPOC}/patient_mesh.glb")
import_time = time.time() - t0
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
total_verts = sum(len(o.data.vertices) for o in mesh_objs)
total_tris = sum(len(o.data.polygons) for o in mesh_objs)  # glTF import is already triangulated
print(f"IMPORT_RESULT objects={len(mesh_objs)} verts={total_verts} tris={total_tris} time={import_time:.2f}s")

# ---- camera: same spherical convention + safe-fit-distance formula as
#      Canvas3D.tsx's computeSafeFitDistance/GnmCameraFit (DEFAULT_ZOOM_MARGIN=1.2, fov=35) ----
FOV_DEG = 35
W, H = scene.render.resolution_x, scene.render.resolution_y
aspect = W / H
fov_rad = math.radians(FOV_DEG)
h_fov = 2 * math.atan(math.tan(fov_rad / 2) * aspect)
limiting_fov = min(fov_rad, h_fov)
MARGIN = 1.2

cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = "FOV"
cam_data.angle = fov_rad
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# real per-mesh bounding-sphere radius (only the actually-rendered, already
# hockey_mask-restricted geometry -- same real data Canvas3D's own
# GnmCameraFit reads from geometry.boundingSphere)
import mathutils
all_pts = []
for o in mesh_objs:
    for v in o.data.vertices:
        all_pts.append(o.matrix_world @ v.co)
cx = sum(p.x for p in all_pts) / len(all_pts)
cy = sum(p.y for p in all_pts) / len(all_pts)
cz = sum(p.z for p in all_pts) / len(all_pts)
radius = max(((p.x-cx)**2 + (p.y-cy)**2 + (p.z-cz)**2) ** 0.5 for p in all_pts)
print(f"MESH_BOUNDS center=({cx:.4f},{cy:.4f},{cz:.4f}) radius={radius:.4f}")

# glTF is Y-up (same convention as Three.js/GLB); Blender's importer auto-
# converts to Blender's own Z-up on import (blender_x=gltf_x, blender_y=
# -gltf_z, blender_z=gltf_y) -- `center` was computed in the RAW GNM/glTF
# frame (from the production API's own fitted_positions), so it must go
# through the same conversion to land on the actually-imported mesh.
target = mathutils.Vector((center[0], -center[2], center[1]))
dist = (radius / math.sin(limiting_fov / 2)) * MARGIN


def set_camera(azimuth_deg, polar_deg):
    # Canvas3D's spherical convention (three.js: Y=up, polar measured from
    # +Y, 90deg=eye-level) with Y and Z swapped to match Blender's own Z-up
    # frame post-gltf-import (same swap already applied to `target` above)
    # -- Z is now the up-driven (cos-polar) component, X/Y form the
    # horizontal plane.
    az = math.radians(azimuth_deg)
    pol = math.radians(polar_deg)
    x = target.x + dist * math.sin(pol) * math.sin(az)
    y = target.y + dist * math.sin(pol) * math.cos(az)
    z = target.z + dist * math.cos(pol)
    cam_obj.location = (x, y, z)
    direction = target - mathutils.Vector((x, y, z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Z").to_euler()


# ---- lights (created per-profile, cleared between profiles) ----
def clear_lights():
    for o in list(scene.objects):
        if o.type == "LIGHT":
            bpy.data.objects.remove(o, do_unlink=True)


def add_light(kind, energy, loc=None, is_sun=False):
    if is_sun:
        ldata = bpy.data.lights.new("sun", type="SUN")
        ldata.energy = energy  # W/m^2 for sun, roughly comparable magnitude to three.js directional intensity after scaling below
        lobj = bpy.data.objects.new("sun", ldata)
        scene.collection.objects.link(lobj)
        lobj.location = loc or (2.5, 4, 3)
        direction = target - mathutils.Vector(lobj.location)
        lobj.rotation_euler = direction.to_track_quat("-Z", "Z").to_euler()
    else:
        ldata = bpy.data.lights.new("world_amb", type="SUN")
        ldata.energy = energy
        lobj = bpy.data.objects.new("world_amb", ldata)
        scene.collection.objects.link(lobj)


def apply_ambient(intensity):
    bg.inputs[1].default_value = intensity


PROFILES = {
    # A -- pre-D34 production material/lighting (the OLD rig, kept here only
    # as a like-for-like reference point against the earlier Three.js "prod"
    # captures -- NOT what's live in Canvas3D today, see report).
    "A_oldProd": {"roughness": 0.38, "metalness": 0.05, "key_energy": 3.0, "ambient": 0.55},
    # B -- Profile E, the material ALREADY merged into production Canvas3D
    # today (D34) -- roughness 0.95/metalness 0/key 0.65/ambient 1.15, no
    # SSAO (Blender has no equivalent pass added here), NoToneMapping (Standard
    # view transform above).
    "B_profileE": {"roughness": 0.95, "metalness": 0.0, "key_energy": 1.0, "ambient": 1.6},
}

# assign material roughness/metalness to every imported mesh's materials
def apply_material_params(roughness, metalness):
    for o in mesh_objs:
        for slot in o.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Roughness"].default_value = roughness
                    node.inputs["Metallic"].default_value = metalness


ANGLES = [("0", 0, 90), ("45", 45, 90), ("90", 90, 90), ("below", 0, 145)]

render_times = {}
for prof_name, p in PROFILES.items():
    clear_lights()
    add_light("sun", p["key_energy"], loc=(2.5, 4, 3), is_sun=True)
    apply_ambient(p["ambient"])
    apply_material_params(p["roughness"], p["metalness"])

    for label, az, pol in ANGLES:
        set_camera(az, pol)
        scene.render.filepath = f"{BPOC}/{label}_{'A' if prof_name.startswith('A') else 'B'}.png"
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        render_times[f"{prof_name}_{label}"] = dt
        print(f"RENDER_DONE {prof_name} {label} time={dt:.2f}s")

with open(f"{BPOC}/render_times.json", "w") as f:
    json.dump(render_times, f, indent=2)

print("ALL_RENDERS_DONE")
