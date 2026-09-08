"""Render exactly ONE angle/profile combo per Blender process invocation --
avoids the accumulating per-frame slowdown seen when rendering all 8 in one
session (5-134s and climbing) on this sandbox's Cycles build. Usage:
blender -b --python render_one.py -- <label> <azimuth> <polar> <profile>"""
import bpy, json, math, mathutils, sys, time

BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"
argv = sys.argv[sys.argv.index("--") + 1:]
label, az_deg, pol_deg, prof_name = argv[0], float(argv[1]), float(argv[2]), argv[3]

with open(f"{BPOC}/scene_meta.json") as f:
    meta = json.load(f)
center = meta["center"]

bpy.ops.wm.read_factory_settings(use_empty=True)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices()
for d in prefs.devices:
    d.use = True

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 350
scene.render.resolution_y = 420
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0, 0, 0, 1)
bg.inputs[1].default_value = 0.0
scene.view_settings.view_transform = "Standard"

bpy.ops.import_scene.gltf(filepath=f"{BPOC}/patient_mesh.glb")
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

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

all_pts = [o.matrix_world @ v.co for o in mesh_objs for v in o.data.vertices]
cx = sum(p.x for p in all_pts) / len(all_pts)
cy = sum(p.y for p in all_pts) / len(all_pts)
cz = sum(p.z for p in all_pts) / len(all_pts)
radius = max(((p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2) ** 0.5 for p in all_pts)

target = mathutils.Vector((center[0], -center[2], center[1]))
dist = (radius / math.sin(limiting_fov / 2)) * MARGIN

# The world-axis spherical formula (sin/cos against raw X/Y/Z) was fighting
# an ambiguous axis convention across two coordinate transforms (glTF Y-up
# -> Blender Z-up, then which world axis is "forward" for THIS mesh) and
# produced wrong framing twice. Fixed properly: build the camera basis from
# the mesh's OWN measured forward direction -- the "nose" object's real
# average vertex normal, which by construction points straight out of the
# face -- instead of assuming a world axis is "forward".
nose_obj = next(o for o in mesh_objs if o.name == "nose")
normals = [nose_obj.matrix_world.to_3x3() @ v.normal for v in nose_obj.data.vertices]
fwd = mathutils.Vector((0, 0, 0))
for n in normals:
    fwd += n
fwd.normalize()
world_up = mathutils.Vector((0, 0, 1))
right = fwd.cross(world_up)
right.normalize()
up = right.cross(fwd)
up.normalize()

az = math.radians(az_deg)
pol = math.radians(pol_deg)
# pol=90 (eye-level): pure fwd/right orbit, no up component. pol>90 ("below"):
# camera drops toward -up (below target) while looking back up at it.
offset = (math.sin(pol) * math.sin(az) * right) + (math.sin(pol) * math.cos(az) * fwd) + (math.cos(pol) * up)
cam_loc = target + offset * dist
cam_obj.location = cam_loc
direction = target - cam_loc
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
print(f"CAM_DEBUG fwd={fwd} right={right} up={up} cam_loc={cam_loc}")

PROFILES = {
    "A_oldProd": {"roughness": 0.38, "metalness": 0.05, "key_energy": 3.0, "ambient": 0.55},
    "B_profileE": {"roughness": 0.95, "metalness": 0.0, "key_energy": 1.0, "ambient": 1.6},
}
p = PROFILES[prof_name]

sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = p["key_energy"]
sun_obj = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.location = (2.5, 4, 3)
d2 = target - mathutils.Vector(sun_obj.location)
sun_obj.rotation_euler = d2.to_track_quat("-Z", "Y").to_euler()
bg.inputs[1].default_value = 8.0

for o in mesh_objs:
    for slot in o.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Roughness"].default_value = p["roughness"]
                node.inputs["Metallic"].default_value = p["metalness"]

suffix = "A" if prof_name.startswith("A") else "B"
scene.render.filepath = f"{BPOC}/{label}_{suffix}.png"
t0 = time.time()
bpy.ops.render.render(write_still=True)
dt = time.time() - t0
print(f"RENDER_ONE_DONE {prof_name} {label} time={dt:.2f}s")
