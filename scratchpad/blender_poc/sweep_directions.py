import bpy, json, math, mathutils, sys
BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"
with open(f"{BPOC}/scene_meta.json") as f:
    meta = json.load(f)
center = meta["center"]

bpy.ops.wm.read_factory_settings(use_empty=True)
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.get_devices()
for d in prefs.devices: d.use = True
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.use_denoising = False
scene.cycles.use_adaptive_sampling = False
scene.render.resolution_x = 120
scene.render.resolution_y = 140
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.3,0.3,0.3,1)
bg.inputs[1].default_value = 5.0
scene.view_settings.view_transform = "Standard"

bpy.ops.import_scene.gltf(filepath=f"{BPOC}/patient_mesh.glb")
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
all_pts = [o.matrix_world @ v.co for o in mesh_objs for v in o.data.vertices]
cx = sum(p.x for p in all_pts)/len(all_pts); cy = sum(p.y for p in all_pts)/len(all_pts); cz = sum(p.z for p in all_pts)/len(all_pts)
radius = max(((p.x-cx)**2+(p.y-cy)**2+(p.z-cz)**2)**0.5 for p in all_pts)
target = mathutils.Vector((center[0], -center[2], center[1]))
dist = radius * 3.5  # generous distance, wide-ish fov below

cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = "FOV"
cam_data.angle = math.radians(60)
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

sun = bpy.data.lights.new("s", type="SUN"); sun.energy=2
sobj = bpy.data.objects.new("s", sun); scene.collection.objects.link(sobj)
sobj.location=(target.x, target.y-1, target.z+1)
sobj.rotation_euler = (target - mathutils.Vector(sobj.location)).to_track_quat("-Z","Y").to_euler()

import numpy as np
dirs = {
    "+X": mathutils.Vector((1,0,0)), "-X": mathutils.Vector((-1,0,0)),
    "+Y": mathutils.Vector((0,1,0)), "-Y": mathutils.Vector((0,-1,0)),
    "+Z": mathutils.Vector((0,0,1)), "-Z": mathutils.Vector((0,0,-1)),
}
results = {}
for name, d in dirs.items():
    cam_obj.location = target + d*dist
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z","Y").to_euler()
    scene.render.filepath = f"{BPOC}/sweep_{name}.png"
    bpy.ops.render.render(write_still=True)
    print(f"SWEEP_DONE {name}")
