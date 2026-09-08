import bpy, json, math, mathutils, sys

BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"
with open(f"{BPOC}/scene_meta.json") as f:
    meta = json.load(f)
center = meta["center"]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.render.resolution_x = 350
scene.render.resolution_y = 425
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0, 0, 0, 1)
bg.inputs[1].default_value = 3.0
scene.view_settings.view_transform = "Standard"

bpy.ops.import_scene.gltf(filepath=f"{BPOC}/patient_mesh.glb")
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = "FOV"
cam_data.angle = math.radians(35)
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

target = mathutils.Vector((center[0], -center[2], center[1]))
dist = 0.5
cam_obj.location = (target.x, target.y, target.z + dist)
direction = target - cam_obj.location
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
print("CAM_LOC", cam_obj.location, "TARGET", target)

sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = 5.0
sun_obj = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.location = (2.5, 4, 3)
d2 = target - mathutils.Vector(sun_obj.location)
sun_obj.rotation_euler = d2.to_track_quat("-Z", "Y").to_euler()

scene.render.filepath = f"{BPOC}/debug_test.png"
bpy.ops.render.render(write_still=True)
print("DEBUG_RENDER_DONE")
