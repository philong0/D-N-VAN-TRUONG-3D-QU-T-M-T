import bpy, math, mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"


scene.render.resolution_x = 300
scene.render.resolution_y = 300
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1)
bg.inputs[1].default_value = 1.0

target = mathutils.Vector((0, 0, 0))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=target)
sphere = bpy.context.active_object
mat = bpy.data.materials.new("redmat")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1)
sphere.data.materials.append(mat)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = "FOV"
cam_data.angle = math.radians(35)
cam_obj = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -0.5, 0)
direction = target - mathutils.Vector(cam_obj.location)
cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
print("CAM_LOC", cam_obj.location)

sun_data = bpy.data.lights.new("sun", type="SUN")
sun_data.energy = 3.0
sun_obj = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.location = (1, -1, 1)
d2 = target - mathutils.Vector(sun_obj.location)
sun_obj.rotation_euler = d2.to_track_quat("-Z", "Y").to_euler()

scene.render.filepath = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc/debug_sphere.png"
bpy.ops.render.render(write_still=True)
print("DONE")
