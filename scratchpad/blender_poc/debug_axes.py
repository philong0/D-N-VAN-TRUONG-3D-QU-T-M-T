import bpy, json, mathutils
BPOC = "/home/ubuntu/dr-vantruong-3d-studio/scratchpad/blender_poc"
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=f"{BPOC}/patient_mesh.glb")
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
pts = [o.matrix_world @ v.co for o in mesh_objs for v in o.data.vertices]
xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
print(f"X range: {min(xs):.4f} to {max(xs):.4f} (span {max(xs)-min(xs):.4f})")
print(f"Y range: {min(ys):.4f} to {max(ys):.4f} (span {max(ys)-min(ys):.4f})")
print(f"Z range: {min(zs):.4f} to {max(zs):.4f} (span {max(zs)-min(zs):.4f})")

with open(f"{BPOC}/scene_meta.json") as f:
    meta = json.load(f)
center = meta["center"]
target = mathutils.Vector((center[0], -center[2], center[1]))
print("target (converted):", target)

# average normal of the "nose" object specifically -- should point outward from the face
for o in mesh_objs:
    if o.name == "nose":
        o.data.calc_normals_split()
        normals = [o.matrix_world.to_3x3() @ v.normal for v in o.data.vertices]
        avg = mathutils.Vector((0,0,0))
        for n in normals: avg += n
        avg.normalize()
        print("nose avg normal (world):", avg)
