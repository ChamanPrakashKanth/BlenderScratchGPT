"""
Curriculum Data Generator & Fine-Tuner for Sparse-AST Models (3M, 10M, 100M, 200M)
Focus Areas:
1. 3D Mathematics: Vectors, Matrices, Quaternions, Euler, Coordinate Frames, Geometry, SLERP, Intersections
2. Core Blender Python Libraries: bpy, mathutils, bmesh, math, numpy, gpu, gpu_extras, bpy_extras
"""

import os
import json
import math
import random

def generate_curriculum_text():
    samples = []
    
    # -------------------------------------------------------------
    # MODULE 1: 3D Vector Math & mathutils.Vector
    # -------------------------------------------------------------
    samples.append('''
# 3D Vector Mathematics in Blender Python using mathutils
import math
import mathutils
from mathutils import Vector

# Vector Initialization
v1 = Vector((1.0, 2.0, 3.0))
v2 = Vector((4.0, 5.0, 6.0))

# Vector Addition & Subtraction
v_add = v1 + v2  # Vector((5.0, 7.0, 9.0))
v_sub = v2 - v1  # Vector((3.0, 3.0, 3.0))

# Magnitude (Length) of Vector
length = v1.length  # sqrt(1^2 + 2^2 + 3^2) = sqrt(14)
length_manual = math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)

# Normalization (Unit Vector)
# A unit vector has length 1.0, preserving direction: v_hat = v / ||v||
v_unit = v1.normalized()

# Dot Product (Scalar Product)
# Formula: a . b = ||a|| * ||b|| * cos(theta) = ax*bx + ay*by + az*bz
# Applications: Angle between vectors, lighting calculations, projections
dot_val = v1.dot(v2)
cos_theta = dot_val / (v1.length * v2.length)
theta_radians = math.acos(max(min(cos_theta, 1.0), -1.0))
angle_deg = math.degrees(v1.angle(v2))

# Cross Product (Vector Product)
# Formula: a x b produces a vector perpendicular (orthogonal) to both a and b
# Magnitude: ||a x b|| = ||a|| * ||b|| * sin(theta)
# Crucial for computing polygon face normals and tangent spaces
normal_vec = v1.cross(v2)

# Vector Projection: Projecting v1 onto v2
# proj_v2(v1) = ( (v1 . v2) / ||v2||^2 ) * v2
proj = (v1.dot(v2) / (v2.length**2)) * v2

# Vector Rejection: Perpendicular component of v1 relative to v2
# rej_v2(v1) = v1 - proj_v2(v1)
rej = v1 - proj

# Vector Reflection: Reflecting vector v off a surface with normal n
# r = v - 2 * (v . n) * n
def reflect_vector(v: Vector, n: Vector) -> Vector:
    n_unit = n.normalized()
    return v - 2.0 * v.dot(n_unit) * n_unit

# Linear Interpolation (LERP) between vectors: p(t) = (1 - t)*v1 + t*v2
def lerp_vector(a: Vector, b: Vector, t: float) -> Vector:
    return a.lerp(b, t)

# Distance between two points in 3D space
distance = (v2 - v1).length
''')

    # -------------------------------------------------------------
    # MODULE 2: Transformation Matrices & Coordinate Frames
    # -------------------------------------------------------------
    samples.append('''
# 4x4 Affine Transformation Matrices in Blender Python
import math
import mathutils
from mathutils import Matrix, Vector, Euler

# In 3D graphics, a 4x4 matrix encodes Translation, Rotation, and Scale:
# [ R00*Sx  R01*Sy  R02*Sz  Tx ]
# [ R10*Sx  R11*Sy  R12*Sz  Ty ]
# [ R20*Sx  R21*Sy  R22*Sz  Tz ]
# [   0       0       0      1 ]

# 1. Identity Matrix
mat_ident = Matrix.Identity(4)

# 2. Translation Matrix
translation = Vector((3.0, -2.0, 5.0))
mat_trans = Matrix.Translation(translation)

# 3. Rotation Matrix (e.g. 45 degrees around Z axis)
angle = math.radians(45.0)
mat_rot = Matrix.Rotation(angle, 4, 'Z')

# 4. Scale Matrix
mat_scale = Matrix.Diagonal(Vector((2.0, 2.0, 0.5, 1.0)))

# 5. Shear Matrix along axis
mat_shear = Matrix.Shear('XY', 4, 0.5)

# Chaining Transformations:
# In Blender / OpenGL conventions: WorldPos = Mat_Trans @ Mat_Rot @ Mat_Scale @ LocalPos
mat_combined = mat_trans @ mat_rot @ mat_scale

# Transforming a 3D point (using 4x4 affine multiplication)
local_pt = Vector((1.0, 0.0, 0.0))
world_pt = mat_combined @ local_pt

# Matrix Inversion
mat_inv = mat_combined.inverted()

# Decomposing a 4x4 Transformation Matrix into Components:
# loc (Vector), rot (Quaternion), scale (Vector)
loc, rot_quat, scale = mat_combined.decompose()

# Coordinate Transformation: Local to World and World to Local
def local_to_world(matrix_world: Matrix, local_coord: Vector) -> Vector:
    return matrix_world @ local_coord

def world_to_local(matrix_world: Matrix, world_coord: Vector) -> Vector:
    return matrix_world.inverted() @ world_coord
''')

    # -------------------------------------------------------------
    # MODULE 3: Quaternions, Euler Angles & SLERP
    # -------------------------------------------------------------
    samples.append('''
# Quaternions, Rotations, and Gimbal Lock Prevention in Blender
import math
import mathutils
from mathutils import Quaternion, Euler, Vector

# Euler Rotations: Represented by 3 sequential rotations (Pitch, Yaw, Roll)
# Susceptible to Gimbal Lock when two rotation axes align (loss of 1 degree of freedom)
euler_rot = Euler((math.radians(30), math.radians(45), math.radians(60)), 'XYZ')

# Quaternions: Hypercomplex numbers represented as q = w + xi + yj + zk
# Avoid gimbal lock and provide smooth spherical interpolation (SLERP)
quat = euler_rot.to_quaternion()

# Creating a Quaternion from an Axis and Angle
axis = Vector((0.0, 0.0, 1.0)).normalized()
angle = math.radians(90.0)
q_axis_angle = Quaternion(axis, angle)

# Quaternion Multiplication (Combining Rotations)
# Note: Non-commutative! q_total = q2 @ q1 (q1 applied first, then q2)
q1 = Quaternion((1.0, 0.0, 0.0), math.radians(45))
q2 = Quaternion((0.0, 1.0, 0.0), math.radians(30))
q_combined = q2 @ q1

# Rotating a Vector by a Quaternion: v_rotated = q @ v
v = Vector((1.0, 0.0, 0.0))
v_rotated = q_combined @ v

# Spherical Linear Interpolation (SLERP) for smooth camera/character rotation
# Factor t ranges from 0.0 (start) to 1.0 (end)
t = 0.5
q_interpolated = q1.slerp(q2, t)

# Swing-Twist Decomposition: Decompose rotation into swing (axis perpendicular to twist) and twist (along axis)
def swing_twist_decomposition(q: Quaternion, twist_axis: Vector):
    n = twist_axis.normalized()
    proj = Vector((q.x, q.y, q.z)).dot(n) * n
    q_twist = Quaternion((q.w, proj.x, proj.y, proj.z)).normalized()
    q_swing = q @ q_twist.inverted()
    return q_swing, q_twist
''')

    # -------------------------------------------------------------
    # MODULE 4: 3D Geometry Algorithms & Intersections
    # -------------------------------------------------------------
    samples.append('''
# 3D Geometry, Intersections, and Spatial Algorithms
import math
import mathutils
from mathutils import Vector, Matrix, geometry

# 1. Ray-Plane Intersection:
# Ray: P(t) = ray_origin + t * ray_direction
# Plane: (P - plane_point) . plane_normal = 0
def intersect_ray_plane(ray_origin: Vector, ray_dir: Vector, plane_pt: Vector, plane_norm: Vector):
    denom = ray_dir.dot(plane_norm)
    if abs(denom) < 1e-6:
        return None  # Ray is parallel to plane
    t = (plane_pt - ray_origin).dot(plane_norm) / denom
    if t < 0:
        return None  # Intersection behind ray origin
    return ray_origin + t * ray_dir

# 2. Möller-Trumbore Ray-Triangle Intersection Algorithm
def ray_triangle_intersection(ray_origin: Vector, ray_dir: Vector, v0: Vector, v1: Vector, v2: Vector):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)
    if abs(a) < 1e-7:
        return None  # Ray is parallel to triangle
    f = 1.0 / a
    s = ray_origin - v0
    u = f * s.dot(h)
    if u < 0.0 or u > 1.0:
        return None
    q = s.cross(edge1)
    v = f * ray_dir.dot(q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * edge2.dot(q)
    if t > 1e-7:
        hit_pos = ray_origin + ray_dir * t
        return hit_pos, t, u, v
    return None

# 3. Builtin mathutils.geometry algorithms:
v0, v1, v2 = Vector((0,0,0)), Vector((2,0,0)), Vector((0,2,0))
hit_pt = geometry.intersect_ray_tri(v0, v1, v2, Vector((0,0,-1)), Vector((0.5, 0.5, 5.0)), False)

# Triangle Area using cross product
tri_area = 0.5 * (v1 - v0).cross(v2 - v0).length

# Axis-Aligned Bounding Box (AABB) computation
def compute_aabb(points):
    min_pt = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_pt = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_pt, max_pt
''')

    # -------------------------------------------------------------
    # MODULE 5: Core Blender API (bpy.data, bpy.context, bpy.ops, bpy.types)
    # -------------------------------------------------------------
    samples.append('''
# Core bpy API Architecture: Data, Context, Operators, and Custom UI
import bpy
import math
from mathutils import Vector, Matrix, Euler

# 1. bpy.data: Direct access to all datablocks
mesh_datablocks = bpy.data.meshes
object_datablocks = bpy.data.objects
material_datablocks = bpy.data.materials

# Creating a mesh and object cleanly without operators:
mesh = bpy.data.meshes.new(name="ProceduralMesh")
obj = bpy.data.objects.new(name="ProceduralObject", object_data=mesh)
bpy.context.collection.objects.link(obj)

# 2. bpy.context: Context state
active_obj = bpy.context.active_object
selected_objs = bpy.context.selected_objects
current_scene = bpy.context.scene

# Object Transforms
obj.location = Vector((0.0, 2.0, 1.5))
obj.rotation_euler = Euler((0.0, 0.0, math.radians(45)), 'XYZ')
obj.scale = Vector((1.0, 1.0, 2.0))

# 3. Modifiers via bpy:
subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
subsurf.levels = 2
subsurf.render_levels = 3

bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
bevel.width = 0.05
bevel.segments = 4

# 4. bpy.ops: User interface operators
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.shade_smooth()

# 5. Custom Operator & UI Panel Definition:
class MESH_OT_procedural_generator(bpy.types.Operator):
    bl_idname = "mesh.procedural_generator"
    bl_label = "Generate 3D Math Primitive"
    bl_options = {'REGISTER', 'UNDO'}
    
    segments: bpy.props.IntProperty(name="Segments", default=32, min=3, max=256)
    radius: bpy.props.FloatProperty(name="Radius", default=1.0, min=0.01)
    
    def execute(self, context):
        bpy.ops.mesh.primitive_cylinder_add(vertices=self.segments, radius=self.radius, depth=2.0)
        self.report({'INFO'}, f"Generated cylinder with {self.segments} segments.")
        return {'FINISHED'}
''')

    # -------------------------------------------------------------
    # MODULE 6: bmesh Procedural Geometry & Topology
    # -------------------------------------------------------------
    samples.append('''
# High-Performance Procedural Geometry with bmesh
import bpy
import bmesh
import math
from mathutils import Vector, Matrix

# bmesh provides mutable B-Rep geometry with vertices, edges, faces, and loops.

def generate_parametric_mobius_strip(radius=2.0, width=0.5, segments=64):
    mesh = bpy.data.meshes.new("MobiusStrip")
    obj = bpy.data.objects.new("MobiusObject", mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    verts_top = []
    verts_bottom = []
    
    # Parametric Mobius Strip Formula:
    # u in [0, 2*pi], v in [-w/2, w/2]
    # x = (R + v * cos(u/2)) * cos(u)
    # y = (R + v * cos(u/2)) * sin(u)
    # z = v * sin(u/2)
    for i in range(segments):
        u = (2.0 * math.pi * i) / segments
        half_u = u / 2.0
        v_offset = width / 2.0
        
        x1 = (radius - v_offset * math.cos(half_u)) * math.cos(u)
        y1 = (radius - v_offset * math.cos(half_u)) * math.sin(u)
        z1 = -v_offset * math.sin(half_u)
        
        x2 = (radius + v_offset * math.cos(half_u)) * math.cos(u)
        y2 = (radius + v_offset * math.cos(half_u)) * math.sin(u)
        z2 = v_offset * math.sin(half_u)
        
        verts_top.append(bm.verts.new((x1, y1, z1)))
        verts_bottom.append(bm.verts.new((x2, y2, z2)))
        
    bm.verts.ensure_lookup_table()
    
    # Construct Faces
    for i in range(segments):
        next_i = (i + 1) % segments
        bm.faces.new((verts_top[i], verts_bottom[i], verts_bottom[next_i], verts_top[next_i]))
        
    # bmesh Operations: Bevel and Extrude
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj

# Custom Data Layers (UVs and Vertex Weights in bmesh)
def add_custom_layers_bmesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    # UV Layer
    uv_layer = bm.loops.layers.uv.new("CustomUV")
    for face in bm.faces:
        for loop in face.loops:
            loop[uv_layer].uv = (loop.vert.co.x, loop.vert.co.y)
            
    bm.to_mesh(mesh)
    bm.free()
''')

    # -------------------------------------------------------------
    # MODULE 7: mathutils.kdtree & mathutils.bvhtree Spatial Indexing
    # -------------------------------------------------------------
    samples.append('''
# High-Speed Spatial Indexing with mathutils.kdtree and mathutils.bvhtree
import bpy
import mathutils
from mathutils import Vector, kdtree, bvhtree

# 1. KDTree for Fast Nearest-Neighbor Vertex Searches:
# Complexity: O(log N) query time compared to O(N) linear search
def find_nearest_vertices(mesh_obj, query_point: Vector, k_nearest: int = 3):
    mesh = mesh_obj.data
    size = len(mesh.vertices)
    kd = kdtree.KDTree(size)
    
    for i, vert in enumerate(mesh.vertices):
        kd.insert(vert.co, i)
    kd.balance()
    
    # Query nearest point
    co, index, dist = kd.find(query_point)
    
    # Query k nearest points
    nearest_k = kd.find_n(query_point, k_nearest)
    return nearest_k

# 2. BVHTree for Raycasting and Mesh Collision:
# Build bounding volume hierarchy from polygon faces
def raycast_mesh_bvh(mesh_obj, ray_origin: Vector, ray_direction: Vector):
    # Transform ray to local coordinate space of the object
    mat_inv = mesh_obj.matrix_world.inverted()
    local_origin = mat_inv @ ray_origin
    local_dir = (mat_inv.to_3x3() @ ray_direction).normalized()
    
    bvh = bvhtree.BVHTree.FromPolygons([v.co for v in mesh_obj.data.vertices],
                                       [f.vertices for f in mesh_obj.data.polygons],
                                       epsilon=1e-5)
                                       
    hit_loc, hit_norm, hit_idx, hit_dist = bvh.ray_cast(local_origin, local_dir)
    if hit_loc is not None:
        world_hit_loc = mesh_obj.matrix_world @ hit_loc
        world_hit_norm = (mesh_obj.matrix_world.to_3x3() @ hit_norm).normalized()
        return world_hit_loc, world_hit_norm, hit_idx, hit_dist
    return None
''')

    # -------------------------------------------------------------
    # MODULE 8: Vectorized Mesh Operations with numpy & math
    # -------------------------------------------------------------
    samples.append('''
# Vectorized Vertex Deformations using numpy in Blender
import bpy
import numpy as np
import math

# Using foreach_get and foreach_set with numpy provides a 50x-100x speedup
# over standard python for-loops when modifying dense meshes (100k+ vertices).

def apply_procedural_wave_deformation(obj, frequency=2.0, amplitude=0.3, time_val=0.0):
    mesh = obj.data
    num_verts = len(mesh.vertices)
    
    # 1. Allocate flat numpy array and extract coordinates
    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', coords)
    coords = coords.reshape((num_verts, 3))
    
    # 2. Vectorized 3D Mathematical Transformation with numpy
    x = coords[:, 0]
    y = coords[:, 1]
    dist_from_origin = np.sqrt(x**2 + y**2)
    
    # Sine wave displacement along Z axis
    coords[:, 2] += amplitude * np.sin(frequency * dist_from_origin + time_val)
    
    # 3. Write modified coordinates back into Blender mesh datablock
    mesh.vertices.foreach_set('co', coords.ravel())
    mesh.update()

def transform_vertices_numpy(obj, transform_matrix):
    mesh = obj.data
    num_verts = len(mesh.vertices)
    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', coords)
    
    # Reshape and append homogeneous coordinate 1.0: (N, 4)
    ones = np.ones((num_verts, 1), dtype=np.float32)
    homo_coords = np.hstack((coords.reshape((num_verts, 3)), ones))
    
    # Multiply by 4x4 matrix: (N, 4) @ (4, 4).T
    mat_np = np.array(transform_matrix, dtype=np.float32)
    transformed = (homo_coords @ mat_np.T)[:, :3]
    
    mesh.vertices.foreach_set('co', transformed.ravel())
    mesh.update()
''')

    # -------------------------------------------------------------
    # MODULE 9: Viewport Drawing with gpu & gpu_extras.batch
    # -------------------------------------------------------------
    samples.append('''
# 3D Viewport Drawing and Custom Visualizations with gpu and gpu_extras
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

# Drawing 3D vectors and mathematical lines directly into the Blender Viewport
def create_3d_line_batch(start_pt: Vector, end_pt: Vector, color=(1.0, 0.2, 0.2, 1.0)):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    coords = [start_pt[:], end_pt[:]]
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    return shader, batch, color

def draw_callback_3d_gizmo():
    # Example drawing a coordinate frame (X=Red, Y=Green, Z=Blue)
    origin = Vector((0, 0, 0))
    axes = [
        (Vector((1, 0, 0)), (1.0, 0.0, 0.0, 1.0)),
        (Vector((0, 1, 0)), (0.0, 1.0, 0.0, 1.0)),
        (Vector((0, 0, 1)), (0.0, 0.5, 1.0, 1.0))
    ]
    gpu.state.line_width_set(3.0)
    for axis_vec, color in axes:
        shader, batch, col = create_3d_line_batch(origin, axis_vec, color)
        shader.bind()
        shader.uniform_float("color", col)
        batch.draw(shader)
    gpu.state.line_width_set(1.0)

# Register drawing handler in 3D Viewport
# handler = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d_gizmo, (), 'WINDOW', 'POST_VIEW')
# bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
''')

    # -------------------------------------------------------------
    # MODULE 10: bpy_extras & Interactive Viewport Raycasting
    # -------------------------------------------------------------
    samples.append('''
# Mouse Raycasting and Viewport Coordinates with bpy_extras.view3d_utils
import bpy
from bpy_extras import view3d_utils
from mathutils import Vector

def get_mouse_3d_ray(context, event):
    # Converts 2D mouse pixel coordinates in viewport into 3D world origin & direction
    region = context.region
    region_3d = context.space_data.region_3d
    mouse_coord = (event.mouse_region_x, event.mouse_region_y)
    
    # 3D Ray Origin in world space
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, region_3d, mouse_coord)
    
    # 3D Ray Direction in world space (normalized vector pointing into the scene)
    ray_vector = view3d_utils.region_2d_to_vector_3d(region, region_3d, mouse_coord)
    
    return ray_origin, ray_vector

def project_mouse_to_ground_plane(context, event, plane_z=0.0):
    ray_origin, ray_vector = get_mouse_3d_ray(context, event)
    if abs(ray_vector.z) < 1e-6:
        return None
    t = (plane_z - ray_origin.z) / ray_vector.z
    if t < 0:
        return None
    ground_pt = ray_origin + t * ray_vector
    return ground_pt
''')

    # -------------------------------------------------------------
    # MODULE 11: Shader Nodes, Materials & Procedural Vector Math
    # -------------------------------------------------------------
    samples.append('''
# Material & Shader Node Setup in Blender Python with Vector Math Nodes
import bpy

def create_procedural_math_material(mat_name="MathShader"):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output Node
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (400, 0)
    
    # Principled BSDF
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.location = (100, 0)
    node_bsdf.inputs['Roughness'].default_value = 0.2
    node_bsdf.inputs['Metallic'].default_value = 0.8
    
    # Texture Coordinate & Vector Math
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_tex_coord.location = (-600, 0)
    
    # Vector Math Node (DOT_PRODUCT, CROSS_PRODUCT, NORMALIZE, DISTANCE)
    node_vec_math = nodes.new(type='ShaderNodeVectorMath')
    node_vec_math.operation = 'DOT_PRODUCT'
    node_vec_math.location = (-350, 0)
    node_vec_math.inputs[1].default_value = (0.0, 0.0, 1.0) # Up vector
    
    # Color Ramp Node
    node_ramp = nodes.new(type='ShaderNodeValToRGB')
    node_ramp.location = (-100, 0)
    
    # Link Nodes
    links.new(node_tex_coord.outputs['Normal'], node_vec_math.inputs[0])
    links.new(node_vec_math.outputs['Value'], node_ramp.inputs['Fac'])
    links.new(node_ramp.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    
    return mat
''')

    return "\n\n".join(samples)

if __name__ == '__main__':
    out_dir = r"c:\Users\user\Downloads\checkpoint"
    text = generate_curriculum_text()
    out_file = os.path.join(out_dir, "blender_3dmath_curriculum.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Generated Blender 3D Math & Python Libraries curriculum: {len(text):,} characters saved to {out_file}")
