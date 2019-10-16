import bpy
import mathutils
import pathlib
import numpy as np

#create new object
#x = (1.1,2.2,3.3,4.4)
#y = (1.1,2.2,3.3,4.4)
#z = (1.1,2.2,3.3,4.4)

#for index,val in enumerate(x): 
#    new_obj = bpy.data.objects.new('new_obj', None) 
#    new_obj.location = (x[index],y[index],z[index])
#    bpy.context.scene.objects.link(new_obj)

def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=10.0):
    
    looking_direction = camera.location - focus_point
    print('looking_direction',looking_direction)
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    print('rot_quat',rot_quat)
    
    camera.rotation_euler = rot_quat.to_euler()
    print('camera.rotation_euler',camera.rotation_euler)
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))
    print('camera.location',camera.location)
    
def look_at(obj_camera, point):
    '''
    make the camera look at the object
    '''
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
def delete_all():
    '''
    delete all the object 
    '''
    for o in bpy.context.scene.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA' or o.type == 'MESH':
            o.select_set(True)
        else:
            o.select_set(False)

    bpy.ops.object.delete()

def add_mesh(shape,size,location,path = None):
    '''
    add mesh to the scence, it can be primitive, or custom_stl
    '''
    if shape == 'Cube':
        bpy.ops.mesh.primitive_cube_add(size=size, enter_editmode=False, location=location)
        bpy.context.active_object.name = 'new_name'
    
    if shape == 'custom_stl':
        bpy.ops.import_mesh.stl(filepath=path)
        bpy.context.object.scale[0] = 0.1
        bpy.context.object.scale[1] = 0.1
        bpy.context.object.scale[2] = 0.3
        
#        bpy.context.scene.objects[1].scale[0] = 0.1
#        bpy.context.scene.objects[1].scale[1] = 0.1
#        bpy.context.scene.objects[1].scale[2] = 0.3
    
def get_dir_file_path():
    BASE_DIR = pathlib.Path(__file__)
    BASE_DIR = BASE_DIR.parent.parent

    #get base directory
    STL_DIR = BASE_DIR.joinpath('env').joinpath('mesh')
    all_STL = list(STL_DIR.glob('**/*.stl'))
    all_STL = [str(item) for item in all_STL]
    
    return BASE_DIR, STL_DIR, all_STL

def add_camera(location,rotation,align = 'VIEW'):
    bpy.ops.object.camera_add(enter_editmode=False, align=align, location=location, rotation=rotation)
    
def generate_cam_x_y(radius,level = 5,center = (0,0,0),num_loc = 100):
    '''
    generate camera location 
    '''
    locs = np.zeros((num_loc,2))
    locs = np.concatenate((locs,np.ones((num_loc,1)) * level + center[2]),axis = 1)
    
    x_loc = np.random.uniform(-radius,radius,(num_loc,1))
    print(x_loc.shape)
    sign = np.random.choice([-1,1],(num_loc,1))
    y_loc = sign * (radius ** 2 - x_loc ** 2) ** 0.5
    
    print(y_loc.shape)
    for i in range(locs.shape[0]):
        locs[i,0] = x_loc[i] + center[0]
        locs[i,1] = y_loc[i] + center[1]
    
    return locs
        
    
    
    

if __name__ == '__main__':
    delete_all()
    BASE_DIR, STL_DIR, all_STL = get_dir_file_path()
    add_mesh('custom_stl',1,(0,0,0),all_STL[3])
    add_camera((5.0,2.0,6.0),(0,0,0))
    
#    print(bpy.context.object)
    
    #selet object
    obj_camera = bpy.data.objects["Camera"]
    obj_other = bpy.data.objects['small B']
    cam_locs = generate_cam_x_y(5,obj_camera.location[2])
    
    for i in range(30):
        
        obj_camera.location = (2.0, 2.0, 6.0)
        look_at(obj_camera, obj_other.matrix_world.to_translation())

#        update_camera(bpy.data.objects['Camera'])









# def get_calibration_matrix_K_from_blender(camd):
#     f_in_mm = camd.lens
#     scene = bpy.context.scene
#     resolution_x_in_px = scene.render.resolution_x
#     resolution_y_in_px = scene.render.resolution_y
#     scale = scene.render.resolution_percentage / 100
#     sensor_width_in_mm = camd.sensor_width
#     sensor_height_in_mm = camd.sensor_height
#     pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
#     if (camd.sensor_fit == 'VERTICAL'):
#         # the sensor height is fixed (sensor fit is horizontal), 
#         # the sensor width is effectively changed with the pixel aspect ratio
#         s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
#         s_v = resolution_y_in_px * scale / sensor_height_in_mm
#     else: # 'HORIZONTAL' and 'AUTO'
#         # the sensor width is fixed (sensor fit is horizontal), 
#         # the sensor height is effectively changed with the pixel aspect ratio
#         pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
#         s_u = resolution_x_in_px * scale / sensor_width_in_mm
#         s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

#     # Parameters of intrinsic calibration matrix K
#     alpha_u = f_in_mm * s_u
#     alpha_v = f_in_mm * s_v
#     u_0 = resolution_x_in_px*scale / 2
#     v_0 = resolution_y_in_px*scale / 2
#     skew = 0
    
#     K = np.array([[alpha_u,skew,u_0],
#              [0,alpha_v,v_0],
#              [0,0,1]])
#     return K 

# def get_3x4_RT_matrix_from_blender(cam):
#     # bcam stands for blender camera
#     R_bcam2cv = Matrix(
#         ((1, 0,  0),
#          (0, -1, 0),
#          (0, 0, -1)))

#     # Transpose since the rotation is object rotation, 
#     # and we want coordinate rotation
#     # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
#     # T_world2bcam = -1*R_world2bcam * location
#     #
#     # Use matrix_world instead to account for all constraints
#     location, rotation = cam.matrix_world.decompose()[0:2]
#     R_world2bcam = rotation.to_matrix().transposed()

#     # Convert camera location to translation vector used in coordinate changes
#     # T_world2bcam = -1*R_world2bcam*cam.location
#     # Use location from matrix_world to account for constraints:     
#     T_world2bcam = -1*R_world2bcam * location

#     # Build the coordinate transform matrix from world to computer vision camera
#     R_world2cv = R_bcam2cv*R_world2bcam
#     T_world2cv = R_bcam2cv*T_world2bcam

#     # put into 3x4 matrix
#     RT = Matrix((
#         R_world2cv[0][:] + (T_world2cv[0],),
#         R_world2cv[1][:] + (T_world2cv[1],),
#         R_world2cv[2][:] + (T_world2cv[2],)
#          ))
#     return RT

# def get_3x4_P_matrix_from_blender(cam):
#     K = get_calibration_matrix_K_from_blender(cam.data)
#     RT = get_3x4_RT_matrix_from_blender(cam)
#     return K*RT, K, RT

# def save_files(path,rgb_bool,depth_bool,pose_bool,rgb,depth,pose):
#     color_file = path + '.color.jpg'
#     depth_file = path + '.depth.png'
#     pose_file = path + '.pose.txt'

#     if rgb_bool:
#         # screen = cv2.cvtColor(rgb[:,:,:3], cv2.COLOR_RGB2BGR)
#         # cv2.imwrite(color_file,rgb[:,:,:3])
#         im = Image.fromarray(rgb[:,:,:3])
#         im.save(color_file)

#     if depth_bool:
#         # print('-' * 100)
#         cv2.imwrite(depth_file, depth)

#     if pose_bool:
#         np.savetxt(pose_file, pose, fmt="%d")
    
    


# #delete object
# bpy.ops.object.select_all(action = 'SELECT')
# bpy.ops.object.delete(use_global = True)

# #select files
# BASE_DIR = pathlib.Path(__file__)
# BASE_DIR = BASE_DIR.parent.parent

# #get base directory
# STL_DIR = BASE_DIR.joinpath('env').joinpath('mesh')
# all_STL = list(STL_DIR.glob('**/*.stl'))
# all_STL = [str(item) for item in all_STL]

# #bpy.ops.mesh.primitive_cube_add(size = 2,enter_editmode = False, location = (0,0,0))
# ##get stl files 
# print('hahahahhahahaha')
# bpy.ops.import_mesh.stl(filepath=all_STL[2])
# obj_new = bpy.context.selected_objects[0] 
# ob_new.ctive_material.diffuse_color = (0.8, 0.0542605, 0.0451551, 1)
 
# #bpy.ops.object.origin_set(type = 'GEOMETRY_ORIGIN',center = 'MEDIAN')

# #set object scale
# #bpy.context.object.scale[0] = 0.1
# #bpy.context.object.scale[1] = 0.1
# #bpy.context.object.scale[2] = 0.3
# ob = bpy.data.objects[0]
# print('-' * 30)
# vertices_list = [ob.data.vertices[i] for i in range(len(ob.data.vertices))]

# #add camera
# bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(4, 4, 4), rotation=(-math.pi/4, math.pi/4,0))
# bpy.context.scene.camera = bpy.context.object
# obj = bpy.data.objects['Camera'] # bpy.types.Camera



# num = 0
# #save the image
# for num in range(5):
#     scene = bpy.context.scene
    
#     file = BASE_DIR.joinpath('data').joinpath('frame-' +str(0) * (6 - len([char for char in str(num)])) + str(num))
#     color_file = str(file) + '.color.jpg'
#     bpy.context.scene.render.filepath = color_file

#     obj.location.x = 5.3869
#     obj.location.y = 6.0534
#     obj.location.z = 5.7654
#     obj.rotation_euler.x = 54/360* math.pi * 2
#     obj.rotation_euler.y = 13.2/360 * math.pi * 2
#     obj.rotation_euler.z = 134/360 * math.pi * 2
#     bpy.context.scene.render.resolution_x = 640
#     bpy.context.scene.render.resolution_y = 640
#     bpy.context.scene.render.image_settings.compression = 0
    
#     print("Hello World")
#     bpy.data.scenes['Scene'].render.filepath = color_file
#     bpy.ops.render.render( write_still=True )
#     print("Done")
# #    bpy.ops.image.save_as(save_as_render=True, copy=True, filepath=color_file, show_multiview=False, use_multiview=False)






# print('-' * 20)
# print(bpy.data.objects[1])
# #bpy.data.objects[]
# ob = bpy.data.objects[1]

# v = ob.data.vertices[0].co
# mat = ob.matrix_world
# #print(v)
# #print(mat)
# #print(mat@v)



# K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)


# #import bpy
# #import mathutils
# #import bpy



# #print(dir(bpy))
# #candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]

# ## select them only.
# #for object_name in candidate_list:
# #    bpy.context.active_object.select_set(state=True)
# #    
# ## remove all selected.
# #bpy.ops.object.delete()

# #for item in bpy.data.meshes:
# #  bpy.data.meshes.remove(item)

# #bpy.ops.mesh.primitive_cylinder_add(radius = 0.5, depth = 1)

# #import bpy
# #import bmesh
# #from math import radians, sqrt
# #from mathutils import Matrix

# #def getMat(name, color):
# #    oldMat = bpy.data.materials.get(name)
# #    if(oldMat != None):
# #        bpy.data.materials.remove(oldMat)
# #    mat = bpy.data.materials.new(name)
# #    mat.diffuse_color = color
# #    return mat

# #def createObjFromCo(meshName, objName, coList, matName, matColor):
# #    bm = bmesh.new()
# #    verts = []
# #    for co in coList:
# #        verts.append(bm.verts.new(co))
# #    bm.faces.new(verts)
# #    mesh = bpy.data.meshes.new(meshName)
# #    obj = bpy.data.objects.new(objName, mesh)
# #    bpy.context.collection.objects.link(obj)
# #    bm.to_mesh(mesh)
# #    mat = getMat(matName, matColor)
# #    obj.data.materials.append(mat)
# #    return obj

# #def applyMatToVerts(obj, mat):
# #    for vert in obj.data.vertices:
# #        vert.co = mat * vert.co

# #cand_list = [item.name for item in bpy.data.objects if item.type == 'MESH']
# #print(len(cand_list))
# #if len(cand_list) > 0:
# #    for obj_name in cand_list:
# #        o = bpy.data.objects[obj_name]
# #        print(o)
# ##        o.select_set(True) 
# ##        bpy.context.active_object.select_set(state = True)
# #print()
# #bpy.ops.object.delete()

# ##bpy.ops.object.select_all(action='DESELECT')
# ##bpy.data.objects['Camera'].select_set(True) # Blender 2.8x
# ##bpy.ops.object.delete() 





# #origObjName = 'origObj'
# #origCopyName = 'origCopyObj'
# #toMapObjName = 'toMapObj'

# #def createObjs():
# #    t = [(-1, -1, -1), (1, 1, -1), (-1, 1, 1), (1, -1, 1)]
# #    u = [(2, 2, -1), (2, 2, 3), (0, 0, 1), (4, 0, 1)]
# #    v = [n]
# #    
# #    orange = (0.8, 0.2, 0.1,1)
# #    green = (0.2, 0.8, 0.2,1)
# #    blue = (0.2, 0.2, 0.8,1)
# #    origObj = createObjFromCo(origObjName+'_data', origObjName, t, '__obj1Mat__', orange)
# #    origCopy = createObjFromCo(origCopyName+'_data', origCopyName, t, '__objCopyMat__', blue)
# #    toMapObj = createObjFromCo(toMapObjName+'_data', toMapObjName, u, '__obj2Mat__',green)

# #createObjs()

# #origObj = bpy.data.objects['origObj']
# #origCopy = bpy.data.objects['origCopyObj']
# #toMapObj = bpy.data.objects['toMapObj']

# #rot_mat1 = Matrix.Rotation(radians(45), 4, 'Z') 
# #scale_mat1 = Matrix.Scale(sqrt(2), 4, (1,0,0))
# #scale_mat2 = Matrix.Scale(sqrt(2), 4, (0,1,0))
# #trans_mat = Matrix.Translation((2, 1, -1))
# #rot_mat2 = Matrix.Rotation(radians(90), 4, 'X') 

# ## ~ applyMatToVerts(origCopy, rot_mat1)
# ## ~ applyMatToVerts(origCopy, scale_mat1)
# ## ~ applyMatToVerts(origCopy, scale_mat2)
# ## ~ applyMatToVerts(origCopy, trans_mat)
# ## ~ applyMatToVerts(origCopy, rot_mat2)


# #applyMatToVerts(origCopy, rot_mat2 * trans_mat * scale_mat1 * scale_mat2 * rot_mat1)