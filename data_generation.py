import bpy
import mathutils
import pathlib
from mathutils import Matrix, Vector
from math import radians,sqrt
import numpy as np
import os


#create new object
#x = (1.1,2.2,3.3,4.4)
#y = (1.1,2.2,3.3,4.4)
#z = (1.1,2.2,3.3,4.4)

#for index,val in enumerate(x): 
#    new_obj = bpy.data.objects.new('new_obj', None) 
#    new_obj.location = (x[index],y[index],z[index])
#    bpy.context.scene.objects.link(new_obj)
    
def look_at(obj_camera, point):
    '''
    make the camera look at the object
    '''
    loc_camera = obj_camera.matrix_world.to_translation()
    
    direction = point - loc_camera
    
    # Y up, -Z to
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

    
    bpy.context.view_layer.update()
    
    
def reset_all():
    '''
    delete all the object 
    reset frame
    '''
    for o in bpy.context.scene.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA' or o.type == 'MESH':
            o.select_set(True)
        else:
            o.select_set(False)

    bpy.ops.object.delete()
    
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(0)
    
    #change to metric system
    bpy.context.scene.unit_settings.system = 'METRIC'
    
    #change the measure system
    bpy.context.scene.unit_settings.length_unit = 'METERS'
    
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
    light_data.energy = 2500

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

#    # make it active 
#    bpy.context.view_layer.objects.active = light_object

    #change location
    light_object.location = (4, 4, 10)

    # update scene, if needed
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()
    
def add_mesh(shape,size,location,scale,path = None):
    '''
    add mesh to the scence, it can be primitive, or custom_stl
    '''
    if shape == 'Cube':
        bpy.ops.mesh.primitive_cube_add(size=size, enter_editmode=False, location=location)
        bpy.context.active_object.name = 'new_name'
    
    if shape == 'custom_stl':
        bpy.ops.import_mesh.stl(filepath=path)
        bpy.context.object.scale[0] = scale[0]
        bpy.context.object.scale[1] = scale[1]
        bpy.context.object.scale[2] = scale[2]
        
    bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
    bpy.context.object.scale[0] = 40
    bpy.context.object.scale[1] = 40
    bpy.context.object.scale[2] = 40

        
    
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
#    cam = bpy.data.cameras['Camera']
    cam = bpy.context.object.data
    cam.clip_start = 0.5 
    cam.clip_end = 10
    cam.lens = 25
    cam.type = 'PERSP'

#    cam.type = 'PERSP'
#    bpy.context.object.data.lens = 20

    
    
    
    
def generate_cam_x_y(radius,level = 2,center = (0,0,0),num_loc = 100):
    '''
    generate camera location 
    '''
    locs = np.zeros((num_loc,2))
    locs = np.concatenate((locs,np.ones((num_loc,1)) * level + center[2]),axis = 1)
    
    x_loc = np.random.uniform(-radius,radius,(num_loc,1))
    sign = np.random.choice([-1,1],(num_loc,1))
    y_loc = sign * (radius ** 2 - x_loc ** 2) ** 0.5
    
    for i in range(locs.shape[0]):
        locs[i,0] = x_loc[i] + center[0]
        locs[i,1] = y_loc[i] + center[1]
    
    return locs

def get_camera_pose(path, iteration):
    """
    get the pose of camera
    """
    
    bpy.data.objects['Camera'].rotation_mode = 'QUATERNION'
    q = bpy.data.objects['Camera'].rotation_quaternion
    cam_location = bpy.data.objects['Camera'].matrix_world.to_translation()
    
    m = np.array(
    [[1-2*q[2]*q[2]-2*q[3]*q[3], 2*q[1]*q[2]-2*q[0]*q[3],   2*q[1]*q[3]+2*q[0]*q[2],   cam_location[0]], 
     [2*q[1]*q[2]+2*q[0]*q[3],   1-2*q[1]*q[1]-2*q[3]*q[3], 2*q[2]*q[3]-2*q[0]*q[1],   cam_location[1]],
     [2*q[1]*q[3]-2*q[0]*q[2],   2*q[2]*q[3]+2*q[0]*q[1],   1-2*q[1]*q[1]-2*q[2]*q[2], cam_location[2]],
     [0,                         0,                         0,                         1]])
    
    
    print(m)
    pose_path = str(path.joinpath('data').joinpath('frame-camera-{:06}.pose.npy'.format(iteration)))
    np.save(pose_path , m)
    bpy.data.objects['Camera'].rotation_mode = 'XYZ'
    
        
def save_camera_intrinsics(path,camd):
    
    def get_sensor_size(sensor_fit, sensor_x, sensor_y):
        if sensor_fit == 'VERTICAL':
            return sensor_y
        return sensor_x

    def get_sensor_fit(sensor_fit, size_x, size_y):
        if sensor_fit == 'AUTO':
            if size_x >= size_y:
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'
        return sensor_fit
    
    
    scene = bpy.context.scene
    f_in_mm = camd.lens
    
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    intrinsics = np.array([[s_u,skew,u_0],
                  [0,  s_v, v_0],
                  [0,  0,   1]])
    
    intri_path = str(path.joinpath('data').joinpath('camera-intrinsics.npy'))
#    intri_path_txt = str(path.joinpath('data').joinpath('camera-intrinsics.txt'))
    np.save(intri_path, intrinsics)
#    np.savetxt(intri_path_txt , intrinsics)
    
    


def save_image(BASE_DIR,rgb = True, depth = True):
    
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    #stop use extension
    bpy.context.scene.render.use_file_extension = True

    #create composite layer
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    
    #depth node
    if depth:
        map_value_node = tree.nodes.new('CompositorNodeMapValue')
        depth_file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        
        g_depth_clip_start = 0.5
        g_depth_clip_end = 30
        
        g_depth_color_mode = 'BW'
        g_depth_color_depth = '16'
        g_depth_file_format = 'PNG'
#        g_depth_file_format = 'OPEN_EXR'
        
#        map_value_node.offset[0] = -g_depth_clip_start
#        map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
#        map_value_node.use_min = True
#        map_value_node.use_max = True
#        map_value_node.min[0] = 0.0
#        map_value_node.max[0] = 1.0     
        map_value_node.size[0] = 1/ bpy.context.object.data.clip_end
        
        depth_file_output_node.format.color_mode = g_depth_color_mode
        depth_file_output_node.format.color_depth = g_depth_color_depth
        depth_file_output_node.format.file_format = g_depth_file_format 
        depth_file_output_node.base_path = str(BASE_DIR.joinpath('data'))

        #normalized by far cliping
        links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
        links.new(map_value_node.outputs[0], depth_file_output_node.inputs[0])
        
#        links.new(render_layer_node.outputs[2], depth_file_output_node.inputs[0])
        depth_file_output_node.file_slots[0].path = 'frame-######.depth'

    #color node
    if rgb:
        scale_node = tree.nodes.new('CompositorNodeScale')
        alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
        color_file_output_node = tree.nodes.new('CompositorNodeOutputFile')

        g_scale_space = 'RENDER_SIZE'
        scale_node.space = g_scale_space
        color_file_output_node.base_path = str(BASE_DIR.joinpath('data'))

        links.new(render_layer_node.outputs[0], scale_node.inputs[0])
        links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
        links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
        links.new(alpha_over_node.outputs[0], color_file_output_node.inputs[0])
        
        color_file_output_node.file_slots[0].path = 'frame-######.color'
    
    #rendering results  
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.render.image_settings.color_depth = '16'
    scene.view_settings.view_transform = 'Raw'
    scene.sequencer_colorspace_settings.name = 'Raw'

    

    bpy.ops.render.render(write_still=False)
    current_frame = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(current_frame + 1)
    
    
def duplicate_obj(obj):
    obj_copy = obj.copy()
    obj_copy.data = obj_copy.data.copy()
    bpy.context.collection.objects.link(obj_copy)
    

def transform_and_save(path,num,obj,scale,angle,location = (30,0,0)):
    
    #transformation
    rot_mat = Matrix.Rotation(radians(angle), 4, 'Z')        
    trans_mat = Matrix.Translation(location)
    mat = trans_mat @ rot_mat
    
    #record vertices
    vertics = np.zeros((len(obj.data.vertices),3))
    for i,vert in enumerate(obj.data.vertices):
#        vert.co = mat @ vert.co
        vertics[i,:] = vert.co
#        vertics[i,:] = obj.matrix_world @ vert.co

    obj.matrix_world = obj.matrix_world @ mat    
    save_path_npy = str(path.joinpath('data').joinpath('frame-object-{:06}.pose.npy'.format(num)))
#    save_path_txt = str(path.joinpath('data').joinpath('frame-object-{:06}.pose.txt'.format(num)))
    save_path_npy_vertices = str(path.joinpath('data').joinpath('frame-object-{:06}.vertices.npy'.format(num)))
    
    #scale with trans and rot
#    np.save(save_path_npy,np.array(scale @ mat))
    np.save(save_path_npy,np.array(scale @ mat))
#    np.savetxt(save_path_txt,np.array(obj.matrix_world @ mat))
    np.save(save_path_npy_vertices,vertics)
    
    return mat

def save_object_scale(obj,path):
    
    scale_path = str(path.joinpath('data').joinpath('object_scale.npy'))
    scale = np.array(obj.matrix_world.copy())
    np.save(scale_path,scale)



def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit



if __name__ == '__main__':
    
    num_image = 10
    print('\n' * 20 + 'start' + '-' * 30)
    reset_all()
    
    #get the working directory
    BASE_DIR, STL_DIR, all_STL = get_dir_file_path()
    
    
    if not os.path.exists(BASE_DIR.joinpath('data')):
        os.chdir(BASE_DIR)
        os.mkdir('data')
    
    #add custom stl file
    add_mesh('custom_stl',1,(0,0,0),(0.1,0.1,0.1),all_STL[3])
    add_camera(location = (15,0,0),rotation = (0,0,0))
    
    #save intrinsics
    cam = bpy.data.cameras["Camera"]
    cam = bpy.context.object.data
    save_camera_intrinsics(BASE_DIR,cam)

    #selet object
    obj = bpy.data.objects['small B']
    cam_locs = generate_cam_x_y(radius = 2,level = 5,center = (1.5,0,0),num_loc = num_image)
    
    #duplicate
    duplicate_obj(obj)
    obj_copy =bpy.data.objects['small B.001']
    obj_camera = bpy.data.objects["Camera"]
        
    scale = obj.matrix_world.copy()
    save_object_scale(obj,BASE_DIR)
    for num in range(num_image):
        
        #rotate object and save object pose
        angle = np.random.uniform(0,1) * 360
        mat = transform_and_save(BASE_DIR,num,obj,scale,angle = 90,location = (30,0,0))
        print('mat',mat)        
        #change camera location
        print(cam_locs[num,:])
        obj_camera.location = cam_locs[num,:]
        bpy.context.view_layer.update()
        
        #make the camera look at the object
        look_at(obj_camera, mathutils.Vector([1.5,0,0]))


        get_camera_pose(BASE_DIR,num)                
         
        #select the camera
        bpy.context.scene.camera = bpy.context.object
#        obj = bpy.data.objects['Camera']
        
        
        #save image
        save_image(BASE_DIR,rgb = True, depth = True)
        
        print(obj.matrix_world @ list(obj.data.vertices)[0].co)
        
        #reset obj pose
        Matrix.invert(mat)
        obj.matrix_world = obj.matrix_world @ mat
        
        print(obj.matrix_world @ list(obj.data.vertices)[0].co)
        
        
        

        


