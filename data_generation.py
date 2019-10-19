import bpy
import mathutils
import pathlib
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
    bpy.data.cameras['Camera'].clip_start = 0.5 
    bpy.data.cameras['Camera'].clip_end = 4
    
    
    
def generate_cam_x_y(radius,level = 5,center = (0,0,0),num_loc = 100):
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

def get_pose(path, iteration):
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
     
    
    if not os.path.exists(path.joinpath('data')):
        os.chdir(path)
        os.mkdir('data')
    
    
    pose_path = str(path.joinpath('data').joinpath('frame-{:06}.pose.txt'.format(iteration)))
    print(pose_path)
    np.savetxt(pose_path , m)

def save_color_image(BASE_DIR,iteration):
    file = BASE_DIR.joinpath('data').joinpath('frame-' +str(0) * (6 - len([char for char in str(iteration)])) + str(iteration))
    color_file = str(file) + '.color.png'
    bpy.context.scene.render.filepath = color_file
    bpy.ops.render.render(write_still=True)
    bpy.data.images['Render Result'].save_render(filepath = color_file)


if __name__ == '__main__':
    
    num_image = 1
    print('\n' * 20 + 'start' + '-' * 30)
    delete_all()
    
    #get the working directory
    BASE_DIR, STL_DIR, all_STL = get_dir_file_path()
    
    #add custom stl file
    add_mesh('custom_stl',1,(0,0,0),(0.3,0.3,0.4),all_STL[3])
    add_camera((5.0,2.0,6.0),(0,0,0))
    
    #selet object
    obj_camera = bpy.data.objects["Camera"]
    obj_other = bpy.data.objects['small B']
    cam_locs = generate_cam_x_y(5,20,num_loc = num_image)
    

#    obj = bpy.data.objects['Camera']
    for num in range(num_image):
        
        #change camera location
        obj_camera.location = cam_locs[num,:]
        bpy.context.view_layer.update()
        
        #make the camera look at the object
        look_at(obj_camera, obj_other.matrix_world.to_translation())
        get_pose(BASE_DIR,num)
        
        #select the camera
        bpy.context.scene.camera = bpy.context.object
        obj = bpy.data.objects['Camera']
        
        #save the image
        save_color_image(BASE_DIR,num)
        
        
        
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
        
        for node in tree.nodes:
            tree.nodes.remove(node)
            
        render_layer_node = tree.nodes.new('CompositorNodeRLayers')
        map_value_node = tree.nodes.new('CompositorNodeMapValue')
        file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        
        g_depth_clip_start = 0.5
        g_depth_clip_end = 4
        
        g_depth_color_mode = 'BW'
        g_depth_color_depth = '8'
        g_depth_file_format = 'PNG'
        
        map_value_node.offset[0] = -g_depth_clip_start
        map_value_node.size[0] = 1 / (g_depth_clip_end - g_depth_clip_start)
        map_value_node.use_min = True
        map_value_node.use_max = True
        map_value_node.min[0] = 0.0
        map_value_node.max[0] = 1.0
        
        
        file_output_node.format.color_mode = g_depth_color_mode
        file_output_node.format.color_depth = g_depth_color_depth
        file_output_node.format.file_format = g_depth_file_format 
#        file_output_node.base_path = str(BASE_DIR.joinpath('data'))

        links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
        links.new(map_value_node.outputs[0], file_output_node.inputs[0])

#        if not os.path.exists(g_syn_depth_folder):
#            os.mkdir(g_syn_depth_folder)

        file_output_node = bpy.context.scene.node_tree.nodes[2]
#        file_output_node.file_slots[0].path = 'frame-000001.depth' # blender placeholder #
#        
#        depth_file = str(BASE_DIR.joinpath('data').joinpath('frame-{:06}.depth.png'.format(num)))
#        print(depth_file)
#        file_output_node.file_slots[0].path = depth_file
#        
#        bpy.ops.render.render(write_still=True)