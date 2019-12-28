import numpy as np
from Config import Config
import pathlib as PH
import time
import cv2
import matplotlib.pyplot as plt
import os
from tsdf_integration import TSDFVolume
import sys
sys.path.append('..')
from utils import draw_points,find_vertices_correspondence,world_to_voxel,meshwrite,view_geometry



#init hyperparameter
config = Config()

#def nounds metric standard(meter)
vol_bnds = config.vol_bnds
#voxel size in terms of meter 
voxel_size = config.voxel_size
n_imgs = config.n_imgs
#single or mutiple images for tsdf_fusion
multi_images = config.multi_images



#set base dir
BASE_DIR = PH.Path(__file__).parent.parent
WORKING_DIR = BASE_DIR.joinpath('data')
MODEL_DIR = BASE_DIR.joinpath('Model')
MODEL_DATA_DIR = MODEL_DIR.joinpath('data')
if not os.path.exists(MODEL_DATA_DIR):
    os.mkdir(MODEL_DATA_DIR)

#get camera intrinsics, obj_scale
cam_intr = np.load(WORKING_DIR.joinpath('camera-intrinsics.npy'))

if not os.path.exists(BASE_DIR.joinpath('env').joinpath('tsdf_projected_ply')):
    os.mkdir(BASE_DIR.joinpath('env').joinpath('tsdf_projected_ply'))
mesh_save_path = BASE_DIR.joinpath('env').joinpath('tsdf_projected_ply')
vol_dim_path = MODEL_DATA_DIR.joinpath('vol_dim.npy')

if multi_images:
    tsdf_vol = TSDFVolume(vol_bnds,voxel_size=voxel_size)
    np.save(vol_dim_path,tsdf_vol._vol_dim)


#load vertices and ponits inside mesh
vertices = WORKING_DIR.joinpath('frame-object.vertices.npy')
inside_pts = WORKING_DIR.joinpath('frame-object.inside_pts.npy')

vertices_x =  np.load(str(vertices))
inside_pts_x = np.load(str(inside_pts))

# for i in range(n_imgs):
for i in range(1):

    if not multi_images:
        tsdf_vol = TSDFVolume(vol_bnds,voxel_size=voxel_size)
    np.save(vol_dim_path,tsdf_vol._vol_dim)
    color_image_path = WORKING_DIR.joinpath('frame-{number:06}.color.png'.format(number = i))
    depth_image_path = WORKING_DIR.joinpath('frame-{number:06}.depth.png'.format(number = i))
    camera_RT = WORKING_DIR.joinpath('frame-camera-{number:06}.RT.npy'.format(number = i))
    object_pose = WORKING_DIR.joinpath('frame-object-{number:06}.pose.npy'.format(number = i))


    color_image = plt.imread(str(color_image_path),-1)
    depth_im = plt.imread(str(depth_image_path)) * 10
    RT = np.load(str(camera_RT))
    obj_pose = np.load(str(object_pose))


    tsdf_vol.integrate(color_image,depth_im,cam_intr,RT,obs_weight=1.)
    inside_pts_y = find_vertices_correspondence(obj_pose,inside_pts_x)

    voxel_x = world_to_voxel(voxel_size,inside_pts_x,tsdf_vol._vol_lower_bounds)
    voxel_y = world_to_voxel(voxel_size,inside_pts_y,tsdf_vol._vol_lower_bounds)

    if not os.path.exists(str(MODEL_DATA_DIR)):
        os.mkdir(MODEL_DATA_DIR)

    train_test_data_path = MODEL_DATA_DIR.joinpath('train_test_voxel-{number:06}.npy'.format(number = i))
    train_test_correspondence_path = MODEL_DATA_DIR.joinpath('train_test_correspondence-{number:06}.npy'.format(number = i))
 
    np.save(train_test_data_path,tsdf_vol._tsdf_vol_cpu)
    np.save(train_test_correspondence_path,np.concatenate([voxel_x,voxel_y],axis = 1))

    #single frame
    if not multi_images:
        mesh_path = mesh_save_path.joinpath('MESH-{number:06}.ply'.format(number = i))
        verts,faces,norms,colors = tsdf_vol.get_mesh()
        meshwrite(mesh_path,verts,faces,norms,colors)

if multi_images:
    mesh_path = mesh_save_path.joinpath('MESH-object.ply'.format(number = i))
    verts,faces,norms,colors = tsdf_vol.get_mesh()
    meshwrite(mesh_path,verts,faces,norms,colors)


##check if mathcing is correct
view_geometry(str(mesh_path),inside_pts_x,inside_pts_y,show_match = False)
# draw_points(color_image_path,RT,cam_intr,inside_pts_x,inside_pts_y)
    # tsdf_vol.find_voxel_correspondence(cam_intr,RT,keypts)





