import numpy as np
from skimage import measure
import pathlib as PH
import time
import cv2
import matplotlib.pyplot as plt
import os
from tsdf_integration import TSDFVolume, draw_points,find_vertices_correspondence,world_to_voxel


#def nounds metric standard(meter)
vol_bnds = np.array([[-1,5],[-2,2],[0,1]])
voxel_size = 0.1

#set base dir
BASE_DIR = PH.Path(__file__).parent.parent
WORKING_DIR = BASE_DIR.joinpath('data')
MODEL_DIR = BASE_DIR.joinpath('Model')
MODEL_DATA_DIR = MODEL_DIR.joinpath('data')

#get camera intrinsics, obj_scale
cam_intr = np.load(WORKING_DIR.joinpath('camera-intrinsics.npy'))
obj_scale = np.load(WORKING_DIR.joinpath('object_scale.npy'))


#num of images used
n_imgs = 10

for i in range(n_imgs):
    tsdf_vol = TSDFVolume(vol_bnds,voxel_size=voxel_size)
    color_image_path = WORKING_DIR.joinpath('frame-{number:06}.color.png'.format(number = i))
    depth_image_path = WORKING_DIR.joinpath('frame-{number:06}.depth.png'.format(number = i))
    camera_pose = WORKING_DIR.joinpath('frame-camera-{number:06}.pose.npy'.format(number = i))
    object_pose = WORKING_DIR.joinpath('frame-object-{number:06}.pose.npy'.format(number = i))
    vertices = WORKING_DIR.joinpath('frame-object-{number:06}.vertices.npy'.format(number = i))


    color_image = plt.imread(str(color_image_path),-1)
    depth_im = cv2.imread(str(depth_image_path),-1)
    depth_im = plt.imread(str(depth_image_path))
    cam_pose = np.load(str(camera_pose))
    obj_pose = np.load(str(object_pose))
    vertices_x =  np.load(str(vertices))
    vertices_x = (obj_scale[:3,:3] @ np.load(str(vertices)).T).T
    # print(vertices_x[:3,:])

    print(depth_im.max())
    print(depth_im.min())

    # homo = np.ones((1,1))
    # homo_coor = np.concatenate([vertices[:1,:],homo],axis = 1)
    tsdf_vol.integrate(color_image,depth_im,cam_intr,cam_pose,obs_weight=1.)

    vertices_y = find_vertices_correspondence(obj_scale,obj_pose,vertices_x)

    # print(vertices_y[:3,:][0])
    voxel_x = world_to_voxel(voxel_size,vertices_x,tsdf_vol._vol_lower_bounds)
    voxel_y = world_to_voxel(voxel_size,vertices_y,tsdf_vol._vol_lower_bounds)

    # print(voxel_x)
    # print(voxel_y)

    tsdf_vol._tsdf_vol_cpu

    if not os.path.exists(str(MODEL_DATA_DIR)):
        os.mkdir(MODEL_DATA_DIR)

    train_test_data_path = MODEL_DATA_DIR.joinpath('train_test_voxel-{number:06}.npy'.format(number = i))
    train_test_correspondence_path = MODEL_DATA_DIR.joinpath('train_test_correspondence-{number:06}.npy'.format(number = i))
    np.save(train_test_data_path,tsdf_vol._tsdf_vol_cpu)
    np.save(train_test_correspondence_path,np.concatenate([voxel_x,voxel_y],axis = 1))


    # draw_points(color_image_path,tsdf_vol._pix)
    # draw_points(color_image_path,cam_pose,cam_intr,vertices_x,obj_scale)
    # tsdf_vol.find_voxel_correspondence(cam_intr,cam_pose,keypts)




