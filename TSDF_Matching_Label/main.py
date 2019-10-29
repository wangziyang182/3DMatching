import numpy as np
from skimage import measure
import pathlib as PH
import time
import cv2
import matplotlib.pyplot as plt
import os

# import OpenEXR
# try:
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from pycuda.compiler import SourceModule
#     FUSION_GPU_MODE = 1
# except Exception as err:
#     print('Warning: %s'%(str(err)))
#     print('Failed to import PyCUDA. Running fusion in CPU mode.')
#     FUSION_GPU_MODE = 0


FUSION_GPU_MODE = 0

class TSDFVolume(object):

    def __init__(self,vol_bnds,voxel_size):

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds # rows: x,y,z columns: min,max in world coordinates in meters
        self._voxel_size = voxel_size # in meters (determines volume discretization and resolution)
        self._trunc_margin = self._voxel_size*5 # truncation on SDF

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int) # ensure C-order contigous
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_lower_bounds = self._vol_bnds[:,0].copy(order='C').astype(np.float32) # ensure C-order contigous

        print("Voxel volume size: %d x %d x %d"%(self._vol_dim[0],self._vol_dim[1],self._vol_dim[2]))

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32) # for computing the cumulative moving average of observations per voxel
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Copy voxel volumes to GPU
        if FUSION_GPU_MODE:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)

            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
              __global__ void integrate(float * tsdf_vol,
                                        float * weight_vol,
                                        float * color_vol,
                                        float * vol_dim,
                                        float * vol_origin,
                                        float * cam_intr,
                                        float * cam_pose,
                                        float * other_params,
                                        float * color_im,
                                        float * depth_im) {

                // Get voxel index
                int gpu_loop_idx = (int) other_params[0];
                int max_threads_per_block = blockDim.x;
                int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
                int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
                
                int vol_dim_x = (int) vol_dim[0];
                int vol_dim_y = (int) vol_dim[1];
                int vol_dim_z = (int) vol_dim[2];

                if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
                    return;

                // Get voxel grid coordinates (note: be careful when casting)
                float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
                float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
                float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);

                // Voxel grid coordinates to world coordinates
                float voxel_size = other_params[1];
                float pt_x = vol_origin[0]+voxel_x*voxel_size;
                float pt_y = vol_origin[1]+voxel_y*voxel_size;
                float pt_z = vol_origin[2]+voxel_z*voxel_size;

                // World coordinates to camera coordinates
                float tmp_pt_x = pt_x-cam_pose[0*4+3];
                float tmp_pt_y = pt_y-cam_pose[1*4+3];
                float tmp_pt_z = pt_z-cam_pose[2*4+3];
                float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
                float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
                float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;

                // Camera coordinates to image pixels
                int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
                int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);

                // Skip if outside view frustum
                int im_h = (int) other_params[2];
                int im_w = (int) other_params[3];
                if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
                    return;

                // Skip invalid depth
                float depth_value = depth_im[pixel_y*im_w+pixel_x];
                if (depth_value == 0)
                    return;

                // Integrate TSDF
                float trunc_margin = other_params[4];
                float depth_diff = depth_value-cam_pt_z;
                if (depth_diff < -trunc_margin)
                    return;
                float dist = fmin(1.0f,depth_diff/trunc_margin);
                float w_old = weight_vol[voxel_idx];
                float obs_weight = other_params[5];
                float w_new = w_old + obs_weight;
                weight_vol[voxel_idx] = w_new;
                tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+dist)/w_new;

                // Integrate color
                float old_color = color_vol[voxel_idx];
                float old_b = floorf(old_color/(256*256));
                float old_g = floorf((old_color-old_b*256*256)/256);
                float old_r = old_color-old_b*256*256-old_g*256;
                float new_color = color_im[pixel_y*im_w+pixel_x];
                float new_b = floorf(new_color/(256*256));
                float new_g = floorf((new_color-new_b*256*256)/256);
                float new_r = new_color-new_b*256*256-new_g*256;
                new_b = fmin(roundf((old_b*w_old+new_b)/w_new),255.0f);
                new_g = fmin(roundf((old_g*w_old+new_g)/w_new),255.0f);
                new_r = fmin(roundf((old_r*w_old+new_r)/w_new),255.0f);
                color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;

              }""")

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
            self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))


    def integrate(self,color_im,depth_im,cam_intr,cam_pose,obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[:,:,2]*256*256+color_im[:,:,1]*256+color_im[:,:,0])

        # GPU mode: integrate voxel volume (calls CUDA kernel)
        if FUSION_GPU_MODE:
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     cuda.InOut(self._vol_dim.astype(np.float32)),
                                     cuda.InOut(self._vol_origin.astype(np.float32)),
                                     cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                     cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                     cuda.InOut(np.asarray([gpu_loop_idx,self._voxel_size,im_h,im_w,self._trunc_margin,obs_weight],np.float32)),
                                     cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                     cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                     block=(self._max_gpu_threads_per_block,1,1),grid=(int(self._max_gpu_grid_dim[0]),int(self._max_gpu_grid_dim[1]),int(self._max_gpu_grid_dim[2])))

        # CPU mode: integrate voxel volume (vectorized implementation)
        else:

            # Get voxel grid coordinates
            xv,yv,zv = np.meshgrid(range(self._vol_dim[0]),range(self._vol_dim[1]),range(self._vol_dim[2]),indexing='ij')
            vox_coords = np.concatenate((xv.reshape(1,-1),yv.reshape(1,-1),zv.reshape(1,-1)),axis=0).astype(int)


            # Voxel coordinates to world coordinates
            # world_pts = self._vol_lower_bounds.reshape(-1,1)+vox_coords.astype(float)*self._voxel_size

            # # World coordinates to camera coordinates
            # world2cam = np.linalg.inv(cam_pose)

            # ### delete later
            # cam_pts = np.dot(world2cam[:3,:3],world_pts)+np.tile(world2cam[:3,3].reshape(3,1),(1,world_pts.shape[1]))

            # world_pts_homo = np.concatenate([world_pts,np.ones((1,world_pts.shape[1]))],axis = 0)

            # cam_pts = (world2cam @ world_pts_homo)[:3,:]

            # pix = np.round((cam_intr @ cam_pts / (cam_intr @ cam_pts)[2,:])).astype(int)[:2,:]

            # pix_x = pix[0,:]
            # pix_y = pix[1,:]
            

            world_pts = self._vol_lower_bounds.reshape(-1,1)+vox_coords.astype(float)*self._voxel_size

            # World coordinates to camera coordinates
            world2cam = np.linalg.inv(cam_pose)
            cam_pts = np.dot(world2cam[:3,:3],world_pts)+np.tile(world2cam[:3,3].reshape(3,1),(1,world_pts.shape[1]))

            # Camera coordinates to image pixels
            pix_x = np.round(cam_intr[0,0]*(cam_pts[0,:]/cam_pts[2,:])+cam_intr[0,2]).astype(int)
            pix_y = np.round(cam_intr[1,1]*(cam_pts[1,:]/cam_pts[2,:])+cam_intr[1,2]).astype(int)

            print('cam_pts',pix_x)
            print('cam_pts',pix_y)

            # Skip if outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                        np.logical_and(pix_x < im_w,
                        np.logical_and(pix_y >= 0,
                        np.logical_and(pix_y < im_h,
                                       cam_pts[2,:] < 0))))

            print('valid_pix',np.sum(valid_pix))

            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix],pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val-cam_pts[2,:]

            print(cam_pts)

            valid_pts = np.logical_and(depth_val > 0,depth_diff >= -self._trunc_margin)
            dist = np.minimum(1.,np.divide(depth_diff,self._trunc_margin))
            w_old = self._weight_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]]
            w_new = w_old + obs_weight
            self._weight_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = w_new
            tsdf_vals = self._tsdf_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]]
            self._tsdf_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = np.divide(np.multiply(tsdf_vals,w_old)+dist[valid_pts],w_new)

            # Integrate color
            old_color = self._color_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]]
            old_b = np.floor(old_color/(256.*256.))
            old_g = np.floor((old_color-old_b*256.*256.)/256.)
            old_r = old_color-old_b*256.*256.-old_g*256.
            new_color = color_im[pix_y[valid_pts],pix_x[valid_pts]]
            new_b = np.floor(new_color/(256.*256.))
            new_g = np.floor((new_color-new_b*256.*256.)/256.)
            new_r = new_color-new_b*256.*256.-new_g*256.
            new_b = np.minimum(np.round(np.divide(np.multiply(old_b,w_old)+new_b,w_new)),255.);
            new_g = np.minimum(np.round(np.divide(np.multiply(old_g,w_old)+new_g,w_new)),255.);
            new_r = np.minimum(np.round(np.divide(np.multiply(old_r,w_old)+new_r,w_new)),255.);
            self._color_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = new_b*256.*256.+new_g*256.+new_r;

def get_view_frustum(depth_im,cam_intr,cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
                               (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
                                np.array([0,max_depth,max_depth,max_depth,max_depth])])
    view_frust_pts = np.dot(cam_pose[:3,:3],view_frust_pts)+np.tile(cam_pose[:3,3].reshape(3,1),(1,view_frust_pts.shape[1])) # from camera to world coordinates
    return view_frust_pts

def find_vertices_correspondence(scale,object_pose,vertices):
    object_pose = np.linalg.inv(scale) @ object_pose
    scale = np.array([scale[0][0],scale[1][1],scale[2][2]])[...,None]
    object_pose[:3,3:4] = object_pose[:3,3:4] * scale
    y = object_pose @ np.concatenate([vertices,np.ones((vertices.shape[0],1))],axis = 1).T
    return y.T[:,:3]

def world_to_voxel(voxel_size,vertices,bounds):
    print(bounds)
    voxel = np.round(vertices/ voxel_size)
    voxel[:,:1] = voxel[:,:1] + np.round((-bounds[0] / voxel_size)).astype(int)
    voxel[:,1:2] = voxel[:,1:2] +np.round((-bounds[1] / voxel_size)).astype(int)
    voxel[:,2:3] = voxel[:,2:3] +np.round((-bounds[2] / voxel_size)).astype(int)
    return voxel


def draw_points(path,cam_pose,cam_intr,vertices,obj_scale):

    # World coordinates to camera coordinates
    world2cam = np.linalg.inv(cam_pose)
    # print(world2cam[:3,:])
    # print(vertices.shape)
    # print(cam_pose)
    # print(cam_intr)

    world_pts_homo = np.concatenate([vertices,np.ones((vertices.shape[0],1))],axis = 1)

    cam_pts = (world2cam @ world_pts_homo.T)[:3,:]

    pix = np.round((cam_intr @ cam_pts / (cam_intr @ cam_pts)[2,:])).astype(int)[:2,:]

    pix = pix.T


    # K = cam_intr @ world2cam[:3,:]
    # vertices_homo = np.concatenate([vertices,np.ones((vertices.shape[0],1))],axis = 1)
    # pix = (K @ vertices_homo.T).T
    # pix = np.round(pix[:,:2] / pix[:,2:3]).astype('int')

    img = cv2.imread(str(path),cv2.IMREAD_COLOR)
    for item in pix:
        cv2.circle(img,(item[0],item[1]), 1, (0,255,0), -1)
    cv2.imshow('img',img)
    cv2.waitKey(0)





if __name__ == '__main__':


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


    #TSDF fusion class

    #num of images need to be fused
    n_imgs = 1
    
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




