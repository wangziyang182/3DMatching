import numpy as np
from skimage import measure
import time
import numpy as np

class TSDFVolume(object):

    def __init__(self,vol_bnds,voxel_size):

        # Define voxel volume parameters
        # rows: x,y,z columns: min,max in world coordinates in meters
        self._vol_bnds = vol_bnds 
        # in meters (determines volume discretization and resolution)
        self._voxel_size = voxel_size 
        # truncation on SDF
        self._trunc_margin = self._voxel_size*5 

        # Adjust volume bounds
        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int) # ensure C-order contigous
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_lower_bounds = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

        print("Voxel volume size: %d x %d x %d"%(self._vol_dim[0],self._vol_dim[1],self._vol_dim[2]))

        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32) 
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)


    def integrate(self,color_im,depth_im,cam_intr,RT,obs_weight=1.):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[:,:,2]*256*256+color_im[:,:,1]*256+color_im[:,:,0])


        # Get voxel grid coordinates
        xv,yv,zv = np.meshgrid(range(self._vol_dim[0]),range(self._vol_dim[1]),range(self._vol_dim[2]),indexing='ij')
        vox_coords = np.concatenate((xv.reshape(1,-1),yv.reshape(1,-1),zv.reshape(1,-1)),axis=0).astype(int)


        # Voxel coordinates to world coordinates
        world_pts = self._vol_lower_bounds.reshape(-1,1)+vox_coords.astype(float)*self._voxel_size

        world_pts_homo = np.concatenate([world_pts,np.ones((1,world_pts.shape[1]))],axis = 0)

        cam_pts = (RT @ world_pts_homo)[:3,:]

        pix = np.round((cam_intr @ cam_pts / (cam_intr @ cam_pts)[2,:])).astype(int)[:2,:]

        pix_x = pix[0,:]
        pix_y = pix[1,:]

        # Skip if outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < im_w,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < im_h,
                                   cam_pts[2,:] > 0))))

        print('valid_pix',np.sum(valid_pix))

        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix],pix_x[valid_pix]] 

        # Integrate TSDF
        depth_diff = depth_val-cam_pts[2,:]
        
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

    # Copy voxel volume to CPU
    def get_volume(self):
        return self._tsdf_vol_cpu,self._color_vol_cpu


    # Get mesh of voxel volume via marching cubes
    def get_mesh(self):
        tsdf_vol,color_vol = self.get_volume()

        # Marching cubes
        verts,faces,norms,vals = measure.marching_cubes_lewiner(tsdf_vol,level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size+self._vol_lower_bounds # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:,0],verts_ind[:,1],verts_ind[:,2]]
        colors_b = np.floor(rgb_vals/(256*256))
        colors_g = np.floor((rgb_vals-colors_b*256*256)/256)
        colors_r = rgb_vals-colors_b*256*256-colors_g*256
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)
        return verts,faces,norms,colors


