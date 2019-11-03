import numpy as np
from skimage import measure
import time
import cv2
import numpy as np
import open3d as o3d


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: %s'%(str(err)))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0


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


    def integrate(self,color_im,depth_im,cam_intr,RT,obs_weight=1.):
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
        if FUSION_GPU_MODE:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu,self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu,self._color_vol_gpu)
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


# Get corners of 3D camera view frustum of depth image
def get_view_frustum(depth_im,cam_intr,cam_pose):
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([(np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
                               (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
                                np.array([0,max_depth,max_depth,max_depth,max_depth])])
    view_frust_pts = np.dot(cam_pose[:3,:3],view_frust_pts)+np.tile(cam_pose[:3,3].reshape(3,1),(1,view_frust_pts.shape[1])) # from camera to world coordinates
    return view_frust_pts


# Save 3D mesh to a polygon .ply file
def meshwrite(filename,verts,faces,norms,colors):

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(verts[i,0],verts[i,1],verts[i,2],norms[i,0],norms[i,1],norms[i,2],colors[i,0],colors[i,1],colors[i,2]))
    
    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0],faces[i,1],faces[i,2]))

    ply_file.close()

def draw_points(path,RT,cam_intr,vertices_a,vertices_b,num_pts = 5):

    '''
    test if 3d points transform to pxiel is correct
    '''
    world_pts_homo_a = np.concatenate([vertices_a,np.ones((vertices_a.shape[0],1))],axis = 1)
    cam_pix_a = cam_intr @ RT @ world_pts_homo_a.T
    cam_pix_a = np.round((cam_pix_a[:2,:] / cam_pix_a[2,:]).T).astype('int')
    
    world_pts_homo_b = np.concatenate([vertices_b,np.ones((vertices_b.shape[0],1))],axis = 1)

    cam_pix_b = cam_intr @ RT @ world_pts_homo_b.T
    cam_pix_b = np.round((cam_pix_b[:2,:] / cam_pix_b[2,:]).T).astype('int')


    img = cv2.imread(str(path),cv2.IMREAD_COLOR)
    index = np.random.choice(len(vertices_a),num_pts,replace = False)

    for item in cam_pix_a[index,:]:
        cv2.circle(img,(item[0],item[1]), 1, (0,255,0), -1)
    for item in cam_pix_b[index,:]:
        cv2.circle(img,(item[0],item[1]), 1, (0,255,0), -1)

    for i in range(cam_pix_a[index,:].shape[0]):
        lineThickness = 1
        cv2.line(img, (cam_pix_a[i,0], cam_pix_a[i,1]), (cam_pix_b[i,0], cam_pix_b[i,1]), (0,255,0), lineThickness)
        
        cv2.imshow('img',img)
    cv2.waitKey()

def view_geometry(ply_path,vertices_a,vertices_b,num_pts = 5):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    #random sample
    index = np.random.choice(len(vertices_a),num_pts,replace = False)
    vertices_a = vertices_a[index,:]
    vertices_b = vertices_b[index,:]
    points = np.concatenate([vertices_a,vertices_b],axis = 0)
    lines = [[i, i + len(vertices_a)] for i in range(len(vertices_a))]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd,line_set])


def find_vertices_correspondence(object_pose,vertices):
    y = object_pose @ np.concatenate([vertices,np.ones((vertices.shape[0],1))],axis = 1).T
    return y.T[:,:3]

def world_to_voxel(voxel_size,vertices,bounds):
    voxel = np.round(vertices/ voxel_size)
    voxel[:,:1] = voxel[:,:1] + np.round((-bounds[0] / voxel_size)).astype(int)
    voxel[:,1:2] = voxel[:,1:2] +np.round((-bounds[1] / voxel_size)).astype(int)
    voxel[:,2:3] = voxel[:,2:3] +np.round((-bounds[2] / voxel_size)).astype(int)
    return voxel

