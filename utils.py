import numpy as np
import open3d as o3d
import cv2
import tensorflow as tf

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

def draw_points(path,RT,cam_intr,vertices_a,vertices_b,num_pts = 10):

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
        cv2.line(img, (cam_pix_a[index[i],0], cam_pix_a[index[i],1]), (cam_pix_b[index[i],0], cam_pix_b[index[i],1]), (0,255,0), lineThickness)
        
        cv2.imshow('img',img)
    cv2.waitKey()

def view_geometry(ply_path,vertices_a,vertices_b,num_pts = 10):
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
    voxel[:,:1] = np.round(voxel[:,:1] + (-bounds[0] / voxel_size)).astype(int)
    voxel[:,1:2] = np.round(voxel[:,1:2] + (-bounds[1] / voxel_size)).astype(int)
    voxel[:,2:3] = np.round(voxel[:,2:3] + (-bounds[2] / voxel_size)).astype(int)
    return voxel



def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am * Bm.T

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t


def RANSAC(Xs, Xd, max_iter, eps):
    
    H = []
    inliers_id = []
    M = -np.inf
    
    for _ in range(max_iter):
        sample_idxs = random.sample(range(Xs.shape[0]), 4)
        src = Xs[sample_idxs, :]
        dst = Xd[sample_idxs, :]

        H_3x3 = compute_homography(src, dst)
        dst_pts_nx2 = apply_homography(Xs, H_3x3)

        err_by_idx = [np.sqrt(
            (dst_pts_nx2[i,0] - Xd[i,0])**2 + 
            (dst_pts_nx2[i, 1] - Xd[i, 1])**2)
            for i in range(Xd.shape[0])]

        inliers_by_idx = [idx for idx, err in enumerate(err_by_idx) if err < eps]

        if len(inliers_by_idx) > M:
            M = len(inliers_by_idx)
            H = H_3x3
            inliers_id = inliers_by_idx

    return inliers_id, H


def draw_points(path,RT,cam_intr,vertices_a,vertices_b,num_pts = 10):

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
        cv2.line(img, (cam_pix_a[index[i],0], cam_pix_a[index[i],1]), (cam_pix_b[index[i],0], cam_pix_b[index[i],1]), (0,255,0), lineThickness)
        
        cv2.imshow('img',img)
    cv2.waitKey()

def view_geometry(ply_path,vertices_a,vertices_b,num_pts = 10):
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

def get_top_10_match(batch,src,descriptor_object,descriptor_package):

    src_des = descriptor_object[batch,src[0],src[1],src[2]]

    x_range = tf.range(descriptor_package.shape[1])
    y_range = tf.range(descriptor_package.shape[2])
    z_range = tf.range(descriptor_package.shape[3])
    X_grid, Y_grid,Z_grid= tf.meshgrid(x_range, y_range,z_range)
    X_grid = tf.reshape(X_grid,[-1,1])
    Y_grid = tf.reshape(Y_grid,[-1,1])
    Z_grid = tf.reshape(Z_grid,[-1,1])
    grid = tf.concat([X_grid,Y_grid,Z_grid],axis = 1)

    descriptor_column = tf.reshape(descriptor_package,(-1, descriptor_package.shape[-1]))    
    diff = tf.reduce_sum((descriptor_column - src_des) ** 2,axis = 1)
    
    #best_match
    src_match = tf.argmin(diff,axis = 0)

    #top 10 match
    top_10_idx = tf.argsort(diff,axis=-1,direction='ASCENDING')[:10]
    top_10_dest = tf.gather(grid,top_10_idx,axis = 0)
    top_10_matching_distance = tf.gather(diff,top_10_idx,axis = 0)[...,None]
    # top_10_dest = tf.dtypes.cast(top_10_dest,'float32')

    return top_10_dest,top_10_matching_distance

def plot_3d_heat_map(src_des,descriptor_package):
    distance_diff = tf.reduce_sum(tf.square((descriptor_package - src_des)),axis = -1)
    print(distance_diff.shape)



