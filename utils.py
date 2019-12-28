import numpy as np
import open3d as o3d
import cv2
import tensorflow as tf
from numpy.linalg import inv
import colorsys
from matplotlib.colors import hsv_to_rgb



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

def view_geometry(ply_path,vertices_a,vertices_b,num_pts = 10,show_match = True):
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
    
    if show_match:
        o3d.visualization.draw_geometries([pcd,line_set])
    else:
        o3d.visualization.draw_geometries([pcd])


def get_top_match(batch,src,descriptor_object,descriptor_package,dest):

    src_des = descriptor_object[batch,src[0],src[1],src[2]]

    x_range = tf.range(descriptor_package.shape[1])
    y_range = tf.range(descriptor_package.shape[2])
    z_range = tf.range(descriptor_package.shape[3])
    X_grid, Y_grid,Z_grid= np.meshgrid(x_range, y_range,z_range, indexing='ij')

    X_grid = tf.reshape(X_grid,[-1,1])
    Y_grid = tf.reshape(Y_grid,[-1,1])
    Z_grid = tf.reshape(Z_grid,[-1,1])

    grid = tf.concat([X_grid,Y_grid,Z_grid],axis = 1)

    descriptor_column = np.reshape(descriptor_package[batch],(-1, descriptor_package.shape[-1])) 

    descriptor_column_combine = np.concatenate([grid,descriptor_column],axis = 1)

    diff = np.sqrt(np.sum((descriptor_column - src_des) ** 2,axis = 1))
    
    #best_match
    src_match = tf.argmin(diff)
    #top 10 match
    top_idx = tf.argsort(diff,direction='ASCENDING')[:10]
    top_best = tf.gather(grid,top_idx,axis = 0)
    top_matching_distance = tf.gather(diff,top_idx,axis = 0)[...,None]

    return top_best,top_matching_distance,top_idx


def plot_3d_heat_map(batch,src,dest,descriptor_object,descriptor_package,top_idx):

    src_des = descriptor_object[batch,src[0],src[1],src[2]]

    distance_diff = tf.sqrt(tf.reduce_sum(tf.square((descriptor_package[batch] - src_des)),axis = -1))
    print('average_loss',tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(distance_diff)))))

    #generate grid
    x_range = tf.range(descriptor_package.shape[1])
    y_range = tf.range(descriptor_package.shape[2])
    z_range = tf.range(descriptor_package.shape[3])
    X_grid, Y_grid,Z_grid= np.meshgrid(x_range, y_range,z_range, indexing='ij')
    X_grid = tf.reshape(X_grid,[-1,1])
    Y_grid = tf.reshape(Y_grid,[-1,1])
    Z_grid = tf.reshape(Z_grid,[-1,1])
    grid = tf.concat([X_grid,Y_grid,Z_grid],axis = 1)

    #get the 
    distance_diff_column = tf.reshape(distance_diff,(-1, 1))

    heatmap_rgb = to_rgb(distance_diff_column)
    radius = calculate_radius(distance_diff_column,raidus_min = 0.2, raidus_max = 1)

    #draw sphere one by one
    object_list = visualize_3D_heatmap(grid,heatmap_rgb,radius,src,dest,top_idx)

    # #draw point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid)
    pcd.colors = o3d.utility.Vector3dVector(heatmap_rgb)
    object_list.append(pcd)

    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.5)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.paint_uniform_color([0, 0, 0])
    
    # mesh_sphere = mesh_sphere.translate(np.array([[dest[0]],[dest[1]],[dest[2]]]))
    # object_list.append(mesh_sphere)

    return object_list
    # o3d.visualization.draw_geometries(object_list)

def visualize_3D_heatmap(locations,color,radius,src,dest,top_idx):

    # locations = locations[top_idx,:]
    # color = color[top_idx,:]
    # raidus = raidus[top_idx]
    object_list = []
    #draw ground truth
    ground_truth = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    # ground_truth.compute_vertex_normals()
    ground_truth.paint_uniform_color([0, 0, 0])
    ground_truth = ground_truth.translate(np.array([[dest[0]],[dest[1]],[dest[2]]]))
    object_list.append(ground_truth)

    obj = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    obj.paint_uniform_color([0, 0, 0])
    obj = obj.translate(np.array([[src[0] - 30],[src[1]],[src[2]]]))
    object_list.append(obj)

    for idx in top_idx:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius[idx])
         # mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color[idx,:])
        mesh_sphere = mesh_sphere.translate(locations[idx])
        object_list.append(mesh_sphere)
        
    return object_list
    # o3d.visualization.draw_geometries(object_list)

def visualize_ground_truth(batch,src,dest,descriptor_object,descriptor_package,top_idx,ply_test,shift):
    
    object_list = plot_3d_heat_map(batch,src,dest,descriptor_object,descriptor_package,top_idx)
    pcd = o3d.io.read_point_cloud(ply_test[batch])
    transformation_matrix = np.eye(4) * 10
    transformation_matrix[-1,-1] = 1
    pcd.transform(transformation_matrix)
    pcd.translate([[10],[20],[0]])
    pcd.translate(np.array(-shift))
    object_list.append(pcd)
    # mesh = compute_mesh(pcd,None)
    # object_list.append(mesh)

    o3d.visualization.draw_geometries(object_list)




def compute_mesh(ply_object,displacement):
    # ply_object.translate(displacement)
    ply_object.estimate_normals()
    distances = ply_object.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           ply_object,o3d.utility.DoubleVector([radius, radius * 2]))
    
    # trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),vertex_normals=np.asarray(mesh.vertex_normals)))

    return mesh


def calculate_radius(distance_diff_column,raidus_min = 0.1, raidus_max = 0.4):
    
    A = np.array([[np.amin(distance_diff_column),1],[np.amax(distance_diff_column),1]])
    b = np.array([[raidus_max],[raidus_min]])
    x = inv(A) @ b
    radius = np.concatenate([distance_diff_column,np.ones(distance_diff_column.shape)],axis = 1) @ x

    return radius

def calculate_angle(distance_diff_column,angle_min = 0, angle_max = 240/360):
    
    A = np.array([[np.amin(distance_diff_column),1],[np.amax(distance_diff_column),1]])
    b = np.array([[angle_min],[angle_max]])
    x = inv(A) @ b
    angle = np.concatenate([distance_diff_column,np.ones(distance_diff_column.shape)],axis = 1) @ x

    return angle


def to_rgb(values):
    minimum = np.min(values,axis= 0)
    maximum = np.max(values,axis = 0)
    
    heatmap_rgb = np.zeros((values.shape[0],3))
    ratio = 2 * (values-minimum) / (maximum - minimum)

    #R
    heatmap_rgb[:,0:1] = np.maximum(0, 255*(ratio - 1)) / 255
    #B
    heatmap_rgb[:,2:3] = np.maximum(0, 255*(1 - ratio)) / 255
    #G
    heatmap_rgb[:,1:2] = 1 -  heatmap_rgb[:,0:1] - heatmap_rgb[:,2:3]

    angle = calculate_angle(values)
    hsv = np.concatenate([angle,np.ones(angle.shape) ,np.ones(angle.shape)],axis = 1)
    heatmap_rgb = hsv_to_rgb(hsv)
    # print('rgb',rgb)
    return heatmap_rgb


