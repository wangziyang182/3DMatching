import numpy as np

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