# import numpy
# from stl import mesh
# import os
# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
# import pathlib 

# BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
# # print(pathlib.Path(__file__).parent.parent)
# TEMPLATES_DIR = BASE_DIR.joinpath('env').joinpath('mesh')

# print('TEMPLATES_DIR',TEMPLATES_DIR)
# all_stl = list(TEMPLATES_DIR.glob('**/*.stl'))
# all_stl = [str(path) for path in all_stl]

# print(all_stl[2])
# your_mesh = mesh.Mesh.from_file(all_stl[2])


# print(your_mesh.vectors.shape)
# assert (your_mesh.points[0][0:3] == your_mesh.v0[0]).all()

# # figure = pyplot.figure()
# # axes = mplot3d.Axes3D(figure)
# # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
# # scale = your_mesh.points.flatten(-1)
# # axes.auto_scale_xyz(scale, scale, scale)
# # pyplot.show()






# # class OurClass:

# #     def __init__(self, a):
# #         self.OurAtt = a

# #     @property
# #     def OurAtt(self):
# #         return self.__OurAtt

# #     @OurAtt.setter
# #     def OurAtt(self, val):
# #         if val < 0:
# #             self.__OurAtt = 0
# #         elif val > 1000:
# #             self.__OurAtt = 1000
# #         else:
# #             self.__OurAtt = val


# # x = OurClass(10)
# # print(x.OurAtt)

# # import numpy as np

# # a = np.random.randint(0,20,30)
# # print(a)
# # print(a[:5])
# # print(a[-5:])



# import numpy as np

# print((19 ** 0.5)/2)
 
# import scipy.linalg as lingalg


# print(lingalg.det(np.array([[1,2,3,0],[0,0,4,2020],[-1,2,-3,4],[1,2,3,4]])))


# a = np.array([[-0.27,0.1,0.25], [0.02,-0.2,0.02],[1.25,1.02,0.73]])
# b = np.array([0,0,1])
# x = np.linalg.solve(a, b)
# print(x)



# #q2
# import random
# def sortArray(nums):
#     if len(nums) == 1:
#         return nums
#     if not nums:
#         return []
#     pick = random.choice(nums)
#     left, mid, right = [], [], []
#     for i in nums:
#         if i < pick:
#             left.append(i)
#         elif i > pick:
#             right.append(i)
#         else:
#             mid.append(i)
#     return sortArray(left) + mid + sortArray(right)

# def sortArray_reverse(nums):
#     if len(nums) == 1:
#         return nums
#     if not nums:
#         return []
#     pick = random.choice(nums)
#     left, mid, right = [], [], []
#     for i in nums:
#         if i < pick:
#             right.append(i)
#         elif i > pick:
#             left.append(i)
#         else:
#             mid.append(i)
#     return sortArray(left) + mid + sortArray(right)


# string = ['3','4','6','1','8','9','5','2','7','0']

# int_string = [int(item) for item in string]

# code = ['990','337','103','103','334','33334','34']

# dict_code_string = {}

# all_num = []
# for item in code:
#     number =''
#     for char in item:
        
#         index = int_string.index(int(char))
#         number += str(index)
#     try:
#         dict_code_string[int(number)].append(item)
#     except:
#         dict_code_string[int(number)] = [item]
#     all_num.append(int(number))

# all_num = sortArray(all_num)
# final_string = []

# for key in set(all_num):
#     final_string += dict_code_string[key]

# print(final_string)




# string = 'ABBCZBAC'
# string = [i for i in string]
# print(string)
# count = 0
# # if len(string) <=2:
# #     return count
# for i in range(len(string)):
#     flag_A = False
#     flag_B = False
#     flag_C = False
#     if string[i] == 'A':
#         flag_A = True
#     if string[i] == 'B':
#         flag_B = True
#     if string[i] == 'C':
#         flag_C = True
#     for j in range(i + 1,len(string)):
#         if string[j] == 'A':
#             flag_A = True
#         if string[j] == 'B':
#             flag_B = True
#         if string[j] == 'C':
#             flag_C = True

#         if flag_A == True and flag_B == True and flag_C == True:
#             count += (len(string) - j)
#             break

# print((16/25 * 0.01 + 1/25 * 0.04) ** 0.5)



# string = 'hurart'

# rotation =[1,1,1,1,1,1,1]
# amount = [10,3,5,9,2,10,3]

# string = [item for item in string]
# length = len([item for item in string])
# move = 0
# for i in range(len(rotation)):

#     if rotation[i] == 0:
#         move -= amount[i]
#         move = move % (-length)
#     if rotation[i] == 1 :
#         move += amount[i]
#         move = move % length

# print(move)

# if move >= 0:
#     string_return = string[(length - move):] + string[:(length - move)]
# if move < 0:
#     print('hahaha')
#     move = -move
#     print(string)
#     print(length - move)
#     string_return = string[move:] + string[:move]

#     print(''.join(string_return))

# print(string_return)



# grid = [[0,1,1,0],[1,1,0,0]]
# k = 3
# rules = [1,2]
# # def gridGame(grid, k, rules):

# live_nb = [[0]*len(grid[i]) for i in range(len(grid))]


# flag = True
# for i in range(len(grid)):
#     grid[i].insert(0,0)
#     grid[i].insert(-1,0)
# grid.insert(0,[0] * len(grid[0]))
# grid.insert(-1,[0] * len(grid[0]))
# for i in range(1,len(grid)-2):
#     for j in range(1,len(grid[i])-2):
#         print(i,j)
#         for k in range(i-1,i+1):
#             for z in range(j-1,j+1):
#                 if grid[k][z] == 1:
#                     live_nb[i][j]+= 1

# print(live_nb)


# import bpy
# import bpy_extras

# scene = bpy.context.scene
# obj = bpy.context.object
# co = bpy.context.scene.cursor_location

# co_2d = bpy_extras.object_utils.world_to_camera_view(scene, obj, co)
# print("2D Coords:", co_2d)

# # If you want pixel coords
# render_scale = scene.render.resolution_percentage / 100
# render_size = (
#         int(scene.render.resolution_x * render_scale),
#         int(scene.render.resolution_y * render_scale),
#         )
# print("Pixel Coords:", (
#       round(co_2d.x * render_size[0]),
#       round(co_2d.y * render_size[1]),
#       ))



# from scipy.special import comb
# import numpy as np

# # x = np.array([[1,-1],[0,1]])
# # print(x ** 10000001)
# # print(13**0.5)
# a = 0
# intt = [0,2,4,6,8,10]
# for i in intt:
#     print(i)
#     a += (comb(10,i) * (0.1 ** i) * (0.9 ** (10-i)))

# print(a)
# print(1 - (a * 0.5**10))

# a = [1,1,1,1,1,1,1,1]
# string = []
# if len(a) < 7:
#     print('nonono')
# for i in range(6,len(a)):
#     b = sum(a[i - 6:i + 1])/7
#     string.append('%.2f'%b)
# print(string)

# import scipy.linalg as ling
# import numpy as np

# A = np.array([[1,2],[3,4]])
# B = np.array([1,2])
# print(ling.inv(A)@B)


import numpy as np
from skimage import measure
import pathlib as PH
import numpy as np
import cv2
import time
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image

class TSDFVolume(object):

    def __init__(self,vol_bnds,voxel_size):

        self._vol_bnds = vol_bnds

        self._voxel_size = voxel_size

        self._trunc_margin = self._voxel_size * 5

        self._vol_dim = np.ceil((self._vol_bnds[:,1] - self._vol_bnds[:,0]) / self._voxel_size).copy(order = 'C').astype(int)

        self._vol_origin = self._vol_bnds[:,0].copy(order = 'C').astype(np.float32)

        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        self._weight_vol_cpu = np.ones(self._vol_dim).astype(np.float32)

        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)


    def integrate(self, color_im,depth_im,cam_intr,cam_pose,obs_weight = 1.):

        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]

        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[:,:,2]*256 * 256 + color_im[:,:,1] * 256 + color_im[:,:,0])

        xv,yv,zv = np.meshgrid(range(self._vol_dim[0]),range(self._vol_dim[1]),range(self._vol_dim[2]),indexing = 'ij')
        vox_coords = np.stack([xv.flatten(),yv.flatten(),zv.flatten()])
        world_pts = self._vol_origin.reshape(-1,1) + vox_coords.astype(float) * self._voxel_size

        # world coordinates to camera coordinates
        world2cam = np.linalg.inv(cam_pose)
        cam_pts = np.dot(world2cam[:3,:3],world_pts) + np.tile(world2cam[:3,3].reshape(3,1),(1,world_pts.shape[1]))

        pix_x = np.round(cam_intr[0,0]*(cam_pts[0,:]/cam_pts[2,:])+cam_intr[0,2]).astype(int)
        pix_y = np.round(cam_intr[1,1]*(cam_pts[1,:]/cam_pts[2,:])+cam_intr[1,2]).astype(int)

        # print(cam_pts)
        valid_pix = np.logical_and(pix_x >= 0,
                        np.logical_and(pix_x < im_w,
                        np.logical_and(pix_y >= 0,
                        np.logical_and(pix_y < im_h,
                                       cam_pts[2,:] > 0))))


        depth_val = np.zeros(pix_x.shape) - 1
        depth_val[valid_pix] = depth_im[pix_y[valid_pix],pix_x[valid_pix]]


        # Integrate TSDF
        depth_diff = depth_val - cam_pts[2,:]
        
        zero_distance_point = np.where((depth_diff == 0) & (depth_diff > -1))

        print(np.sum(zero_distance_point))
        
        valid_pts = np.logical_and(depth_val > 0,depth_diff >= -self._trunc_margin)

        dist = np.minimum(1.,np.divide(depth_diff,self._trunc_margin))
        
        w_old = self._weight_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]]
        
        w_new = w_old + obs_weight

        self._weight_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = w_new
        
        tsdf_vals = self._tsdf_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]]
        
        self._tsdf_vol_cpu[vox_coords[0,valid_pts],vox_coords[1,valid_pts],vox_coords[2,valid_pts]] = np.divide(np.multiply(tsdf_vals,w_old)+dist[valid_pts],w_new)

    def find_voxel_correspondence(self,cam_intr,cam_pose,keypts):
        keypts = np.array(keypts).T

        cam_extr = np.zeros((4,4))
        cam_extr[:3,:3] = np.linalg.inv(cam_pose[:3,:3])
        cam_extr[:3,3:] = -np.linalg.inv(cam_pose[:3,:3])@cam_pose[:3,3:]
        cam_extr[3,3] = 1
        
        camera_mat = np.concatenate((cam_intr@cam_extr[:3,:],np.zeros((1,4))),axis = 0)
        camera_mat[3,3] = 1
        

        world2cam = np.linalg.inv(cam_pose)

        xv,yv,zv = np.meshgrid(range(self._vol_dim[0]),range(self._vol_dim[1]),range(self._vol_dim[2]),indexing = 'ij')
        vox_coords = np.stack([xv.flatten(),yv.flatten(),zv.flatten()])
        world_pts = self._vol_origin.reshape(-1,1) + vox_coords.astype(float) * self._voxel_size

        cam_pts = np.dot(world2cam[:3,:3],[[0],[0],[0]]) + world2cam[:3,3].reshape(3,1)

        pix_x = np.round(cam_intr[0,0]*(cam_pts[0,:]/cam_pts[2,:])+cam_intr[0,2]).astype(int)
        pix_y = np.round(cam_intr[1,1]*(cam_pts[1,:]/cam_pts[2,:])+cam_intr[1,2]).astype(int)


        #get the camera matrix
        cam_extr = np.zeros((4,4))
        cam_extr[:3,:3] = np.linalg.inv(cam_pose[:3,:3])
        cam_extr[:3,3:] = -np.linalg.inv(cam_pose[:3,:3])@cam_pose[:3,3:]
        cam_extr[3,3] = 1
        camera_mat = np.concatenate((cam_intr@cam_extr[:3,:],np.zeros((1,4))),axis = 0)
        camera_mat[3,3] = 1

        point = np.array([[-12],[144],[1]])
        print('pix',pix_x,pix_y)
        print(np.linalg.inv(cam_intr) @ point)

        a = cam_intr@cam_extr[:3,:]@[[0],[0],[0],[1]]
        a = (a /a[2,0])
        print(a)
        print(camera_mat@[[0],[0],[0],[1]])
        print(np.linalg.inv(camera_mat)@camera_mat@[[0],[0],[0],[1]])

        # np.linalg.inv(cam_extr)[:3,:]@np.linalg.inv(cam_intr)@a


def FLANN_Matcher_draw(img_1,img_2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_1,None)
    kp2, des2 = sift.detectAndCompute(img_2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(color_image_1,kp1,color_image_2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

def genSIFTMatchPairs(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    pts1 = np.zeros((250,2))
    pts2 = np.zeros((250,2))
    for i in range(250):
        pts1[i,:] = kp1[matches[i].queryIdx].pt
        pts2[i,:] = kp2[matches[i].trainIdx].pt
    
    return pts1, pts2, matches[:250], kp1, kp2

def test_matches():
    img1 = cv2.imread('mountain_left.png')
    img2 = cv2.imread('mountain_center.png')

    pts1, pts2, matches, kp1, kp2 = genSIFTMatchPairs(img1, img2)
    
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2, matchColor=(0,0,255))
    plt.imshow(cv2.cvtColor(matching_result, cv2.COLOR_BGR2RGB))



if __name__ == '__main__':
    vol_bnds = np.array([[-1.5,1.5],[-1.5,1.5],[-1.5,1.5]])
    BASE_DIR = PH.Path(__file__).parent.parent
    IMAGE_DIR = BASE_DIR.joinpath('data')
    WORKING_DIR = PH.Path(__file__).parent.parent.joinpath('tool').joinpath('tsdf-fusion-python').joinpath('data')
    cam_intr = np.loadtxt(WORKING_DIR.joinpath('camera-intrinsics.txt'),delimiter = ' ')

    tsdf_vol = TSDFVolume(vol_bnds,voxel_size=0.1)

    n_imgs = 1
    t0_elapse = time.time()


    # image_1 = WORKING_DIR.joinpath('frame-{number:06}.color.jpg'.format(number = 0))
    # image_2 = WORKING_DIR.joinpath('frame-{number:06}.color.jpg'.format(number = 80))

    image_1_depth = IMAGE_DIR.joinpath('frame-{number:06}.depth.png'.format(number = 0))
    image_2_depth = IMAGE_DIR.joinpath('frame-{number:06}.depth.png'.format(number = 0))

    image_1_color = IMAGE_DIR.joinpath('frame-{number:06}.color.jpg'.format(number = 0))
    image_2_color = IMAGE_DIR.joinpath('frame-{number:06}.color.jpg'.format(number = 0))

    depth_image_1 = cv2.imread(str(image_1_depth)) 
    depth_image_2 = cv2.imread(str(image_2_depth))

    color_image_1 = cv2.imread(str(image_1_color)) 
    color_image_2 = cv2.imread(str(image_2_color))

    # plt.imshow(np.concatenate((color_image_1,color_image_2),axis = 1))
    # plt.show()

    print(color_image_1.max())
    print(color_image_2.min())

    img = cv2.imread(str(image_1_color)) 

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    # sift = cv2.xfeatures2d.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(color_image_1, None)
    # kp2, des2 = sift.detectAndCompute(color_image_2, None)

    # print(kp1)
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key = lambda x:x.distance)
    
    # pts1 = np.zeros((20,2))
    # pts2 = np.zeros((20,2))
    # for i in range(20):
    #     pts1[i,:] = kp1[matches[i].queryIdx].pt
    #     pts2[i,:] = kp2[matches[i].trainIdx].pt

    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2, matchColor=(0,0,255))
    plt.imshow(cv2.cvtColor(matching_result, cv2.COLOR_BGR2RGB))


    for i in range(n_imgs):
        color_image_path = WORKING_DIR.joinpath('frame-{number:06}.color.jpg'.format(number = i))
        depth_image_path = WORKING_DIR.joinpath('frame-{number:06}.depth.png'.format(number = i))
        pose_path = WORKING_DIR.joinpath('frame-{number:06}.pose.txt'.format(number = i))
        color_image = cv2.cvtColor(cv2.imread(str(color_image_path),-1),cv2.COLOR_BGR2RGB)

        depth_im = cv2.imread(str(depth_image_path) , -1) /1000.
        cam_pose = np.loadtxt(str(pose_path),delimiter = ' ')

        tsdf_vol.integrate(color_image,depth_im,cam_intr,cam_pose,obs_weight=1.)

        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(grey_image,None)
        keypts = [item.pt for item in kp1][:20]
        img = cv2.drawKeypoints(color_image, kp1,color_image)


        # plt.imshow(img)
        # plt.show()
        tsdf_vol.find_voxel_correspondence(cam_intr,cam_pose,keypts)




