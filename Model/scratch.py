import matplotlib.pyplot as plt
import pathlib as ph
import numpy as np
# import OpenEXR as exr

BASE_PATH = ph.Path(__file__)

DATA_PATH = BASE_PATH.parent.parent.joinpath('data')

#check depth map
all_png = list(DATA_PATH.glob('**/*.png'))
print(all_png)

depth = all_png[-1]
file = plt.imread(str(depth))

print('path',str(depth))
print('depth_shape',file.shape)
print('depth_max',file.max())
print('depth_min',file.min())

#check intrinsics
cam_intr = np.load(DATA_PATH.joinpath('camera-intrinsics.npy'))
print('cam_intr',cam_intr)

#check vertices
all_npy = list(DATA_PATH.glob('**/*.vertices.npy'))
vertices = np.load(all_npy[0])
print('vertices',vertices)