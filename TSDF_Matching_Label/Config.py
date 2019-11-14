import tensorflow as tf
import numpy as np

class Config(object):

    def __init__(self):
        

        #for tsdf intergration
        self._vol_bnds = np.array([[-1,5],[-2,2],[0,1]])
        self._voxel_size = 0.1
        self._n_imgs = 2
        self._multi_images = False


    @property
    def vol_bnds(self):
        return self._vol_bnds

    @property
    def voxel_size(self):
        return self._voxel_size
    
    @property
    def n_imgs(self):
        return self._n_imgs
    
    @property
    def multi_images(self):
        return self._multi_images
   
    
    