import tensorflow as tf
import numpy as np

class Config(object):

    def __init__(self):
        
        #for training
        self._num_match = 50
        self._num_non_match = 10
        self._batch_size = 1
        self._learning_rate = 1e-4
        self._optimizer = tf.keras.optimizers.Adam(self._learning_rate)
        self._non_match_margin = 0.5
        self._from_scratch = True
        ##Random Seed for trian test split
        self._random_seed = 0 
        self._epoch = 1  

        #for tsdf intergration
        self._vol_bnds = np.array([[-1,5],[-2,2],[0,1]])
        self._voxel_size = 0.1
        self._n_imges = 2
        self._multi_images = False


    @property
    def num_match(self):
        return self._num_match
    
    @property
    def num_non_match(self):
        return self._num_non_match

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def non_match_margin(self):
        return self._non_match_margin

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
    
    @property
    def from_scratch(self):
        return self._from_scratch
    
    @property
    def random_seed(self):
        return self._random_seed

    @property
    def epoch(self):
        return self._epoch
    
    