import tensorflow as tf
import numpy as np

class Config(object):

    def __init__(self):
        
        #for training
        self._num_match = 50
        self._num_non_match = 50
        self._batch_size = 1
        self._learning_rate = 1e-4
        self._optimizer = tf.keras.optimizers.Adam(self._learning_rate)
        self._non_match_margin = 0.5
        self._from_scratch = True
        self._non_match_distance_clip = 3
        ##Random Seed for trian test split
        self._random_seed = 0 
        self._epoch = 50 




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
    def from_scratch(self):
        return self._from_scratch
    
    @property
    def random_seed(self):
        return self._random_seed

    @property
    def epoch(self):
        return self._epoch

    @property
    def non_match_distance_clip(self):
        return self._non_match_distance_clip
    
    
    
