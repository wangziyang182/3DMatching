import tensorflow as tf
class config(object):

    def __init__(self):
        self._batch_match = 10
        self._batch_non_match = 10
        self._batch_size = 3
        self._optimizer = tf.keras.optimizers.Adam()


    @property
    def batch_match(self):
        return self._batch_match
    
    @property
    def batch_non_match(self):
        return self._batch_match

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def optimizer(self):
        return self._optimizer

    