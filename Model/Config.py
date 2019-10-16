class config(object):

    def __init__(self):
        self._batch_match = 10
        self._batch_non_match = 10
        self._batch_size = 3

    @property
    def batch_match(self):
        return self._batch_match
    
    @property
    def batch_non_match(self):
        return self._batch_match

    @property
    def batch_size(self):
        return self._batch_size

    