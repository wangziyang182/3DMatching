import numpy as np
import pathlib as PH

class dataset(object):

    def __init__(self):
        current_path = PH.Path(__file__).parent
        data_path = current_path.joinpath('data')
        
        self._tsdf_volume_list = tsdf_volume_list = [str(tsdf_volume) for tsdf_volume in data_path.glob('**/*voxel*.npy')]
        self._correspondence_list = correspondence_list = [str(correspondence) for correspondence in data_path.glob('**/*correspondence*.npy')]
        self.data_size = len(self._tsdf_volume_list)
        self._pointer_start = 0
        self._pointer_end = 0
        # self._config = config



    def generate_data(self,batch_size = 1):
        #update pointer
        if batch_size == 0:
            raise Exception('batch_size need to be greater than 0')
        if batch_size > self.data_size:
            raise Exception('batch_size cannot be greater than total number of data')

        self._pointer_end += batch_size
        self._pointer_start = self._pointer_end - batch_size

        if self._pointer_end >= self.data_size:
            self._pointer_end = self.data_size
        volume = np.concatenate([np.load(x)[None,...,None] for x in self._tsdf_volume_list[self._pointer_start:self._pointer_end]],axis = 0)

        correspondence = np.concatenate([np.load(x)[None,...] for x in self._correspondence_list[self._pointer_start:self._pointer_end]],axis = 0).astype('int')
        if self._pointer_end >= self.data_size:
            self._pointer_end = 0

        return volume,correspondence





    @property
    def tsdf_volume_list(self):
        return self._tsdf_volume_list

    @property
    def correspondence_list(self):
        return self._correspondence_list

    @property
    def pointer_start(self):
        return self._pointer_start

    @property
    def pointer_end(self):
        return self._pointer_end

    @tsdf_volume_list.setter
    def tsdf_volume_list(self,value):
        self._tsdf_volume_list = value
   
   
if __name__ == '__main__':
    data = dataset()
    # print(data.tsdf_volume_list)
    for i in range(10):
        # x,y = data.generate_data()
        x,y = data.generate_data(1)
        # print(x)
        print('-------')
        print(y)
