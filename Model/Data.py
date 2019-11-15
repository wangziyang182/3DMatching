import numpy as np
import pathlib as PH
from sklearn.model_selection import train_test_split,ShuffleSplit

class dataset(object):

    def __init__(self):
        current_path = PH.Path(__file__).parent
        data_path = current_path.joinpath('data')
        vol_path = data_path.joinpath('vol_dim.npy')
        self._tsdf_volume_list = tsdf_volume_list = [str(tsdf_volume) for tsdf_volume in data_path.glob('**/*voxel*.npy')]
        self._correspondence_list = correspondence_list = [str(correspondence) for correspondence in data_path.glob('**/*correspondence*.npy')]
        
        self.data_size = len(self._tsdf_volume_list)
        self._pointer_start = 0
        self._pointer_end = 0
        self._vol_dim = np.load(vol_path)
        self._shift = np.array([[self._vol_dim[0] // 2],[0],[0]])
        # self._config = config

    def x_y_split(self,random_seed = 0):

        self._tsdf_volume_list_train,\
        self._tsdf_volume_list_test,\
        self._correspondence_list_train,\
        self._correspondence_list_test =train_test_split(self._tsdf_volume_list, self._correspondence_list, test_size=0.33, random_state=random_seed)
        self.train_size = len(self._correspondence_list_train)
        self.test_size = len(self._correspondence_list_test)

    def generate_train_data_batch(self,num_match, num_non_match,batch_size = 1,Non_Match_Distance_Clip = 5):
        #update pointer
        if batch_size == 0:
            raise Exception('batch_size need to be greater than 0')
        if batch_size > self.data_size:
            raise Exception('batch_size cannot be greater than total number of data')

        self._pointer_end += batch_size
        self._pointer_start = self._pointer_end - batch_size

        if self._pointer_end >= self.train_size:
            self._pointer_end = self.train_size


        volume = np.concatenate([np.load(x)[None,...,None] for x in self._tsdf_volume_list_train[self._pointer_start:self._pointer_end]],axis = 0)

        match = np.concatenate([np.load(x)[None,...] for x in self._correspondence_list_train[self._pointer_start:self._pointer_end]],axis = 0).astype('int')
        
        match[:,:,3:] = match[:,:,3:] - self._shift.T
        non_match = self.generate_non_matches(match,Non_Match_Distance_Clip)

        #random sample points
        if num_match <= match.shape[1]:
            match_sample_idx = np.random.choice(match.shape[1], size=num_match, replace=False)
        else:
            raise Exception('number of matching sampled cannot be greater than the total number of points inside a mesh')

        if num_match <= non_match.shape[1]:
            non_match_sample_idx = np.random.choice(non_match.shape[1], size=num_non_match, replace=False)
        else:
            raise Exception('number of non-matching sampled cannot be greater than the total number of points inside a mesh')
        



        ####test!!!!!!
        # match_sample_idx = np.arange(match.shape[1])
        




        match = match[:,match_sample_idx,:]
        non_match = non_match[:,non_match_sample_idx,:]

        if self._pointer_end >= self.train_size:
            self._pointer_end = 0

        volume_object = volume[:,:self._shift[0,0],:]
        volume_package = volume[:,self._shift[0,0]:,:]

        if volume_object.shape != volume_package.shape:
            raise Exception('object volume is different from package volume')

        return volume_object,volume_package,match,non_match

    def generate_test_data_batch(self,batch_size = 1):
        #update pointer
        if batch_size == 0:
            raise Exception('batch_size need to be greater than 0')
        if batch_size > self.data_size:
            raise Exception('batch_size cannot be greater than total number of data')

        self._pointer_end += batch_size
        self._pointer_start = self._pointer_end - batch_size

        if self._pointer_end >= self.test_size:
            self._pointer_end = self.test_size

        volume = np.concatenate([np.load(x)[None,...,None] for x in self._tsdf_volume_list_test[self._pointer_start:self._pointer_end]],axis = 0)

        match = np.concatenate([np.load(x)[None,...] for x in self._correspondence_list_test[self._pointer_start:self._pointer_end]],axis = 0).astype('int')

        match[:,:,3:] = match[:,:,3:] - self._shift.T

        # non_matches = self.generate_non_matches(match)

        if self._pointer_end >= self.test_size:
            self._pointer_end = 0

        volume_object = volume[:,:self._shift[0,0],:]
        volume_package = volume[:,self._shift[0,0]:,:]

        return volume_object,volume_package,match

    def generate_non_matches(self,match,Non_Match_Distance_Clip = 5):
        non_matches_batch = np.zeros_like(match[:,:,:3])

        for i,batch in enumerate(match):
            vertex_a = batch[:,:3]
            non_matches = np.zeros_like(vertex_a)
            for j,vert in enumerate(vertex_a):
                x = np.random.randint(0,self._vol_dim[0] - self._shift[0,0])
                y = np.random.randint(0,self._vol_dim[1] - self._shift[1,0])
                z = np.random.randint(0,self._vol_dim[2] - self._shift[2,0])
                non_match = np.array([x,y,z])
                if np.sum((vert - non_match) ** 2) ** 0.5 > Non_Match_Distance_Clip:
                    non_matches[j,:] = non_match

            non_matches_batch[i] = non_matches
        non_matches_batch = np.concatenate([match[:,:,:3],non_matches_batch],axis = 2)

        return non_matches_batch

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
   

