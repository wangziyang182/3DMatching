import numpy as np
import pathlib as PH
from sklearn.model_selection import train_test_split,ShuffleSplit

class dataset(object):

    def __init__(self):
        BASE_PATH = PH.Path(__file__).parent.parent
        current_path = BASE_PATH.joinpath('model')
        data_path = current_path.joinpath('data')
        vol_path = data_path.joinpath('vol_dim.npy')
        ply_path = BASE_PATH.joinpath('env').joinpath('tsdf_projected_ply')

        self._tsdf_volume_list = [str(tsdf_volume) for tsdf_volume in sorted(data_path.glob('**/*voxel*.npy'))]
        self._correspondence_list = [str(correspondence) for correspondence in sorted(data_path.glob('**/*correspondence*.npy'))]
        self._obj_ply = [str(ply) for ply in sorted(ply_path.glob('**/*.ply'))]


        # sorted(self._tsdf_volume_list)
        # sorted(self._correspondence_list)
        # sorted(self._obj_ply)
        
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

    def generate_train_data_batch(self,num_match, num_non_match,batch_size = 1,Non_Match_Distance_Clip = 3):
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

        # #random sample points
        # if num_match <= match.shape[1]:
        #     match_sample_idx_list = []
        #     for i in range(match.shape[0]):
        #         match_sample_idx = np.random.choice(match.shape[1], size=num_match, replace=True)
        #         match_sample_idx_list.append(match_sample_idx)
        # else:
        #     raise Exception('number of matching sampled cannot be greater than the total number of points inside a mesh')

        match_sample_idx_list = []
        for i in range(match.shape[0]):
            match_sample_idx = np.random.choice(match.shape[1], size=num_match, replace=True)
            match_sample_idx_list.append(match_sample_idx)

        # if num_match <= non_match.shape[1]:
        #     non_match_sample_idx = np.random.choice(non_match.shape[1], size=num_non_match, replace=False)
        # else:
        #     raise Exception('number of non-matching sampled cannot be greater than the total number of points inside a mesh')
        

        ####test!!!!!!
        # match_sample_idx = np.arange(match.shape[1])
        match_list = []
        for i,item in enumerate(match_sample_idx_list):
            match_ele = match[i,item,:][None,...]
            match_list.append(match_ele)
        match = np.concatenate(match_list,axis = 0)


        # match = match[:,match_sample_idx_list,:]
        non_match = self.generate_non_matches(match,Non_Match_Distance_Clip)

        # non_match = non_match[:,non_match_sample_idx,:]

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

    def generate_non_matches(self,match,Non_Match_Distance_Clip = 3):
        non_matches_batch = np.zeros_like(match[:,:,:3])
        dist_check = lambda x,y: np.sqrt(np.sum((x - y) ** 2,axis = -1)) < Non_Match_Distance_Clip
        for i,batch in enumerate(match):
            non_matches = None
            vertex_a = batch[:,:3]
            flag = True
            while flag:
                x = np.random.randint(0,self._vol_dim[0] - self._shift[0,0],(vertex_a.shape[0],1))
                y = np.random.randint(0,self._vol_dim[1] - self._shift[1,0],(vertex_a.shape[0],1))
                z = np.random.randint(0,self._vol_dim[2] - self._shift[2,0],(vertex_a.shape[0],1))
                non_match_ele = np.concatenate([x,z,y],axis = 1)
                idx = dist_check(vertex_a,non_match_ele)
                try:
                    non_matches = np.concatenate([non_matches,non_match_ele[~idx,:]],axis = 0)
                except:
                    non_matches = non_match_ele[~idx,:]

                vertex_a = vertex_a[idx,:]
                if np.sum(idx) == 0:
                    flag = False

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
   

# if __name__ == '__main__':
#     data = dataset()
#     data.x_y_split()
#     data.generate_train_data_batch(int(1e6), int(1e6),batch_size = 3,Non_Match_Distance_Clip = 3)

