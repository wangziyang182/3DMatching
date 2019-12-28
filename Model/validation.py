import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
import pickle
# from Config import config
import pathlib as PH
from Data import dataset
from absl import flags
from absl import app
import sys
from tqdm import tnrange

sys.path.append('..')
from utils import get_top_match,plot_3d_heat_map,compute_mesh,visualize_ground_truth



# def main(argv):

def main():
    # if FLAGS.debug:
    #     print('non-flag arguments:', argv)
    # if FLAGS.age is not None:
    #     pass
    data = dataset()
    data.x_y_split(random_seed = 0)
    from_scratch = False
    optimizer = tf.keras.optimizers.Adam()


    
    BASE_DIR = PH.Path(__file__).parent.parent
    MODEL_WEIGHTS_PATH = BASE_DIR.joinpath('Model').joinpath('Model_Weights')
    weights_path = str(MODEL_WEIGHTS_PATH.joinpath('ckpt'))
    Results_path = BASE_DIR.joinpath('Model')


    # define Matching Net
    Model = TDDD_Net('3D_U_Net',from_scratch,weights_path,optimizer)
    # Model.optimizer = optimizer
    # Model.create_ckpt_manager(weights_path)
    # Model.restore()

    batch = 0

    # x_point_idx = 1800
    # y_point_idx = 1800
    match_count = {}

    #load correspondence and tsdf_volume
    # for i in range(data.test_size):
    for i in range(1):

        match_count['Test_Object_{}'.format(i)] = {}
        match_count['Test_Object_{}'.format(i)]['exact_match'] = 0
        match_count['Test_Object_{}'.format(i)]['one_dist_off_match'] = 0
        match_count['Test_Object_{}'.format(i)]['two_dist_off_match'] = 0
        # tsdf_volume_test_object_batch,tsdf_volume_test_package_batch,match,non_match,ply_train = data.generate_train_data_batch(10, 10,batch_size = 2,Non_Match_Distance_Clip = 5)

        tsdf_volume_test_object_batch,tsdf_volume_test_package_batch,match,ply_test = data.generate_test_data_batch(1)


        

        #get the descriptor for object and package
        descriptor_object = Model(tsdf_volume_test_object_batch).numpy()
        descriptor_package = Model(tsdf_volume_test_package_batch).numpy()
        
        print()
        # for j in tnrange(match[batch].shape[0]):
        for j in range(1):

            x_point_idx = j
            y_point_idx = j


            x_point_idx = 4000
            y_point_idx = 4000

            #get the src and destination ground truth for first batch point_idxth point 
            print(match.shape)
            src = match[batch,x_point_idx,:][:3]
            dest = match[batch,y_point_idx,:][3:]

            top_match,top_matching_distance,top_idx = get_top_match(batch,src,descriptor_object,descriptor_package,dest)

            src_des = descriptor_object[batch,src[0],src[1],src[2]]
            dest_des = descriptor_package[batch,dest[0],dest[1],dest[2]]

            print('top_best',top_match)
            print('top_matching_distance',top_matching_distance)

            # print('matching descriptor',dest)
            print('Ground Truth',[dest[0],dest[1],dest[2]])
            # print('ground_truth_diff',np.sqrt(np.sum((src_des - dest_des) ** 2)))

            # print(top_match[0,:].numpy())
            # print(dest)

            if np.sqrt(np.sum((top_match[0,:] - dest) ** 2)) == 0:
                match_count['Test_Object_{}'.format(i)]['exact_match'] += 1
            if np.sqrt(np.sum((top_match[0,:] - dest) ** 2)) <= 1:
                match_count['Test_Object_{}'.format(i)]['one_dist_off_match'] += 1
            if np.sqrt(np.sum((top_match[0,:] - dest) ** 2)) <= 2:
                match_count['Test_Object_{}'.format(i)]['two_dist_off_match'] += 1 

            # print('best_match',top_match[0,:])
            # print('ground_truth',dest)
            if j % 500 == 0:
                print('match_count',match_count)

            # with open(Results_path.joinpath('Results.pickle'), 'wb') as handle:
                # pickle.dump(match_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

            visualize_ground_truth(batch,src,dest,descriptor_object,descriptor_package,top_idx,ply_test,data.shift)

   



if __name__ == '__main__':
    # app.run(main)
    main()

