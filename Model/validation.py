import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
# from Config import config
import pathlib as PH
from Data import dataset
from absl import flags
from absl import app
import sys
sys.path.append('..')
from utils import get_top_10_match,plot_3d_heat_map


flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')



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


    # define Matching Net
    Model = TDDD_Net()
    Model.optimizer = optimizer
    Model.create_ckpt_manager(weights_path)
    Model.restore()

    steps = 1

    x_point_idx = 3
    y_point_idx = 3
    batch = 0
    for i in range(steps):

        #load correspondence and tsdf_volume
        tsdf_volume_test_object_batch,tsdf_volume_test_package_batch,match,non_match = data.generate_train_data_batch(50, 50,batch_size = 1,Non_Match_Distance_Clip = 5)
        # tsdf_volume_test_object_batch,tsdf_volume_test_package_batch,match = data.generate_test_data_batch(1)

        print('tsdf_volume_test_object_batch',tsdf_volume_test_object_batch.shape)
        print('match',match.shape)
        #get the descriptor for object and package
        descriptor_object = Model(tsdf_volume_test_object_batch).numpy()
        descriptor_package = Model(tsdf_volume_test_package_batch).numpy()

        #get the src and destination ground truth for first batch point_idxth point 
        src = match[batch,x_point_idx,:][:3]
        dest = match[batch,y_point_idx,:][3:]

        top_10_dest,top_10_matching_distance = get_top_10_match(batch,src,descriptor_object,descriptor_package)

        src_des = descriptor_object[batch,src[0],src[1],src[2]]
        dest_des = descriptor_package[batch,dest[0],dest[1],dest[2]]

        print('diff',np.sqrt(np.sum((src_des - dest_des) ** 2)))
        print('top_10_dest',top_10_dest,top_10_matching_distance)
        print('Ground Truth',[dest[0],dest[1],dest[2]])

        plot_3d_heat_map(src_des,descriptor_package)



if __name__ == '__main__':
    # app.run(main)
    main()

