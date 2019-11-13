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

# flags = tf.compat.v1.flags.Flag
FLAGS = flags.FLAGS

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

    x_point_idx = 0
    y_point_idx = 0

    for i in range(steps):

        #load correspondence and tsdf_volume
        tsdf_volume_test_batch,correspondence_test_batch,non_matches = data.generate_train_data_batch(50, 50,batch_size = 1,Non_Match_Distance_Clip = 5)
        # tsdf_volume_test_batch,correspondence_test_batch = data.generate_test_data_batch(1)
        descriptor = Model(tsdf_volume_test_batch).numpy()

        print(descriptor.shape)

        x = correspondence_test_batch[0,x_point_idx,:][:3]
        y = correspondence_test_batch[0,y_point_idx,:][3:]

        x_range = tf.range(descriptor.shape[1])
        y_range = tf.range(descriptor.shape[2])
        z_range = tf.range(descriptor.shape[3])
        print(x_range.shape)
        X_grid, Y_grid,Z_grid= tf.meshgrid(x_range, y_range,z_range)
        X_grid = tf.reshape(X_grid,[-1,1])
        Y_grid = tf.reshape(Y_grid,[-1,1])
        Z_grid = tf.reshape(Z_grid,[-1,1])

        grid = tf.concat([X_grid,Y_grid,Z_grid],axis = 1)

        x_de = descriptor[0,x[0],x[1],x[2]]

        tf.argmin(tf.square((descriptor - x_de)),axis = 0)
        descriptor_column = tf.reshape(descriptor,(-1, descriptor.shape[-1]))

        # print(descriptor_column[2])
        # print(descriptor[0,0,0,2,:])
        # print(grid)


        diff = tf.reduce_sum((descriptor_column - x_de) ** 2,axis = 1)
        x_de_match_idx = tf.argmin(diff,axis = 0)
        print('based on descriptor',grid[x_de_match_idx,:])
        for idx in tf.argsort(diff,axis=-1,direction='ASCENDING')[:10]:
            print(grid[idx,:])
        # print('top 10',tf.gather_nd(grid,tf.argsort(diff,axis=-1,direction='ASCENDING')[:10]))
        print('Ground Truth',[y[0],y[1],y[2]])
        # print('idx',x_de_match_idx[])
        print(tf.argsort(diff,axis=-1,direction='ASCENDING')[:10])
        # descriptor_column = tf.reshape(descriptor_column,descriptor.shape)

        # print(descriptor_reshape[0,0,0,1,:])

        y_de = descriptor[0,y[0],y[1],y[2]]


        print('diff',np.sqrt(np.sum((x_de - y_de) ** 2)))



if __name__ == '__main__':
    # app.run(main)
    main()

