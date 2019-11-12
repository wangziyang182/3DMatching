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

    x_point_idx = 1
    y_point_idx = 1

    for i in range(steps):

        #load correspondence and tsdf_volume
        tsdf_volume_test_batch,correspondence_test_batch = data.generate_test_data_batch(1)
        descriptor = Model(tsdf_volume_test_batch).numpy()

        print(descriptor.shape)

        x = correspondence_test_batch[0,x_point_idx,:][:3]
        y = correspondence_test_batch[0,y_point_idx,:][3:]

        x_de = descriptor[0,x[0],x[1],x[2]]
        y_de = descriptor[0,y[0],y[1],y[2]]

        print('diff',np.sqrt(np.sum((x_de - y_de) ** 2)))



if __name__ == '__main__':
    # app.run(main)
    main()

