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

    for i in range(steps):

        #load correspondence and tsdf_volume
        tsdf_volume_batch,correspondence_batch = data.generate_data(1)
        
        # print(Model(tsdf_volume_batch))



if __name__ == '__main__':
    # app.run(main)
    main()

