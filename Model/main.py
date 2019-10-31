import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
from Config import config
import pathlib as PH
from Data import dataset
from absl import flags
from absl import app

# flags = tf.compat.v1.flags.Flag
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')



def main(argv):
    # if FLAGS.debug:
    #     print('non-flag arguments:', argv)
    # if FLAGS.age is not None:
    #     pass

    data = dataset()
    steps = 1
    Model = TDDD_Net(config)

    for i in range(steps):

        #load correspondence and tsdf_volume
        tsdf_volume_batch,correspondence_batch = data.generate_data(2)
        Model.train(tsdf_volume_batch,correspondence_batch,non_match = None)



if __name__ == '__main__':
    app.run(main)