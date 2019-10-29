import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
from Config import config
import pathlib as PH
from Data import dataset

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
# flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
# flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
#                  'Must divide evenly into the dataset sizes.')
# flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
#                  'for unit testing.')


#load data_set
data = dataset()
steps = 1
Model = TDDD_Net(config)

for i in range(steps):

    #load correspondence and tsdf_volume
    tsdf_volume_batch,correspondence_batch = data.generate_data(2)
    Model.train(tsdf_volume_batch,correspondence_batch,non_match = None)



