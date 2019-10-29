import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
from Config import config
import pathlib as PH

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


current_path = PH.Path(__file__).parent
data_path = current_path.joinpath('data')
tsdf_volume_list = [str(tsdf_volume) for tsdf_volume in data_path.glob('**/*voxel*.npy')]
correspondence_list = [str(correspondence) for correspondence in data_path.glob('**/*correspondence*.npy')]

steps = 1

Model = TDDD_Net(config)
for i in range(steps):

    #load correspondence and tsdf_volume
    correspondence = np.load(correspondence_list[i]).astype('int')
    tsdf_volume = np.load(tsdf_volume_list[i])[None,...,None]
    Model.train(tsdf_volume,correspondence,non_match = None)




















# config = config()
# data = dataset(config)
# data.generate_data()

# # print(data.x_train_item.shape)
# # print(data.x_train_package.shape)
# # print(data.y_train_match.shape)


# ddd_Net = TDDD_Net(config)

# ddd_Net.compute_loss(data.x_train_item,data.x_train_package,data.y_train_item_match,data.y_train_item_non_match,data.y_train_package_match,data.y_train_package_non_match)



# # print(ddd_Net.config)
# # ddd_Net.config = 10
# # print(ddd_Net.config)

