import tensorflow as tf
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
from Config import Config
import shutil
import pathlib as PH
from Data import dataset
from absl import flags
from absl import app
from absl import logging
from tqdm import tqdm

# flags = tf.compat.v1.flags.Flag

FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_float('non_match_M', 1, 'cut_off of non_match')
flags.DEFINE_integer('epoch', 10,'number of epoches to train')

# def main(argv):

def main():
    # if FLAGS.debug:
    #     print('non-flag arguments:', argv)
    # if FLAGS.age is not None:
    #     pass

    #init parameter
    config = Config()
    batch_size = config.batch_size
    epoch = config.epoch
    optimizer = config.optimizer
    from_scratch = config.from_scratch
    random_seed = config.random_seed
    num_match = config.num_match
    num_non_match = config.num_non_match
    non_match_margin = config.non_match_margin
    non_match_distance_clip = config.non_match_distance_clip

    data = dataset()
    data.x_y_split(random_seed = random_seed)
    
    steps_per_epoch = data.train_size // batch_size + 1
    
    BASE_DIR = PH.Path(__file__).absolute().parent.parent
    MODEL_WEIGHTS_PATH = BASE_DIR.joinpath('Model').joinpath('Model_Weights')
    
    if from_scratch:
        try:
            shutil.rmtree(str(MODEL_WEIGHTS_PATH))
        except:
            pass

    if not os.path.exists(str(MODEL_WEIGHTS_PATH)):
        os.mkdir(str(MODEL_WEIGHTS_PATH))
    weights_path = str(MODEL_WEIGHTS_PATH.joinpath('ckpt'))


    # define Matching Net
    Model = TDDD_Net()
    Model.optimizer = optimizer
    Model.create_ckpt_manager(weights_path)

   
    for i in range(epoch):
        print('epoch',i)
        for j in tqdm(range(data.train_size // batch_size + 1)):
            #load correspondence and tsdf_volume
            tsdf_volume_batch_train,correspondence_batch_train,non_correspondence_train = data.generate_train_data_batch(num_match,num_non_match,batch_size,non_match_distance_clip)
            Model.train_and_checkpoint(tsdf_volume_batch_train,correspondence_batch_train,non_match = non_correspondence_train,Non_Match_Margin = non_match_margin,from_scratch = from_scratch)

        # Model.save_parameter(weights_path)
    
    #load model weights
    # Model.load_weights(weights_path)
    # tsdf_volume_batch,correspondence_batch = data.generate_data(5)
    # print(Model(tsdf_volume_batch))



if __name__ == '__main__':
    # app.run(main)
    main()
