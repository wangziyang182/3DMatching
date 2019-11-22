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


def main():

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
    model = config.model

    data = dataset()
    data.x_y_split(random_seed = random_seed)
    steps_per_epoch = data.train_size / batch_size + 1
    
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
    Model = TDDD_Net(model,from_scratch,weights_path,optimizer)

    for i in tqdm(range(epoch),desc="Epoch",position = 0,leave = True):
        for j in tqdm(range(data.train_size // batch_size + 1),desc="Step",position=0, leave=True):
            #load correspondence and tsdf_volume
            tsdf_volume_object_batch_train,tsdf_volume_package_batch_train,correspondence_batch_train,non_correspondence_train,_ = data.generate_train_data_batch(num_match,num_non_match,batch_size,non_match_distance_clip)


            Model.train_and_checkpoint(tsdf_volume_object_batch_train,tsdf_volume_package_batch_train,correspondence_batch_train,non_match = non_correspondence_train,Non_Match_Margin = non_match_margin)
        

if __name__ == '__main__':
    main()
