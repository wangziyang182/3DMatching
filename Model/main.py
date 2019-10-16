import tensorflow as tf
import keras
import os
import numpy as np
from Model import TDDD_Net
from Data import dataset
from Config import config

print(tf.__version__)
config = config()
data = dataset(config)
data.generate_data()

# print(data.x_train_item.shape)
# print(data.x_train_package.shape)
# print(data.y_train_match.shape)


ddd_Net = TDDD_Net(config)

ddd_Net.compute_loss(data.x_train_item,data.x_train_package,data.y_train_item_match,data.y_train_item_non_match,data.y_train_package_match,data.y_train_package_non_match)



# print(ddd_Net.config)
# ddd_Net.config = 10
# print(ddd_Net.config)

