import tensorflow as tf
import numpy as np
from Config import config

class TDDD_Net(tf.keras.Model):
  
    def __init__(self,config):
        super(TDDD_Net,self).__init__()
        
        self._config = config
        self.optimizer = tf.keras.optimizers.Adam()

        with tf.name_scope("Layer_1") as scope:
            self.conv_3d_l1 = tf.keras.layers.Conv3D(filters = 32, kernel_size = [3,3,2], strides = [1,1,1], padding = 'same',activation='relu')
            self.batch_l1 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_2") as scope:
            self.conv_3d_l2 = tf.keras.layers.Conv3D(filters = 32, kernel_size = [3,3,2],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l2 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_3") as scope:
            self.conv_3d_l3 = tf.keras.layers.Conv3D(filters = 64,kernel_size = [5,5,3],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l3 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_4") as scope:
            self.conv_3d_l4 = tf.keras.layers.Conv3D(filters = 64,kernel_size = [5,5,3],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l4 = tf.keras.layers.BatchNormalization()
            self.pool_3d_l4 = tf.keras.layers.AveragePooling3D()

        with tf.name_scope("Layer_5") as scope:
            self.conv_3d_l5 = tf.keras.layers.Conv3DTranspose(filters = 64,kernel_size=[3,3,2],strides = [1,1,1],data_format="channels_last")
            self.conv_3d_upool_l5 = tf.keras.layers.UpSampling3D()

        with tf.name_scope("Layer_6") as scope:
            self.conv_3d_l6 = tf.keras.layers.Conv3DTranspose(filters = 32,kernel_size = [3,3,2],strides = [1,1,1],data_format = "channels_last")

        with tf.name_scope("Layer_7") as scope:
            self.conv_3d_l7 = tf.keras.layers.Conv3DTranspose(filters = 32,kernel_size = [5,5,4],strides = [1,1,1],data_format = "channels_last")


    def call(self,input_tensor):

        print('layer_1')
        tensor = self.conv_3d_l1(input_tensor)        
        tensor = self.batch_l1(tensor + input_tensor)

        print('layer_2')
        tensor = self.conv_3d_l2(tensor)
        tensor = self.batch_l2(tensor)

        print('layer_3')
        tensor = self.conv_3d_l3(tensor)
        tensor = self.batch_l3(tensor)


        print('layer_4')
        tensor = self.conv_3d_l4(tensor)
        tensor = self.batch_l4(tensor)
        tensor = self.pool_3d_l4(tensor)

        print('layer_5')
        tensor = self.conv_3d_l5(tensor)
        tensor = self.conv_3d_upool_l5(tensor)

        print('layer_6')
        tensor = self.conv_3d_l6(tensor)

        print('layer_7')
        tensor = self.conv_3d_l7(tensor)



        return tensor
        

    # @tf.function
    def train(self,tsdf_volume,match,non_match = None):
        dim_0_index = tf.range(match.shape[0])
        # index = tf.transpose(tf.tile(index, [match.shape[1]]))
        # print(index)
        dim_0_index = tf.keras.backend.repeat_elements(dim_0_index, rep=match.shape[1], axis=0)
        dim_0_index = tf.reshape(dim_0_index,[match.shape[0],match.shape[1],1])
        match_a = tf.concat([dim_0_index,match[:,:,:3]],axis = 2)
        match_b = tf.concat([dim_0_index,match[:,:,3:6]],axis = 2)
        with tf.GradientTape() as tape:
            # jdx = tf.range(len(yp))
            # jdx = tf.tile(jdx, [len(yp)])
            voxel_descriptor = self.call(tsdf_volume)
            descriptor_a = tf.gather_nd(voxel_descriptor, match_a)
            descriptor_b =  tf.gather_nd(voxel_descriptor, match_b)

            # descriptor_b =  tf.gather_nd(voxel_descriptor[0,:], match[:,3:6])
            match_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.square(descriptor_a - descriptor_b) , axis = 2)))

            #need implement
            non_match_loss = 0
            loss = match_loss + non_match_loss

            print(loss)
        gradients = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


    @property
    def config(self):
        return self._config

    # @config.setter
    # def config(self, val):
    #     if isinstance(val,dict):
    #         self.__config = val
    #     else:
    #         print('please use a dict')


