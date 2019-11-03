import tensorflow as tf
import numpy as np
# from Config import config

class TDDD_Net(tf.keras.Model):
  
    def __init__(self):
        super(TDDD_Net,self).__init__()
        
        self._optimizer = None
        # self._config = config

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

        tensor = self.conv_3d_l1(input_tensor)        
        tensor = self.batch_l1(tensor + input_tensor)

        tensor = self.conv_3d_l2(tensor)
        tensor = self.batch_l2(tensor)

        tensor = self.conv_3d_l3(tensor)
        tensor = self.batch_l3(tensor)

        tensor = self.conv_3d_l4(tensor)
        tensor = self.batch_l4(tensor)
        tensor = self.pool_3d_l4(tensor)

        tensor = self.conv_3d_l5(tensor)
        tensor = self.conv_3d_upool_l5(tensor)

        tensor = self.conv_3d_l6(tensor)

        tensor = self.conv_3d_l7(tensor)

        return tensor
        

    @tf.function
    def train(self,tsdf_volume,match,non_match = None,Non_March_Margin = 1):
        dim_0_index = tf.range(match.shape[0])
        dim_0_index = tf.keras.backend.repeat_elements(dim_0_index, rep=match.shape[1], axis=0)
        dim_0_index = tf.reshape(dim_0_index,[match.shape[0],match.shape[1],1])
        
        dim_0_index = tf.dtypes.cast(dim_0_index,tf.int32)
        match = tf.dtypes.cast(match,tf.int32)
        non_match = tf.dtypes.cast(non_match,tf.int32)

        vert_a = tf.concat([dim_0_index,match[:,:,:3]],axis = 2)
        match_a = tf.concat([dim_0_index,match[:,:,3:6]],axis = 2)
        non_match_a = tf.concat([dim_0_index,non_match[:,:,3:6]],axis = 2)
        
        with tf.GradientTape() as tape:
            voxel_descriptor = self.call(tsdf_volume)
            descriptor_a = tf.gather_nd(voxel_descriptor, vert_a)
            descriptor_match_a =  tf.gather_nd(voxel_descriptor, match_a)
            descriptor_non_match_a = tf.gather_nd(voxel_descriptor, non_match_a)

            print('voxel_descriptor',voxel_descriptor.shape)
            #checking
            # print(descriptor_a[0,1])
            # print(voxel_descriptor[0,match[0,1,0],match[0,1,1],match[0,1,2]])

            # print(descriptor_match_a[0,1])
            # print(voxel_descriptor[0,match[0,1,3],match[0,1,4],match[0,1,5]])

            # descriptor_match_a =  tf.gather_nd(voxel_descriptor[0,:], match[:,3:6])
            match_l2_diff = tf.reduce_sum(tf.square(descriptor_a - descriptor_match_a) , axis = 2)

            match_loss = tf.reduce_mean(tf.reduce_mean(match_l2_diff))

            
            non_match_l2_diff = tf.sqrt(tf.reduce_sum(tf.square(descriptor_a - descriptor_non_match_a) , axis = 2))

            hard_negatives = tf.greater((Non_March_Margin - non_match_l2_diff),0)
            hard_negatives = tf.cast(hard_negatives,tf.int32)
            hard_negatives = tf.dtypes.cast(tf.reduce_sum(hard_negatives,axis = 1),tf.float32)

            non_match_loss = tf.reduce_sum((1/ hard_negatives) * tf.reduce_sum(tf.maximum((Non_March_Margin - non_match_l2_diff),0) ** 2))

            loss = match_loss + non_match_loss

        gradients = tape.gradient(loss,self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def save_parameter(self, file_path):
        self.save_weights(file_path, save_format='tf')


    def load_weights(self, file_path):
        self.load_weights('path_to_my_weights')


    def create_ckpt_manager(self,weights_path,max_to_keep = 3):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, weights_path, max_to_keep=3)

    def train_and_checkpoint(self,tsdf_volume,match,non_match = None,Non_March_Margin = 10):
        self.ckpt.restore(self.manager.latest_checkpoint)

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        loss = self.train(tsdf_volume,match,non_match,Non_March_Margin)
        self.ckpt.step.assign_add(1)

        if int(self.ckpt.step) % 1 == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            print("loss {:1.2f}".format(loss.numpy()))
    
    def restore(self):
        self.ckpt.restore(self.manager.latest_checkpoint)


    # def 
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,val):
        self._optimizer = val
    # @property
    # def config(self):
    #     return self._config

    # @config.setter
    # def config(self, val):
    #     if isinstance(val,dict):
    #         self.__config = val
    #     else:
    #         print('please use a dict')


