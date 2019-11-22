import tensorflow as tf
import numpy as np
# from Config import config
from U_Net_Module import U_net_down_sampling_block,U_net_up_sampling_block

class TDDD_Net(tf.keras.Model):
    
    def __init__(self,model,from_scratch,weights_path,optimizer):
        super(TDDD_Net,self).__init__()
        
        self._optimizer = optimizer
        self._from_scratch = from_scratch
        self._model = model

        if model not in ['Standard_3D_Encoder_Decoder','3D_U_Net']:
            raise Exception('Only Standard_3D_Encoder_Decoder and 3D_U_Net are supported')

        if self._model == 'Standard_3D_Encoder_Decoder':
            with tf.name_scope("Layer_1") as scope:
                self.conv_3d_l1 = tf.keras.layers.Conv3D(filters = 32, kernel_size = [5,5,3], strides = [1,1,1], padding = 'valid',activation='relu')
                self.batch_l1 = tf.keras.layers.BatchNormalization()
            
            with tf.name_scope("Layer_2") as scope:
                self.conv_3d_l2 = tf.keras.layers.Conv3D(filters = 32, kernel_size = [5,5,3],strides = [1,1,1],padding = 'valid',activation = 'relu')
                self.batch_l2 = tf.keras.layers.BatchNormalization()
            
            with tf.name_scope("Layer_3") as scope:
                self.conv_3d_l3 = tf.keras.layers.Conv3D(filters = 64,kernel_size = [3,3,2],strides = [1,1,1],padding = 'valid',activation = 'relu')
                self.batch_l3 = tf.keras.layers.BatchNormalization()
            
            with tf.name_scope("Layer_4") as scope:
                self.conv_3d_l4 = tf.keras.layers.Conv3D(filters = 128,kernel_size = [3,3,2],strides = [1,1,1],padding = 'valid',activation = 'relu')
                self.batch_l4 = tf.keras.layers.BatchNormalization()
                # self.pool_3d_l4 = tf.keras.layers.AveragePooling3D()

            with tf.name_scope("Layer_5") as scope:
                self.conv_3d_l5_T = tf.keras.layers.Conv3DTranspose(filters = 128,kernel_size=[3,3,2],strides = [1,1,1],data_format="channels_last",activation = 'relu')
                self.batch_l5 = tf.keras.layers.BatchNormalization()

                # self.conv_3d_upool_l5 = tf.keras.layers.UpSampling3D()

            with tf.name_scope("Layer_6") as scope:
                self.conv_3d_l6_T = tf.keras.layers.Conv3DTranspose(filters = 64,kernel_size = [3,3,2],strides = [1,1,1],data_format = "channels_last",activation = 'relu')
                self.batch_l6 = tf.keras.layers.BatchNormalization()


            with tf.name_scope("Layer_7") as scope:
                self.conv_3d_l7_T = tf.keras.layers.Conv3DTranspose(filters = 32,kernel_size = [5,5,3],strides = [1,1,1],data_format = "channels_last",activation = 'relu')
                self.batch_l7 = tf.keras.layers.BatchNormalization()


            with tf.name_scope("Layer_8") as scope:
                self.conv_3d_l8_T = tf.keras.layers.Conv3DTranspose(filters = 32,kernel_size = [5,5,3],strides = [1,1,1],data_format = "channels_last") 

        if self._model == '3D_U_Net':

            self.left_block_level_1 = U_net_down_sampling_block(filters = [32,64],kernel_size = (3,3,3),pool_size = (2,2,1))

            self.left_block_level_2 = U_net_down_sampling_block(filters = [64,128],kernel_size = (3,3,3),pool_size = (2,2,2))

            self.left_block_level_3 = U_net_down_sampling_block(filters = [128,256],kernel_size = (3,3,3),pool_size = (2,2,1))

            self.conv_3d_bottom_1 = tf.keras.layers.Conv3D(filters = 256,kernel_size = [3,3,3],strides = [1,1,1],padding = 'same',data_format = "channels_last",activation = 'relu')
            self.batch_norm_1 = tf.keras.layers.BatchNormalization()

            self.conv_3d_bottom_2 = tf.keras.layers.Conv3D(filters = 512,kernel_size = [3,3,3],strides = [1,1,1],padding = 'same',data_format = "channels_last",activation = 'relu')
            self.batch_nrom_2 = tf.keras.layers.BatchNormalization()

            self.right_block_level_1 = U_net_up_sampling_block([256,256],kernel_size = (3,3,3),size = (2,2,1))


            self.right_block_level_2 = U_net_up_sampling_block([128,128],kernel_size = (3,3,3),size = (2,2,2))

            self.right_block_level_3 = U_net_up_sampling_block([64,64],kernel_size = (3,3,3),size = (2,2,1),change_size = False)

            self.conv_3d_right_level_1 = tf.keras.layers.Conv3D(filters = 32,kernel_size = [3,3,3],strides = [1,1,1],padding = 'same',data_format = "channels_last")

            self.fc_layer_1 = tf.keras.layers.Dense(128,activation = 'relu')
            self.fc_layer_2 = tf.keras.layers.Dense(3,activation = 'relu')

        self.create_ckpt_manager(weights_path)
        if not self._from_scratch:
            self.restore()
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")



    def call(self,input_tensor):
        if self._model == 'Standard_3D_Encoder_Decoder':
            tensor = self.conv_3d_l1(input_tensor)        
            tensor = self.batch_l1(tensor)

            tensor = self.conv_3d_l2(tensor)
            tensor = self.batch_l2(tensor)

            tensor = self.conv_3d_l3(tensor)
            tensor = self.batch_l3(tensor)

            tensor = self.conv_3d_l4(tensor)
            tensor = self.batch_l4(tensor)

            tensor = self.conv_3d_l5_T(tensor)
            tensor = self.batch_l5(tensor)

            tensor = self.conv_3d_l6_T(tensor)
            tensor = self.batch_l6(tensor)

            tensor = self.conv_3d_l7_T(tensor)
            tensor = self.batch_l7(tensor)

            tensor = self.conv_3d_l8_T(tensor)

        if self._model == '3D_U_Net':
            tensor,tensor_concat_level_1 = self.left_block_level_1(input_tensor)
            tensor,tensor_concat_level_2 = self.left_block_level_2(tensor)
            tensor,tensor_concat_level_3 = self.left_block_level_3(tensor)

            #Bottom Layer
            tensor = self.conv_3d_bottom_1(tensor)
            tensor = self.batch_norm_1(tensor)
            tensor = self.conv_3d_bottom_2(tensor)
            tensor = self.batch_nrom_2(tensor)

            tensor = self.right_block_level_1(tensor_concat_level_3,tensor)

            tensor = self.right_block_level_2(tensor_concat_level_2,tensor)
            tensor = self.right_block_level_3(tensor_concat_level_1,tensor)
            tensor = self.conv_3d_right_level_1(tensor)

        return tensor
        
    # @tf.function
    def compute_loss(self,tsdf_volume_object,tsdf_volume_package,match,non_match ,Non_March_Margin):

        dim_0_index_match = tf.range(match.shape[0])
        dim_0_index_match = tf.keras.backend.repeat_elements(dim_0_index_match, rep=match.shape[1], axis=0)
        dim_0_index_match = tf.reshape(dim_0_index_match,[match.shape[0],match.shape[1],1])
        dim_0_index_match = tf.dtypes.cast(dim_0_index_match,tf.int32)


        dim_0_index_non_match = tf.range(non_match.shape[0])
        dim_0_index_non_match = tf.keras.backend.repeat_elements(dim_0_index_non_match, rep=non_match.shape[1], axis=0)
        dim_0_index_non_match = tf.reshape(dim_0_index_non_match,[non_match.shape[0],non_match.shape[1],1])
        dim_0_index_non_match = tf.dtypes.cast(dim_0_index_non_match,tf.int32)

        match = tf.dtypes.cast(match,tf.int32)
        non_match = tf.dtypes.cast(non_match,tf.int32)

        # shift = match[:,:,:3] - match[:,:,3:6]
        # shift = tf.dtypes.cast(shift,tf.float32)

        points = tf.concat([dim_0_index_match,match[:,:,:3]],axis = 2)
        match_points = tf.concat([dim_0_index_match,match[:,:,3:6]],axis = 2)

        points_ = tf.concat([dim_0_index_non_match,non_match[:,:,:3]],axis = 2)
        non_match_points_ = tf.concat([dim_0_index_non_match,non_match[:,:,3:6]],axis = 2)

        
        with tf.GradientTape() as tape:

            voxel_descriptor_object = self.call(tsdf_volume_object)
            voxel_descriptor_package = self.call(tsdf_volume_package)

            voxel_descriptor_combine = tf.concat([voxel_descriptor_object,voxel_descriptor_package],axis = -1)

            #matching_descriptor
            descriptor_points = tf.gather_nd(voxel_descriptor_object, points)
            descriptor_match_points =  tf.gather_nd(voxel_descriptor_package, match_points)

            #non_matching_descriptor
            descriptor_points_ = tf.gather_nd(voxel_descriptor_object,points_)
            descriptor_non_match_a = tf.gather_nd(voxel_descriptor_package, non_match_points_)


            # #combine_feature
            # combine_feature = tf.gather_nd(voxel_descriptor_combine,points)

            # distance_shift = self.fc_layer_2(self.fc_layer_1(combine_feature))


            # shift_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((distance_shift - shift)),axis = 1)))

            #checking
            print('checking')
            # print(descriptor_points.shape)
            # print(descriptor_points[0,49])
            # print(voxel_descriptor_object[0,match[0,49,0],match[0,49,1],match[0,49,2]])

            # print(descriptor_match_points[0,2])
            # print(voxel_descriptor_package[0,match[0,2,3],match[0,2,4],match[0,2,5]])

            match_l2_diff = tf.sqrt(tf.reduce_sum(tf.square(descriptor_points - descriptor_match_points) , axis = 2))

            match_loss = tf.reduce_mean(tf.reduce_mean(match_l2_diff))

            non_match_l2_diff = tf.sqrt(tf.reduce_sum(tf.square(descriptor_points_ - descriptor_non_match_a) , axis = 2))

            print('Non_March_Margin',Non_March_Margin)
            hard_negatives = tf.greater((Non_March_Margin - non_match_l2_diff),0)
            hard_negatives = tf.cast(hard_negatives,tf.int32)
            hard_negatives = tf.dtypes.cast(tf.reduce_sum(hard_negatives,axis = 1),tf.float32)


            non_match_loss = tf.reduce_mean((1/ (hard_negatives + 1)) * tf.reduce_sum(tf.maximum((Non_March_Margin - non_match_l2_diff),0),axis = 1))

            #triple item loss
            loss = match_loss + non_match_loss
            # loss = match_loss
            self.match_loss = match_loss
            self.non_match_loss = non_match_loss
            self.hard_negatives = hard_negatives
            # print('shift_loss',shift_loss)

            # print(match_l2_diff)
            # print(tf.sqrt(match_l2_diff))


        # print('\n' + 'backward_propogating' + '\n')
        gradients = tape.gradient(loss,self.trainable_variables)
        print(len(gradients))
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss



    def save_parameter(self, file_path):
        self.save_weights(file_path, save_format='tf')


    def load_weights(self, file_path):
        self.load_weights('path_to_my_weights')


    def create_ckpt_manager(self,weights_path,max_to_keep = 3):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, weights_path, max_to_keep=3)

    def train_and_checkpoint(self,tsdf_volume_object,tsdf_volume_package,match,non_match = None,Non_Match_Margin = 0.1):

        loss = self.compute_loss(tsdf_volume_object,tsdf_volume_package,match,non_match,Non_Match_Margin)

        self.ckpt.step.assign_add(1)

        print("step : {}    |   loss : {:1.2f}  |   match_loss : {:1.2f}    |   non_match_loss : {1.2f}    |    hard_negatives_average".format(int(self.ckpt.step),loss.numpy(),self.match_loss,self.non_match_loss,sum(self.hard_negatives)/len(self.hard_negatives)))
        print('hard_negatives : {}'.format(self.hard_negatives))
        if int(self.ckpt.step) % 3 == 0:
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def restore(self):
        print('\n' + 'restore from :',self.manager.latest_checkpoint)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()


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


