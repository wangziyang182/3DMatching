import tensorflow as tf
import numpy as np

class TDDD_Net(tf.keras.Model):
  
    def __init__(self,config):
        super(TDDD_Net,self).__init__()
        
        self._config = config

        with tf.name_scope("Layer_1") as scope:
            self.conv_3d_l1 = tf.keras.layers.Conv3D(filters = 32, kernel_size = [3,3,3], strides = [1,1,1], padding = 'same',activation='relu')
            self.batch_l1 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_2") as scope:
            self.conv_3d_l2 = tf.keras.layers.Conv3D(filters = 64, kernel_size = [3,3,3],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l2 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_3") as scope:
            self.conv_3d_l3 = tf.keras.layers.Conv3D(filters = 128,kernel_size = [5,5,5],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l3 = tf.keras.layers.BatchNormalization()
        
        with tf.name_scope("Layer_4") as scope:
            self.conv_3d_l4 = tf.keras.layers.Conv3D(filters = 256,kernel_size = [5,5,5],strides = [1,1,1],padding = 'valid',activation = 'relu')
            self.batch_l4 = tf.keras.layers.BatchNormalization()
            self.pool_3d_l4 = tf.keras.layers.AveragePooling3D()

        with tf.name_scope("Layer_5") as scope:
            self.conv_3d_l5 = tf.keras.layers.Conv3DTranspose(filters = 256,kernel_size=[3,3,3],strides = [1,1,1],data_format="channels_last")
            self.conv_3d_upool_l5 = tf.keras.layers.UpSampling3D()

        with tf.name_scope("Layer_6") as scope:
            self.conv_3d_l6 = tf.keras.layers.Conv3DTranspose(filters = 128,kernel_size = [3,3,3],strides = [1,1,1],data_format = "channels_last")

        with tf.name_scope("Layer_7") as scope:
            self.conv_3d_l7 = tf.keras.layers.Conv3DTranspose(filters = 64,kernel_size = [5,5,5],strides = [1,1,1],data_format = "channels_last")


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
        

    def compute_loss(self,x_train_item,x_train_package,y_train_item_match,y_train_item_non_match,y_train_package_match,y_train_package_non_match):
        '''
        Simason Fashion Training

        '''
        x_train_item_descriptor = self.call(x_train_item)
        x_train_package_descriptor = self.call(x_train_package)

        print(x_train_item_descriptor.shape)
        print(x_train_package_descriptor.shape)
        print(y_train_item_match.shape)

        for i in range(self.config.batch_size):

            try:
                ele_item_match = tf.stack([x_train_item_descriptor[i,y_train_item_match[i,j,0],y_train_item_match[i,j,1],y_train_item_match[i,j,2],:] for j in range(y_train_item_match.shape[1])],axis = 0)[None,...]

                ele_item_non_match = tf.stack([x_train_item_descriptor[i,y_train_item_non_match[i,j,0],y_train_item_non_match[i,j,1],y_train_item_non_match[i,j,2],:] for j in range(y_train_item_non_match.shape[1])],axis = 0)[None,...]

                ele_package_match = tf.stack([x_train_package_descriptor[i,y_train_package_match[i,j,0],y_train_package_match[i,j,1],y_train_package_match[i,j,2],:] for j in range(y_train_package_match.shape[1])],axis = 0)[None,...]

                ele_package_non_match = tf.stack([x_train_package_descriptor[i,y_train_package_non_match[i,j,0],y_train_package_non_match[i,j,1],y_train_package_non_match[i,j,2],:] for j in range(y_train_package_non_match.shape[1])],axis = 0)[None,...]

                # print(ele_item_match)
                # print(ele_item_non_match)
                # print(ele_package_match)
                # print(ele_package_non_match)

                item_match = tf.concat((item_match,ele_item_match),axis =0)
                item_non_match = tf.concat((item_non_match,ele_item_non_match),axis = 0)
                package_match = tf.concat((package_match,ele_package_match),axis = 0)
                package_non_match = tf.concat((package_non_match,ele_package_non_match),axis = 0)
                # print(ele_item_match.shape)
            except:
                item_match = tf.stack([x_train_item_descriptor[i,y_train_item_match[i,j,0],y_train_item_match[i,j,1],y_train_item_match[i,j,2],:] for j in range(y_train_item_match.shape[1])],axis = 0)[None,...]

                item_non_match = tf.stack([x_train_item_descriptor[i,y_train_item_non_match[i,j,0],y_train_item_non_match[i,j,1],y_train_item_non_match[i,j,2],:] for j in range(y_train_item_non_match.shape[1])],axis = 0)[None,...]

                package_match = tf.stack([x_train_package_descriptor[i,y_train_package_match[i,j,0],y_train_package_match[i,j,1],y_train_package_match[i,j,2],:] for j in range(y_train_package_match.shape[1])],axis = 0)[None,...]

                package_non_match = tf.stack([x_train_package_descriptor[i,y_train_package_non_match[i,j,0],y_train_package_non_match[i,j,1],y_train_package_non_match[i,j,2],:] for j in range(y_train_package_non_match.shape[1])],axis = 0)[None,...]
        print(item_match.shape,item_non_match.shape,package_match.shape,package_non_match.shape)

        l2_match = tf.reduce_mean(tf.reduce_mean(tf.math.sqrt(tf.reduce_sum((item_match - package_match) ** 2,axis = -1))))
        l2_non_match = tf.reduce_mean(tf.reduce_mean(tf.math.sqrt(tf.reduce_sum((item_non_match - package_non_match) ** 2,axis = -1))))

        print(l2_match.numpy())
        print(l2_non_match.numpy())



    @property
    def config(self):
        return self._config

    # @config.setter
    # def config(self, val):
    #     if isinstance(val,dict):
    #         self.__config = val
    #     else:
    #         print('please use a dict')
