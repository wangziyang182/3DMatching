import tensorflow as tf 

class U_net_down_sampling_block(tf.keras.Model):
    def __init__(self, filters,kernel_size = (3,3,3),pool_size = (2,2,2)):
        filter_1, filter_2 = filters
        super(U_net_down_sampling_block, self).__init__()

        self.conv_3d_filter_1 = tf.keras.layers.Conv3D(filters = filter_1, kernel_size = kernel_size, strides = [1,1,1], padding = 'same',activation='relu')
        self.batch_filter_1 = tf.keras.layers.BatchNormalization()

        self.conv_3d_filter_2 = tf.keras.layers.Conv3D(filters = filter_2, kernel_size = kernel_size, strides = [1,1,1], padding = 'same',activation='relu')
        self.batch_filter_2 = tf.keras.layers.BatchNormalization()

        self.Max_Pool_3D = tf.keras.layers.MaxPool3D(pool_size = pool_size)


    def call(self, tensor):
        tensor = self.conv_3d_filter_1(tensor)
        tensor = self.batch_filter_1(tensor)

        tensor_concat = self.conv_3d_filter_2(tensor)
        tensor = self.batch_filter_2(tensor)

        tensor = self.Max_Pool_3D(tensor_concat)

        return tensor, tensor_concat



class U_net_up_sampling_block(tf.keras.Model):
    def __init__(self, filters,kernel_size = (3,3,2),size = (2,2,2),change_size = True):
        filter_1, filter_2= filters
        self.change_size = change_size

        super(U_net_up_sampling_block, self).__init__()
        self.Up_Sample_3D = tf.keras.layers.UpSampling3D(size = size)
        self.conv_3d_filter_1 = tf.keras.layers.Conv3D(filters = filter_1, kernel_size = kernel_size, strides = [1,1,1], padding = 'same',activation='relu')
        self.batch_filter_1 = tf.keras.layers.BatchNormalization()

        self.conv_3d_filter_2 = tf.keras.layers.Conv3D(filters = filter_2, kernel_size = kernel_size, strides = [1,1,1], padding = 'same',activation='relu')
        self.batch_filter_2 = tf.keras.layers.BatchNormalization()

        if self.change_size:

            self.conv_3d_T = tf.keras.layers.Conv3DTranspose(filters = 128,kernel_size=[2,1,1],strides = [1,1,1],data_format="channels_last",activation = 'relu')
            self.batch_norm= tf.keras.layers.BatchNormalization()




    def call(self, tensor_concat,tensor):
        tensor = self.Up_Sample_3D(tensor)

        if self.change_size:
            tensor = self.conv_3d_T(tensor)
            tensor = self.batch_norm(tensor)

        tensor = tf.concat([tensor_concat,tensor],axis = -1)
        tensor = self.conv_3d_filter_1(tensor)
        tensor = self.batch_filter_1(tensor)

        tensor = self.conv_3d_filter_2(tensor)
        tensor = self.batch_filter_2(tensor)


        return tensor
