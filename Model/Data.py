import numpy as np

class dataset(object):

    def __init__(self,config):
        self._x_train_item = None
        self._x_train_package = None

        self._y_train_item_match = None
        self._y_train_package_match = None
        self._y_train_item_non_match = None
        self._y_train_package_non_match = None

        self._config = config


    def generate_data(self):
        batch_size = self.config.batch_size
        batch_match = self.config.batch_match
        batch_non_match = self.config.batch_non_match

        self.x_train_item = np.random.normal(0,1,[batch_size,30,30,30,1])
        self.x_train_package = np.random.normal(0,1,[batch_size,30,30,30,1])

        for i in range(batch_size):
            idx_item = np.random.choice(30,batch_match + batch_non_match,replace= False)
            idy_item = np.random.choice(30,batch_match + batch_non_match,replace= False)
            idz_item = np.random.choice(30,batch_match + batch_non_match,replace= False)

            idx_package = np.random.choice(30,batch_match + batch_non_match,replace= False)
            idy_package = np.random.choice(30,batch_match + batch_non_match,replace= False)
            idz_package = np.random.choice(30,batch_match + batch_non_match,replace= False)

            try:
                ele_match = np.stack((idx_item,idy_item,idz_item),axis = -1)[:batch_match,:][None,...]
                ele_non_match = np.stack((idx_item,idy_item,idz_item),axis = -1)[-batch_non_match:,:][None,...]
                self.y_train_item_match = np.concatenate((self.y_train_item_match,ele_match),axis = 0)
                self.y_train_item_non_match = np.concatenate((self.y_train_item_non_match,ele_non_match),axis = 0)

                ele_match = np.stack((idx_package,idy_package,idz_package),axis = -1)[:batch_match,:][None,...]
                ele_non_match = np.stack((idx_package,idy_package,idz_package),axis = -1)[-batch_non_match:,:][None,...]
                
                self.y_train_package_match = np.concatenate((self.y_train_package_match,ele_match),axis = 0)

                self.y_train_package_non_match = np.concatenate((self.y_train_package_non_match,ele_non_match),axis = 0)


            except:
                self.y_train_item_match = np.stack((idx_item,idy_item,idz_item),axis = -1)[:batch_match,:][None,...]
                self.y_train_item_non_match =  np.stack((idx_item,idy_item,idz_item),axis = -1)[-batch_non_match:,:][None,...]

                self.y_train_package_match = np.stack((idx_package,idy_package,idz_package),axis = -1)[:batch_match,:][None,...]
                self.y_train_package_non_match =  np.stack((idx_package,idy_package,idz_package),axis = -1)[-batch_non_match:,:][None,...]


    @property
    def x_train_item(self):
        return self._x_train_item

    @property
    def x_train_package(self):
        return self._x_train_package

    @property
    def y_train_item_match(self):
        return self._y_train_item_match

    @property
    def y_train_package_match(self):
        return self._y_train_package_match

    @property
    def y_train_item_non_match(self):
        return self._y_train_item_non_match

    @property
    def y_train_package_non_match(self):
        return self._y_train_package_non_match

    @property
    def config(self):
        return self._config


    @x_train_item.setter
    def x_train_item(self,value):
        self._x_train_item = value

    @x_train_package.setter
    def x_train_package(self,value):
        self._x_train_package = value

    @y_train_item_match.setter
    def y_train_item_match(self,value):
        self._y_train_item_match = value

    @y_train_package_match.setter
    def y_train_package_match(self,value):
        self._y_train_package_match = value

    @y_train_item_non_match.setter
    def y_train_item_non_match(self,value):
        self._y_train_item_non_match = value

    @y_train_package_non_match.setter
    def y_train_package_non_match(self,value):
        self._y_train_package_non_match = value

    @config.setter
    def config(self,value):
        self._config = value

    # @x_train.setter
    # def x_train(self,value):
    #     self._x_train = value

    # @y_train.setter
    # def y_train(self,value):
    #     self._y_train = value

    # @config.setter
    # def config(self,value):
    #     self._config = value



