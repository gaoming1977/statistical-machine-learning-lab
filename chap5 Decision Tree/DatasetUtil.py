# process iris dataset
import numpy as np

class DATAUtil:
    g_categories = ["setosa", "versicolor", "virginica"]
    g_SuperParam = {'K': 30, }

    def __init__(self):
        self.__xx = None
        self.__yy = None

        self.__xx_max = None
        self.__xx_min = None

    @staticmethod
    def __load_data_c(filename):
        try:
            data = []
            with open(filename, mode='r') as f:
                for line in f:
                    data.append(line.strip().split())
                f.close()
            _, x1, x2, x3, y1, y2 = zip(*data)
            x1 = list(map(lambda a: float(a), x1))
            x2 = list(map(lambda a: float(a), x2))
            x3 = list(map(lambda a: float(a), x3))

            y1 = list(map(lambda a: float(a), y1))
            y2 = list(map(lambda a: DATAUtil.y_str2int(a), y2))

            xx = np.array(list(zip(x1, x2, x3, y1)))
            yy = np.array(list(zip(y1, y2)))
            pass
        except IOError as err:
            print(err)
            return None
        return xx, yy

    @staticmethod
    def __load_data_r(filename):
        try:
            data = []
            with open(filename, mode='r') as f:
                for line in f:
                    data.append(line.strip().split())
                f.close()
            _, x1, x2, x3, y1, y2 = zip(*data)
            x1 = list(map(lambda a: float(a), x1))
            x2 = list(map(lambda a: float(a), x2))
            x3 = list(map(lambda a: float(a), x3))

            y1 = list(map(lambda a: float(a), y1))
            y2 = list(map(lambda a: DATAUtil.y_str2int(a), y2))

            xx = np.array(list(zip(x1, x2, x3, y2)))
            yy = np.array(list(zip(y1, y2)))
            pass
        except IOError as err:
            print(err)
            return None
        return xx, yy

    def load(self, c_r, filename):
        if c_r:
            xx, yy = DATAUtil.__load_data_c(filename)
        else:
            xx, yy = DATAUtil.__load_data_r(filename)
        if self.__xx is None:
            self.__xx = xx
        else:
            self.__xx = np.concatenate((self.__xx, xx), axis=0)

        if self.__yy is None:
            self.__yy = yy
        else:
            self.__yy = np.concatenate((self.__yy, yy), axis=0)

        # return max min by column
        self.__xx_max = np.max(self.__xx, axis=0)
        self.__xx_min = np.min(self.__xx, axis=0)

        return xx, yy

    def release(self):
        self.__xx = None
        self.__xx_max = None
        self.__xx_min = None
        self.__yy = None

    def discrete_vec(self, x, K):
        inter = 1.0 / float(K-1)
        temp_x = np.copy(x)
        # normalized x and discrete
        for j in range(x.shape[1]):
            temp_x[:, j] = (temp_x[:, j] - self.__xx_min[j]) / (self.__xx_max[j] - self.__xx_min[j])
            temp_x[:, j] = temp_x[:, j] / inter + 0.5  # decimals round-up

        # int(array)
        xx = temp_x.astype(int)
        return xx

    @staticmethod
    def y_str2int(a):
        return DATAUtil.g_categories.index(a)

    @staticmethod
    def y_int2str(a):
        return list(DATAUtil.g_categories[int(a1)] for a1 in a)
