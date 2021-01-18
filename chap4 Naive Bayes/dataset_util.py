# dataset utility functions
import numpy as np


g_Labels = ['largeDoses', 'smallDoses', 'didntLike']

def fun_zip(x, y, z):
    ret = (float(x), float(y), float(z))
    return ret


def load_dataset(filename):
    """load data"""
    xx = []
    with open(filename, mode='r',) as f:
        for line in f:
            xx.append(line.strip().split())
        x1, x2, x3, y = zip(*xx)
        return np.array(list(map(fun_zip, x1, x2, x3))), np.array(y)
    pass

def load_test_ds(filename):
    xx = []
    with open(filename, mode='r') as f:
        for line in f:
            xx.append(line.strip().split())
        x1, x2, x3, _ = zip(*xx)
        return np.array(list(map(fun_zip, x1, x2, x3)))
    pass


class DataUtil:
    def __init__(self, x):
        self._x_max = np.max(x, axis=0)
        self._x_min = np.min(x, axis=0)
        pass

    def discrete_variant_EqtWidth(self, x, K=100):
        """
        tranform variant from continuous variant to discrete variant by column
        x1 -> xx1,
        equal width method,
        future, try clustering algorithm
        for Bayes method.
        :param x: X[x1, x2, ..., xn]
        K is interval
        :return: XX[xx1, xx2, ..., xxn]
        """
        inter = 1.0 / float(K)

        # normalized x and discrete
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - self._x_min[j]) / (self._x_max[j] - self._x_min[j])
            x[:, j] = x[:, j] / inter

        # int(array)
        xx = x.astype(int)
        return xx
        pass


if __name__ == '__main__':

    x, y = load_dataset(r'.\dataset.dat')
    test_x = load_test_ds(r'.\testds.dat')
    print(test_x)
    pass