# main function
import numpy as np

import dataset_util as DS
import KNN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

g_Labels = ['largeDoses', 'smallDoses', 'didntLike']
g_color = ['red', 'orange', 'gray', 'blue']


if __name__ == '__main__':
    print("\t\t\t============ Chap3 KNN ============")
    x, y = DS.load_dataset(r'.\dataset.dat')
    ## convert y to label
    y = np.array(list(map(lambda y1: g_Labels.index(y1), y)))
    y = np.reshape(y, (-1, 1))

    model = KNN.KNN_Model()
    model.train(x, y)

    x_test = DS.load_test_ds(r'.\testds.dat')
    print(x_test)
    y_test = model(x_test)

    yy_test = list(g_Labels[int(y1)] for y1 in y_test)
    print(yy_test)

    '''draw plot'''
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    for i in range(len(y)):
        _x = x[i]
        _y = int(y[i])
        ax.scatter(_x[0], _x[1], _x[2], c=g_color[_y])

    #draw test data
    for i in range(len(x_test)):
        _x = x_test[i]
        ax.scatter(_x[0], _x[1], _x[2], c=g_color[3])

    plt.show()

    pass