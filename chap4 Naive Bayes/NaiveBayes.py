# naive bayes algorithm
# Task Description: using Naive Bayes to

import numpy as np

"""
P(X,Y) = P(Y)P(X|Y) Bayes formula

P(X,Y) the joint possibility
P(Y) the y priority possibility
P(X|Y) the condition possibility

this class use train data to get P(Y) and P(X|Y) ,then predict the result P(X, Y)
and return the max possibility of P(X, Y)
#
"""
class Naive_Bayes():

    def __init__(self, K=100, r=1):
        self._K = K
        self._r = r  # Laplace smoothing constant

        # P(Y), y priority possibility
        self._y_possibility = {}

        # P(X|Y) , x|y condition possibility
        self._x_y_possibility = {}
        pass

    def __call__(self, xx):
        yy = self._calc_forward(xx)
        return yy[:, 0], yy[:, 1]
        pass

    def train(self, x_train, y_train):
        """
        According to train data (x_train, y_train), calculate the priority possibility,
            P(Y=c1, c2,...c)
        and the condition possibility,
            P(X1=a1, a2,...a | Y=c1),
            P(X2=b1, b2,...b | Y=c1),
            P(X3=d1, d2,...d | Y=c1),

            P(X1=a1, a2,...a | Y=c2),
            P(X2=b1, b2,...b | Y=c2),
            P(X3=d1, d2,...d | Y=c2),

            P(X1=a1, a2,...a | Y=c3),
            P(X2=b1, b2,...b | Y=c3),
            P(X3=d1, d2,...d | Y=c3),
        :param x_train: X input shape [N, 3] normalized vector (X1, X2, X3) ,[0 99]
        :param y_train: Y label shape [N, 1] (Y) ,[0,2]

        ***y_train should include all "c1, c2, ...c" data***
        :return:
        """
        print("---- training begin -----")
        # calc y priority possibility
        size_y = y_train.shape[0]
        size_x = x_train.shape[1]  # get x column number
        assert(size_y > 3)
        c_unique, c_count = np.unique(y_train, return_counts=True)
        i = 0
        for c_i in c_unique:
            y_p = c_count[i] / size_y
            i += 1
            print(f">>>calc Y{c_i} priority possibility is {y_p}>>>")
            self._y_possibility[c_i] = y_p

            c_i_idx, _ = np.where(y_train == c_i)
            x_c_i = x_train[c_i_idx, :]

            x_y_p = np.zeros((size_x, self._K), dtype=float)
            for ii in range(self._K):
                for jj in range(size_x):
                    temp = x_c_i[:, jj]
                    temp = np.where((temp == ii), 1, 0)
                    x_y_p[jj][ii] = temp.sum() / x_c_i.shape[0]

            self._x_y_possibility[c_i] = x_y_p
            print(f">>>cal Y={c_i}|X condition possibility>>>")
        pass

    def _calc_forward(self, xx):
        yy = np.zeros((xx.shape[0], 2), dtype=float)
        size_x = xx.shape[1]
        for i in range(xx.shape[0]):
            p_max = 0
            c_i_max = 0
            for c_i in self._y_possibility.keys():
                p_y = self._y_possibility[c_i]
                x_y_p = self._x_y_possibility[c_i]

                p_xy = p_y
                for jj in range(size_x):
                    x_i = xx[i, jj]
                    p_xy *= x_y_p[jj, x_i]
                if p_max < p_xy:
                    p_max = p_xy
                    c_i_max = c_i

            yy[i, 0] = c_i_max
            yy[i, 1] = p_max
        return yy


        pass
