'''
Name:
    adaboost algorithm
Description:
    adaboost algorithm, using decision stump as weak basic learning algorithm
    using iris as training and test dataset
    classification
    when the sample has multi categories(N categories), it will set up N*(N-1)/2 classifier group.
    each classifier group is BI-categories classifier
Author:
    Gao Ming
Data:
    2021-3
'''

import numpy as np
import re
import BasicLearningUtil as BLU


class Boosting:

    def __init__(self):
        self.__classifier = {}
        self.__cat_n = 0
        print("model is initialized...")
        pass

    def train(self, x, y):
        print("Training begin >>>\n")
        assert (x.shape[0] == y.shape[0])
        y_u = np.unique(y)
        self.__cat_n = y_u.shape[0]
        print(f"\t category number is {self.__cat_n}")
        print(f"\t so, there will be {self.__cat_n * (self.__cat_n -1) /2} classifiers\n")
        for i in range(self.__cat_n):
            for j in range(i+1, self.__cat_n):
                cls_name = f"{y_u[i]}-{y_u[j]}"
                classifier = BLU.DecisionStrump_Classifier()

                # select x, y for classifier and transform y as {-1, +1}

                '''
                注意：Numpy中的Where函数，多条件只能通过'& ', '|' 进行连接，不能用and, or
                '''
                row_i = np.where((y == y_u[i]) | (y == y_u[j]))[0]
                xx = x[row_i, :]
                yy = y[row_i, :]

                yy[np.where(yy == y_u[j])] = -1.0
                yy[np.where(yy == y_u[i])] = 1.0

                print(f"build {cls_name} classifier...")
                classifier.build(xx, yy)
                self.__classifier[cls_name] = classifier
        print("\n<<< Training end")

    def predict(self, x):
        y_p = np.zeros((x.shape[0], self.__cat_n), dtype=float)

        for key in self.__classifier:
            cls = self.__classifier[key]
            y_p_k = cls.predict(x)
            it = re.findall(r'\d', key)
            for i in range(y_p.shape[0]):
                if y_p_k[i] == -1.0:
                    y_p[i, int(it[1])] += 1
                elif y_p_k[i] == 1.0:  # == 1.0
                    y_p[i, int(it[0])] += 1
                else:
                    print(f"ERROR: classifier{key} predict error, value is {y_p_k[i]}")

        r_y_p = np.argmax(y_p, axis=1)
        return r_y_p
        pass
    pass