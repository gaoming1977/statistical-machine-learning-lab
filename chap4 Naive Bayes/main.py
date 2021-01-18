# chapter 4 Naive Bayes machine learning
# using  dataset train Bayes Classifier


import numpy as np
import dataset_util as DS
import NaiveBayes as NB

g_K_param = 100

if __name__ == '__main__':
    print("\t\t\t============ Chap4 Naive Bayes ============")
    x, y = DS.load_dataset(r'.\dataset.dat')
    ## convert y to label
    y = np.array(list(map(lambda y1: DS.g_Labels.index(y1), y)))
    yy = np.reshape(y, (-1, 1))

    du = DS.DataUtil(x)
    xx = du.discrete_variant_EqtWidth(x, g_K_param)

    model = NB.Naive_Bayes(g_K_param)
    model.train(xx, yy)

    x_t = DS.load_test_ds(r'.\testds.dat')
    xx_t = du.discrete_variant_EqtWidth(x_t)
    y_t, y_t_p = model(xx_t)
    yy_t = list(DS.g_Labels[int(y1)] for y1 in y_t)
    print(">>>>Test DATA is: >>>>")
    print(x_t)
    print(">>>>Navie Bayes give the answer is :>>>>")
    print(yy_t)
    print(">>>>Navie Bayes answer possibility is :>>>>")
    print(y_t_p)

    pass