# chapter 5 Decision Tree

import numpy as np
import DatasetUtil as DS
import DecisionTree
import PlotUtil

if __name__ == '__main__':
    print("\t============ Chap5 Decision Tree (CART) ============")
    ds = DS.DATAUtil()

    #initial plt
    print("\n\t\t\t OPTION 1: using CART to forecast CLASSIFICATION.")
    print("\t\t\t OPTION 2: using CART to forecast REGRESSION.")
    print("\t\t\t OPTION x: exit.")
    c = input("\n\t\t\t Please input your option:")

    if c == '1':
        print("\t\t STEP1: Load dataset...")
        xx, yy = ds.load(True, r'.\dataset.dat')
        xx = ds.discrete_vec(xx, K=DS.DATAUtil.g_SuperParam['K'])
        y_c = yy[:, 1]

        xx_t, y_t = ds.load(True, r'.\testds.dat')
        x_t = ds.discrete_vec(xx_t, K=DS.DATAUtil.g_SuperParam['K'])
        y_t_c = y_t[:, 1]

        print("\t\t STEP2: Build CART Tree...")
        model = DecisionTree.CART(c_r=True)
        model.train(xx, y_c)
        print("\t\t STEP3: Display CART Tree...")
        PlotUtil.init_Plot("CART Decision Tree - (C)")
        model.display_Tree()
        PlotUtil.show_Plot()
        print("\t\t STEP4: Predict by CART Tree...")
        y_p = model.predict(x_t)
        y_p = y_p.flatten()
        if y_p.shape[0] > 0:
            eval_y = np.zeros_like(y_p, dtype=int)
            eval_y = np.where((y_p == y_t_c), 1, 0)

            y_p = ds.y_int2str(y_p)
            y_p = np.reshape(y_p, (-1, 1))
            print("The predict result is:")
            print(y_p)
            print("The predict precision is {:.2%}".format(eval_y.sum()/eval_y.shape[0]))
            print(eval_y)
        pass

    elif c == '2':
        print("\t\t STEP1: Load dataset...")
        xx, yy = ds.load(False, r'.\dataset.dat')
        y_r = yy[:, 0]

        xx_t, y_t = ds.load(False, r'.\testds.dat')
        y_t_r = y_t[:, 0]

        print("\t\t STEP2: Build CART Tree...")
        model = DecisionTree.CART(c_r=False)
        model.train(xx, y_r)
        print("\t\t STEP3: Display CART Tree...")
        PlotUtil.init_Plot("CART Decision Tree - (R)")
        model.display_Tree()
        PlotUtil.show_Plot()
        print("\t\t STEP4: Predict by CART Tree...")
        y_p = model.predict(xx_t)
        y_p = y_p.flatten()
        if y_p.shape[0] > 0:
            eval_y = np.zeros_like(y_p, dtype=float)
            y_dist = np.sqrt(np.sum((y_t_r - y_p)**2))
            y_norm = np.linalg.norm(y_t_r)
            y_p = np.reshape(y_p, (-1, 1))
            print("The predict result is:")
            print(y_p)
            print("The predict precision is {:.2}".format(1.0 - y_dist/y_norm))
            pass
        pass
    else:
        exit(0)

