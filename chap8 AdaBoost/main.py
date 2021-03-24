#chap 8 AdaBoost

import DatasetUtil as DU
import AdaBoost as Ada
import numpy as np


if __name__ == "__main__":
    print("\t============ Chap8 AdaBoost ============")

    ds = DU.DATAUtil()
    xx, yy = ds.load(True, r".\dataset.dat")
    print("\t ===== Do Initializing =====")
    model = Ada.Boosting()

    print("\t ===== Do Training =====")
    model.train(xx, yy)

    print("\t ===== Do Testing =====")
    xx_t, yy_t = ds.load(True, r".\testds.dat")
    yy_t = np.squeeze(yy_t)
    y_p = model.predict(xx_t)

    yy_p = ds.y_int2str(y_p)

    print(" Test Result is: ")
    print(yy_p)

    eval_y = np.zeros_like(yy_t, dtype=float)
    eval_y[np.where(yy_t == y_p)] = 1.0
    print(" Test result precision is {:.2%}".format(eval_y.sum()/eval_y.shape[0]))

