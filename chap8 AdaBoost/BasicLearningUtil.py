"""
Basic Learning Util
define the basic learning algorithm
the classifier is BI-categories classifier

"""
import numpy as np


class CLS_Param:
    def __init__(self):
        self.w = None
        self.A_i = -1
        self.A_v = 0
        self.L_or_R = True  # v < value is 1 (l) or -1 (r)
        self.alpha = 0.0

    def copy(self, obj):
        self.w = np.copy(obj.w)
        self.A_i = obj.A_i
        self.A_v = obj.A_v
        self.L_or_R = obj.L_or_R
        self.alpha = obj.alpha


class DecisionStrump_Classifier:

    def __init__(self, maxIter=10, step=50):
        self.__maxIter = maxIter
        self.__cls_params = []
        self.__step_n = step
        pass

    def __calc_err(self, y_pred, y_label, w):
        y_err = np.zeros_like(y_label, dtype=float)
        y_err[np.where(y_pred != y_label)] = 1.0
        r_err = np.dot(w.T, y_err)
        return r_err

    def build(self, x, y):
        assert(x.shape[0] == y.shape[0])
        # step : initialization
        A_n = x.shape[1]
        Row_n = y.shape[0]

        # step : iter
        last_epoch_param = None
        for epoch in range(self.__maxIter):

            epoch_param = CLS_Param()
            if last_epoch_param is None:
                last_epoch_param = CLS_Param()
                epoch_param.w = np.ones_like(y, dtype=float) / float(Row_n)
            else:
                epoch_param.copy(last_epoch_param)

            min_err = 10.0
            y_p_split = None
            # step :
            for i in range(A_n):
                x_A_i = x[:, i]
                x_A_i_max = x_A_i.max()
                x_A_i_min = x_A_i.min()

                step_L = (x_A_i_max - x_A_i_min) / self.__step_n
                x_A_i_v = x_A_i_min
                while x_A_i_v < x_A_i_max:
                    x_A_i_v += step_L
                    # predict the y value

                    # L_or_R is True
                    y_p = np.ones_like(y, dtype=float) * -1.0
                    y_p[np.where(x_A_i < x_A_i_v)] = 1.0
                    x_A_i_err = self.__calc_err(y_p, y, epoch_param.w)

                    # L_or_R is False
                    y_p_r = np.ones_like(y, dtype=float)
                    y_p_r[np.where(x_A_i < x_A_i_v)] = -1.0
                    x_A_i_err_r = self.__calc_err(y_p_r, y, epoch_param.w)

                    if x_A_i_err_r < x_A_i_err:
                        y_p = np.copy(y_p_r)
                        x_A_i_err = x_A_i_err_r
                        epoch_param.L_or_R = False
                    else:
                        epoch_param.L_or_R = True

                    if x_A_i_err < min_err:
                        min_err = x_A_i_err
                        epoch_param.A_i = i
                        epoch_param.A_v = x_A_i_v
                        y_p_split = np.copy(y_p)

            # update weight
            if epoch_param.L_or_R:
                print(f"train epoch {epoch}: error is {min_err}, A[{epoch_param.A_i}] < {epoch_param.A_v}")
            else:
                print(f"train epoch {epoch}: error is {min_err}, A[{epoch_param.A_i}] >= {epoch_param.A_v}")
            alpha = np.log((1.0 - min_err) / max(min_err, 1e-10)) / 2.0
            last_epoch_param.copy(epoch_param)
            Zm = np.sum(epoch_param.w * np.exp(-alpha * y * y_p_split))
            last_epoch_param.w = epoch_param.w * np.exp(-alpha * y * y_p_split) / Zm
            epoch_param.alpha = alpha
            self.__cls_params.append(epoch_param)

    def __calc_once(self, x, cls_param):
        y_p = np.ones((x.shape[0], 1), dtype=float)
        x_i = x[:, cls_param.A_i]
        if cls_param.L_or_R:
            y_p = y_p * -1.0
            y_p[np.where(x_i < cls_param.A_v)] = 1.0
        else:
            y_p[np.where(x_i < cls_param.A_v)] = -1.0
        y_p = y_p * cls_param.alpha
        return y_p

    def predict(self, x):
        y_p = np.zeros((x.shape[0], 1), dtype=float)
        for cls_param in self.__cls_params:
            y_p += self.__calc_once(x, cls_param)
        y_p = np.sign(y_p)
        return y_p
