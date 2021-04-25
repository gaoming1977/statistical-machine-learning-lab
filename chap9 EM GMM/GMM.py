# EM algorithm in GMM
'''
GMM clustering
reference: https://blog.pluskid.org/?p=39
https://blog.csdn.net/weixin_41566471/article/details/106221915


'''

import numpy as np

class cls_param:
    def __init__(self):
        """
        clustering parameters:
        alpha,
        mu, (1, D)
        sigma, (D, D) 对角矩阵
        """
        self.alpha = 0.0
        self.mu = None
        self.sigma = None
        pass


class GMM:
    def __init__(self, category=3):
        self.__x = None
        self.__K = category
        self.__max_iter = 500
        self.__threshold = 1e-5
        self.__D = 0
        self.__N = 0
        self.__cls_param = {}
        pass

    def __safe_inv(self, sigma):
        try:
            sigma_inv = np.linalg.inv(sigma)
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(sigma)
        return sigma_inv

    def __calc_fi(self, x_j, mu_k, sigma_k):
        """
        多维高斯密度函数，参考
        https://www.zhihu.com/question/21024811
        calc r(jk)
        x_j: the j row of x(1, 4)
        :return:
        """
        # calc multi dim gaussian
        try:
            _, sigma_k_det = np.linalg.slogdet(sigma_k)
            temp0 = 1.0 / np.sqrt(np.exp(sigma_k_det))
            temp = 1.0 / ((2.0 * np.pi) ** (self.__D / 2.0))
            temp1 = np.dot(np.dot((x_j - mu_k), self.__safe_inv(sigma_k)), (x_j - mu_k).T)
            fi_jk = temp * temp0 * np.exp(-0.5 * temp1)
        except (OverflowError, ZeroDivisionError):
            print('break')
        return fi_jk

    def __calc_E(self, x):
        """
        :param x: x (N, D)
        :return: gamma (N, K)
        """
        gamma = np.zeros((self.__N, self.__K), dtype=float)
        """
        for k in range(self.__K):
            alpha_k = self.__cls_param[k].alpha
            mu_k = self.__cls_param[k].mu
            sigma_k = self.__cls_param[k].sigma
            for j in range(self.__N):
                x_j = x[j, :]
                gamma[j, k] = alpha_k * self.__calc_fi(x_j, mu_k, sigma_k)
            gamma_k = np.sum(gamma[:, k])
            gamma[:, k] /= gamma_k
        return gamma
        """
        for j in range(self.__N):
            x_j = x[j, :]
            for k in range(self.__K):
                alpha_k = self.__cls_param[k].alpha
                mu_k = self.__cls_param[k].mu
                sigma_k = self.__cls_param[k].sigma
                fi_jk = self.__calc_fi(x_j, mu_k, sigma_k)
                gamma[j, k] = alpha_k * fi_jk
            gamma_k_sum = np.sum(gamma[j, :])
            gamma[j, :] /= gamma_k_sum
        return gamma


    def __calc_M(self, x, gamma):
        for k in range(self.__K):
            gamma_k = gamma[:, k]
            gamma_k_sum = np.sum(gamma_k)

            #1. calc mu
            mu_k = np.zeros_like(self.__cls_param[k].mu)
            for j in range(self.__N):
                x_j = x[j, :]
                mu_k += gamma_k[j] * x_j
            self.__cls_param[k].mu = mu_k / gamma_k_sum
            """
            gamma_k = gamma[:, k]
            gamma_k = np.reshape(gamma_k, (-1, 1))
            gamma_k_sum = np.sum(gamma_k)
            self.__cls_param[k].mu = np.dot(gamma_k.T, x) / gamma_k_sum
            """
            #2. calc sigma
            mu_k = self.__cls_param[k].mu
            sigma_k = np.zeros_like(self.__cls_param[k].sigma)
            for j in range(self.__N):
                x_j = x[j, :]
                sigma_k += gamma_k[j] * np.outer((x_j - mu_k), (x_j - mu_k).T)
            self.__cls_param[k].sigma = sigma_k / gamma_k_sum
            """
            self.__cls_param[k].sigma = np.dot((gamma_k * (x - mu_k)).T, (x - mu_k)) / gamma_k_sum
            """
            #3. calc alpha
            self.__cls_param[k].alpha = gamma_k_sum / self.__N


        pass

    def __calc_Likelihood(self, x):
        rll = 0
        for j in range(self.__N):
            x_j = x[j, :]
            p_x_j = 0
            for k in range(self.__K):
                sigma_k = self.__cls_param[k].sigma
                alpha_k = self.__cls_param[k].alpha
                mu_k = self.__cls_param[k].mu
                fi_jk = self.__calc_fi(x_j, mu_k, sigma_k)
                p_x_j += alpha_k * fi_jk
            rll += np.log(p_x_j)
        return rll
        """
        rll = 0
        for k in range(self.__K):
            gamma_k = gamma[:, k]
            gamma_k = np.reshape(gamma_k, (-1, 1))
            gamma_k_sum = np.sum(gamma_k)

            rll = gamma_k_sum * np.log(self.__cls_param[k].alpha)
            rll -= gamma_k_sum * np.log((2.0 * np.pi) ** (self.__D / 2.0))
            _, sigma_logdet = np.linalg.slogdet(self.__cls_param[k].sigma)
            rll -= gamma_k_sum * sigma_logdet / 2.0
            temp = 0
            for j in range(self.__N):
                x_j = x[j, :]
                mu_k = self.__cls_param[k].mu
                sigma_k = self.__cls_param[k].sigma
                fi_jk = np.dot(np.dot((x_j - mu_k), self.__safe_inv(sigma_k)), (x_j - mu_k).T)
                temp += gamma_k[j] * 0.5 * fi_jk
            rll -= temp
        return rll
        """

    def train(self, x):
        self.__x = x
        self.__N, self.__D = x.shape
        self.__cls_param.clear()

        rand_alpha = np.random.rand(self.__K)
        rand_alpha_sum = rand_alpha.sum()
        rand_alpha /= rand_alpha_sum
        for k in range(self.__K):
            self.__cls_param[k] = cls_param()
            self.__cls_param[k].alpha = rand_alpha[k]
            self.__cls_param[k].mu = np.random.rand(self.__D)
            self.__cls_param[k].sigma = np.eye(self.__D, dtype=float)

        epoch = 0
        l_L = 0
        while epoch < self.__max_iter:
            print(f"training...  epoch{epoch} ...")
            gamma = self.__calc_E(x)
            self.__calc_M(x, gamma)

            # calc L()
            p_L = self.__calc_Likelihood(x)

            if np.abs(p_L - l_L) < self.__threshold:
                print(f"epoch {epoch} likelihood converge, loop terminate!!")
                break
            print(f"this epoch likelihood is {p_L}")
            l_L = p_L
            epoch += 1
            pass


    def predict(self, xx):
        gmma = self.__calc_E(xx)
        y_pred = np.argmax(gmma, axis=1)
        return y_pred
