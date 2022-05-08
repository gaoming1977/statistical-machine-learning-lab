"""
HMM.py
隐马尔可夫工具类，基于HMM模型，实现中文分词。
训练集为：trainCorpus.txt_utf8 “人民日报标注训练集”

"""
import numpy as np
import datasetutil as DS
import pickle

'''
状态列表，对应于每一个字符的状态，B - 开始， E - 结束， M - 中间， S - 单字
例如： 这S 是S 一个BE 隐马尔科夫BMMMME 的S 例子BE 。S 
'''

class HMMSegment:
    def __init__(self):
        self.__A = {}  #状态转移概率 BEMS x BEMS (16 x 16)
        self.__Pi = {}   #初始状态概率,每一句话的第一个字符BEMS概率 (4 x 1)
        self.__B = {}  #观测概率，也称为发射概率
        pass

    def train(self, filename):
        print("model training >>>>>")
        # 清空
        self.__A.clear()
        self.__Pi.clear()
        self.__B.clear()
        print("loading dataset ...")
        ds = DS.DataSet4HMM()
        # load函数读取数据集
        line_c = ds.load(filename)

        # A转移矩阵初始化
        for key_x in DS.STATES:
            self.__A[key_x] = {}
            for key_y in DS.STATES:
                self.__A[key_x][key_y] = 0.0

        # 遍历训练数据， 每一行的每一个词
        # self.__Pi = {'B': 0, 'E': 0, 'M': 0, 'S': 0}
        for key in DS.STATES:
            self.__Pi[key] = 0
        for i in range(line_c):  # 遍历所有行,统计状态
            line_tag = ds.line_tag[i]
            self.__Pi[line_tag[0]] += 1  # 每句话的首字符标签，用于PI计算
            for j in range(len(line_tag) - 1):  # 每行标签的转移统计，用于A计算
                self.__A[line_tag[j]][line_tag[j+1]] += 1
                pass
        # 初始化PI
        for key in self.__Pi:
            if self.__Pi[key] == 0:
                self.__Pi[key] = DS.LOG_ZERO
            else:
                self.__Pi[key] = np.log(self.__Pi[key] / line_c)
        print(f"the Pi initial value is {self.__Pi}")
        pass
        # 初始化A转移矩阵
        print("the A initial value is as below :")
        for key_x in DS.STATES:
            for key_y in DS.STATES:
                if self.__A[key_x][key_y] == 0:
                    self.__A[key_x][key_y] = DS.LOG_ZERO
                else:
                    self.__A[key_x][key_y] = np.log(self.__A[key_x][key_y] / ds.tag_count[key_x])
            print(f"{key_x}: {self.__A[key_x]}")
        pass
        # B观测概率矩阵初始化
        print(f"the B initial value is as below : <random 10>")
        for _ch in ds.char_tag_dic:
            self.__B[_ch] = DS.ZERO_DIC()
            _ch_tag_count = ds.get_char_tag_count(_ch)
            for key in DS.STATES:
                if ds.char_tag_dic[_ch][key] == 0 or _ch_tag_count == 0:
                    self.__B[_ch][key] = DS.LOG_ZERO
                else:
                    self.__B[_ch][key] = np.log(ds.char_tag_dic[_ch][key] / _ch_tag_count)
        """
        _b_sample = random.sample(list(self.__B.keys()), 10)
        for _ch in _b_sample:
            print(f"{_ch}: {self.__B[_ch]}")
        """
        pass
    #################################
        print("<<<<<< model train finished ")

    def load(self, filename):
        self.__A.clear()
        self.__Pi.clear()
        self.__B.clear()
        with open(filename, mode = 'rb') as f:
            self.__Pi = pickle.load(f)
            self.__B = pickle.load(f)
            self.__A = pickle.load(f)

            f.close()
        pass

    def save(self, filename):
        with open(filename, mode='wb') as f:
            pickle.dump(self.__Pi, f)
            pickle.dump(self.__B, f)
            pickle.dump(self.__A, f)

            f.close()
            pass

        pass

    '''
    Viterbi算法，实现从观察序列到状态序列的算法
    输入：观察序列  -- 一句话
    输出：状态序列  -- 'BEMS'序列
    类型：protected
    '''
    def _viterbi(self, sentence):
        print(">>> Enter Viterbi >>>")
        s_len = len(sentence)
        _tag = ['S'] * s_len
        _ch_state_prob = []
        _ch_path = []

        # step1: 计算首字符的初始状态概率
        _ch = sentence[0]
        if _ch not in self.__B:  # 如果字符没有在训练集字库（B矩阵）中，则假设此字符为独立字符,即S
            _ch_state_b = {'B': DS.LOG_ZERO, 'E': DS.LOG_ZERO, 'M': DS.LOG_ZERO, 'S': 0}
        else:
            _ch_state_b = self.__B[_ch]

        _tmp_prob = DS.LOG_ZERO_DIC()
        for key in DS.STATES:
            _tmp_prob[key] = _ch_state_b[key] + self.__Pi[key]
        _ch_state_prob.append(_tmp_prob)
        _ch_path.append(DS.ZERO_DIC())

        # step2: 计算从第二个字符开始的状态概率，取最大值
        for i in range(1, s_len):
            _ch = sentence[i]
            if _ch not in self.__B:  # 如果字符没有在训练集字库（B矩阵）中，则假设此字符为独立字符,即S
                _ch_state_b = DS.LOG_ZERO_DIC()
                _ch_state_b['S'] = 0
            else:
                _ch_state_b = self.__B[_ch]

            _ch_state_prob.append(DS.LOG_ZERO_DIC())
            _ch_path.append(DS.ZERO_DIC())
            for key0 in DS.STATES:
                for key1 in DS.STATES:
                    _tmp_prob_i = _ch_state_prob[i-1][key1] + self.__A[key1][key0] + _ch_state_b[key0]
                    if _tmp_prob_i > _ch_state_prob[i][key0]:
                        _ch_state_prob[i][key0] = _tmp_prob_i
                        _ch_path[i][key0] = key1
            pass

        # step3: 回溯最大概率路径
        i = s_len - 1  # 最后一个字符
        max_prob = DS.LOG_ZERO
        for key in ['S', 'E']:  # 正则干预，结尾字符状态只能为S或E
            if _ch_state_prob[i][key] > max_prob:
                max_prob = _ch_state_prob[i][key]
                _tag[i] = key

        for i in range(s_len-1, 0, -1):
            _tag[i-1] = _ch_path[i][_tag[i]]
            pass

        print(f"{_tag}")
        print(f"<<< leave Viterbi <<<")
        return _tag
        pass

    def doCut(self, sentence):
        seg = []
        words = []
        print(">>the HMM model Viterbi tag output is as below:>>")
        print("\t" + sentence)
        tag = self._viterbi(sentence)

        for i in range(len(tag) - 1):
            ch = sentence[i]
            seg.append(ch)
            if tag[i] == 'S' or tag[i] == 'E':
                seg.append('/')
        seg.append(sentence[-1])
        seg = ''.join(seg)
        print(f"\t{seg}")
        return seg.split("/")



if __name__ == "__main__":

    print("HMM Model")
    model = HMMSegment()
    model.train(r"..\txtdata\trainCorpus.txt_utf8")

    # words = model.doCut("这里是北京时间八点整")
    words = model.doCut("状态空间中经过从一个状态到另一个状态的转换的随机过程。")
    # words = model.doCut("他们硕士毕业于北京航空航天大学。你怎么知道的？哈哈哈！")
    # words = model.doCut("腾讯与阿里都在新零售产业大举布局")
    print("HMM cut result is ")
    print(f"{words}")
    pass


