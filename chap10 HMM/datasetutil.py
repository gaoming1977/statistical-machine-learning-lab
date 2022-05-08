"""
datasetutil.py
用于数据集装载、预处理的工具类
"""
import numpy as np

LOG_ZERO = -3.14e+100
STATES = ['B', 'E', 'M', 'S']


def ZERO_DIC():
    return {'B': 0, 'E': 0, 'M': 0, 'S': 0}


def LOG_ZERO_DIC():
    return {'B': LOG_ZERO, 'E': LOG_ZERO, 'M': LOG_ZERO, 'S': LOG_ZERO}


class DataSet4HMM:
    def __init__(self):
        self.line_tag = {}  # 每一行的BEMS序列
        self.word_tag_dict = {}  # 每个词的BEMS序列
        self.tag_count = ZERO_DIC()  # 每个状态的统计
        self.char_tag_dic = {}  # 每一个字符的BEMS状态序列字典
        pass

    """
    load函数，完成训练集文件的读取，分词标注（BEMS），并保存至两个标注序列数组
    """
    def load(self, filename):
        with open(filename, mode='r', encoding='utf8') as f:
            # step1. 清空序列数组
            self.line_tag.clear()
            self.word_tag_dict.clear()
            self.tag_count = ZERO_DIC()

            line_count = 0
            for line in f:  # 读取一行
                _line_tag = []
                line = line.strip()
                words = line.split(' ')
                for word in words:  # 读取每一个分词，含标点符号
                    w_tag = DataSet4HMM.parse_tag(word)
                    _line_tag += w_tag
                    for i in range(len(word)):
                        _ch = word[i]
                        _tag = w_tag[i]
                        self.tag_count[_tag] += 1
                        if _ch not in self.char_tag_dic:
                            self.char_tag_dic[_ch] = ZERO_DIC()
                        self.char_tag_dic[_ch][_tag] += 1
                    if word in self.word_tag_dict:
                        continue
                    self.word_tag_dict[word] = w_tag
                    pass
                self.line_tag[line_count] = _line_tag
                line_count += 1
                pass
            f.close()
        print(f"the dataset include {line_count} lines")
        print(f"{len(self.word_tag_dict)} words")
        print(f"{len(self.char_tag_dic.keys())} chars")
        print(f"as the count of state is {self.tag_count}")
        return line_count

    def get_char_tag_count(self, ch):
        ch_count = 0
        if ch in self.char_tag_dic:
            for tag in STATES:
                ch_count += self.char_tag_dic[ch][tag]
        return ch_count

    @staticmethod
    def parse_tag(word):
        tag = []
        w_len = len(word)
        if w_len == 1:
            tag = ['S']
        elif w_len == 2:
            tag = ['B', 'E']
        else:
            tag.append('B')
            tag.extend(['M'] * (w_len-2))
            tag.append('E')
        return tag


if __name__ == '__main__':
    print("dataset process...")

    ds = DataSet4HMM()
    lines = ds.load(r"..\txtdata\trainCorpus.txt_utf8")

    pass

