"""
chapter10 HMM (Hidden Markov Model)
task: using HMM to split the Chinese sentences
- 1 supervised learning
- 2 unsupervised learning

dataset:

"""
import HMM as HMM

if __name__ == '__main__':
    print("\t============ Chap10 HMM (Hidden Markov Model) ============")
    print("DESCRIPTION: this model is designed for CHINESE sentence segment by HMM method")
    print("描述: 本模型采用隐形马尔可夫模型实现中文分词")
    model = HMM.HMMSegment()
    try:
        while True:
            print("\r\t====== INPUT MENU ========")
            print("\t1: train model \n \t2: save model \n \t3: load model \n \t4: test model \n\tpress any other keys to exit")
            i_choice = input()
            if i_choice == '1':
                model.train(r"..\txtdata\trainCorpus.txt_utf8")
            elif i_choice == '2':
                model.save(r".\HMMmodel.pkl")
                pass
            elif i_choice == '3':
                model.load(r".\HMMmodel.pkl")
                pass
            elif i_choice == '4':
                print("Please input your sentence:")
                sentence = input()
                words = model.doCut(sentence)
                print("HMM cut result is ")
                print(f"{words}")
                pass
            else:
                break
    except:
        exit()

    # words = model.doCut("这里是北京时间八点整")
    # words = model.doCut("状态空间中经过从一个状态到另一个状态的转换的随机过程。")
    # words = model.doCut("他们硕士毕业于北京航空航天大学。你怎么知道的？哈哈哈！")
    #print("HMM cut result is ")
    # print(f"{words}")


    pass