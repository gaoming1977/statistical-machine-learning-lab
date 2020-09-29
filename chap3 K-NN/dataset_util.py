# dataset utility functions
import numpy as np


def fun_zip(x, y, z):
    ret = (float(x), float(y), float(z))
    return ret


def load_dataset(filename):
    """load data"""
    xx = []
    with open(filename, mode='r',) as f:
        for line in f:
            xx.append(line.strip().split())
        x1, x2, x3, y = zip(*xx)
        return np.array(list(map(fun_zip, x1, x2, x3))), np.array(y)
    pass

def load_test_ds(filename):
    xx = []
    with open(filename, mode='r') as f:
        for line in f:
            xx.append(line.strip().split())
        x1, x2, x3, _ = zip(*xx)
        return np.array(list(map(fun_zip, x1, x2, x3)))
    pass

if __name__ == '__main__':

    """
    class node:
        def __init__(self):
            self.a = 0
            self.a1 = [0, 0]


    a_stack = []
    for i in range(10):
        a_node = node()
        a_node.a = i
        a_node.a1 = [a_node.a, i*10]
        a_stack = np.append(a_stack, [a_node, 10 + np.random.randint(0, 10)])

    print(a_stack)
    a_stack = np.reshape(a_stack, (-1, 2))
    print(a_stack)
    a_stack = a_stack[np.argsort(a_stack[:, -1], axis=0)]
    print(a_stack)

    a_stack = a_stack[0: 5, :]
    print(a_stack)

    b_stack = np.empty((0, 2))
    print(b_stack)
    b_stack = np.concatenate((b_stack, a_stack[0:3, :]), axis=0)
    print(b_stack)
    """

    x, y = load_dataset(r'.\dataset.dat')
    test_x = load_test_ds(r'.\testds.dat')
    print(test_x)
    pass