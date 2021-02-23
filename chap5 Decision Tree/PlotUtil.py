# matplot utility process

import matplotlib.pyplot as plt

nodestyle_internal = dict(boxstyle="circle", fc='yellow', ec='gray')
nodestyle_leaf = dict(boxstyle="round4", fc='green', ec='gray')
arrow_args = dict(arrowstyle='<-')
pos_xoffset = 0.15
pos_yoffset = 0.1

def init_Plot(title):
    fig = plt.figure(figsize=(6.4*2, 4.8*2), dpi=100)
    fig.clf()

    plt.title(title, fontsize=10)
    plt.axis('off')


def draw_Node(nodeText, centerPt, parentPt, nodeType):
    plt.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
                  xytext=centerPt, textcoords='axes fraction',
                  va='center', ha='center', bbox=nodeType, arrowprops=arrow_args, fontsize=8)
    midText = ''
    if centerPt[0] < parentPt[0]:
        midText = 'T'
        plt.text((parentPt[0]+centerPt[0])/2.0, (parentPt[1]+centerPt[1])/2.0,
                  midText, va='center', ha='center', rotation=30, color='green',
                  bbox=dict(facecolor='white', ec='white'), fontsize=8)
    elif centerPt[0] > parentPt[0]:
        midText = 'F'
        plt.text((parentPt[0] + centerPt[0]) / 2.0, (parentPt[1] + centerPt[1]) / 2.0,
                  midText, va='center', ha='center', rotation=-30, color='red',
                  bbox=dict(facecolor='white', ec='white'), fontsize=8)
    else:
        pass

def show_Plot():
    plt.show()
    pass


if __name__ == '__main__':
    init_Plot()
    draw_Node('root', (0.5, 0.9), (0.5, 0.9), nodestyle_internal)
    draw_Node('X3=2', (0.4, 0.8), (0.5, 0.9), nodestyle_leaf)
    draw_Node('X1=3.1', (0.6, -1.0), (0.5, 0.9), nodestyle_internal)
    show_Plot()
    pass