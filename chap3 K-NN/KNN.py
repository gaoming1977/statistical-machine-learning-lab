# KNN class file
# using KD tree to search K neighbour nodes

import numpy as np


class KNN_Model():
    class KD_Tree_Node():

        root = None

        def __init__(self, xy, split):
            self.parent = None
            self.l_child = None
            self.r_child = None
            self.xy = xy
            self.split = split

        def get_x(self):
            x = self.xy[0:-1]
            return x

        def get_y(self):
            y = self.xy[-1]
            return y

        def get_sibling_node(self):
            if self.parent is None:
                return None
            my_parent = self.parent
            if my_parent.l_child is self:
                my_sibling = my_parent.r_child
            elif my_parent.r_child is self:
                my_sibling = my_parent.l_child
            else:
                my_sibling = None
            return my_sibling

        def is_Leaf(self):
            return (self.l_child is None) and (self.r_child is None)

        def is_Root(self):
            return (self.parent is None) and (self is self.root)

    def _push_nearest_node(self, node, dist, nodes_stack):
        """

        :param node:
        :param dist:
        :param nodes_stack: shape(K, 2) <node_obj, dist>
        :return:
        """
        a_row = np.array((node, dist)).reshape((-1, 2))
        nodes_stack = np.concatenate((nodes_stack, a_row), axis=0)  # by row

        # rest_nodes = rest_nodes[np.argsort(rest_nodes[:, dim_split], axis=0)]

        nodes_stack = nodes_stack[np.argsort(nodes_stack[:, -1], axis=0)]  # by dist col

        if nodes_stack.shape[0] > self._K:
            nodes_stack = nodes_stack[0: self._K, :]
        return nodes_stack

    def _search_kd_nearest_leaf(self, x, start_node):
        r_node = start_node
        while r_node.is_Leaf() is False:
            mid = r_node.get_x()
            j = r_node.split
            if x[j] < mid[j]:
                t_node = r_node.l_child
            else:
                t_node = r_node.r_child
            if t_node is None:
                break
            else:
                r_node = t_node
        return r_node

    def _KNN_predict(self, x):
        """
        search kd_tree to find the K neighbours of x, and vote the y
        :param x: shape(1, 3)
        :return: y: shape(1, 1)
        """
        assert(x.shape[0] == self.KD_Tree_Node.root.xy.shape[0] - 1)

        k_nodes = np.empty((0, 2))  # <node, dist>
        # step 1: traverse the kd tree to find the nearest leaf node
        cur_node = self.KD_Tree_Node.root
        cur_node = self._search_kd_nearest_leaf(x, cur_node)

        dist_min = np.linalg.norm(x - cur_node.get_x())
        # step 2: roll up the non-leaf nodes
        is_rootnode = cur_node.is_Root()
        cur_node_2 = cur_node

        while is_rootnode is False:
            dist_temp = np.linalg.norm(x - cur_node_2.get_x())
            k_nodes = self._push_nearest_node(cur_node_2, dist_temp, k_nodes)
            if dist_temp < dist_min:
                dist_min = dist_temp
                sibling_node = cur_node_2.get_sibling_node()
                if sibling_node is not None: #
                    cur_node_2 = self._search_kd_nearest_leaf(x, sibling_node)
            else:
                cur_node_2 = cur_node_2.parent
            is_rootnode = cur_node_2.is_Root()

        # step final, vote the predict y value
        vote_y = np.zeros(self.y_class, dtype=np.int)
        for i in range(k_nodes.shape[0]):
            node_y = k_nodes[i, 0].get_y()
            vote_y[int(node_y)] += 1
        return np.argmax(vote_y)

    def __init__(self, k=5):
        self._K = k
        self.x_max = []
        self.x_min = []

        self.y_class = 0
        pass

    def _recursive_split_KD_nodes(self, xy_nodes, dim, dim_split, parent):
        if xy_nodes.shape[0] < 1:  # row less than 1, should return
            return None

        rest_nodes = np.copy(xy_nodes)

        rest_nodes = rest_nodes[np.argsort(rest_nodes[:, dim_split], axis=0)]
        median = rest_nodes.shape[0] // 2
        median_node = rest_nodes[median]

        kd_mid_node = self.KD_Tree_Node(median_node, dim_split)
        kd_mid_node.parent = parent

        l_rest_nodes = rest_nodes[0: median]
        r_rest_nodes = rest_nodes[median + 1: -1]

        dim_split_c = (dim_split + 1) % dim
        kd_mid_node.l_child = self._recursive_split_KD_nodes(l_rest_nodes, dim, dim_split_c, kd_mid_node)
        kd_mid_node.r_child = self._recursive_split_KD_nodes(r_rest_nodes, dim, dim_split_c, kd_mid_node)

        # print(kd_mid_node.get_x())
        # print(kd_mid_node.get_y())
        return kd_mid_node
        pass

    def _build_KD_Tree(self, xy_nodes):
        dim = xy_nodes.shape[1] - 1  # exclude the y dimension
        assert(dim > 1)
        dim_split = 0

        self.KD_Tree_Node.root = self._recursive_split_KD_nodes(xy_nodes, dim, dim_split, None)

        print("KD Tree root node is :", self.KD_Tree_Node.root.xy)

    def train(self, x_train, y_train):
        '''
        fit the model
        :param x_train: shape(N, 3)
        :param y_train: shape(N, 1) label
        :return:
        '''
        assert(x_train.shape[0] == y_train.shape[0])
        xx_train = np.copy(x_train)
        yy_train = np.copy(y_train)

        self.y_class = len(np.unique(yy_train))

        #normalization x_train, every dimension normal the value between 0 and 1
        self.x_max = np.max(x_train, axis=0)
        self.x_min = np.min(x_train, axis=0)

        for j in range(x_train.shape[1]):
            xx_train[:, j] = (x_train[:, j] -
                              self.x_min[j])/(self.x_max[j] - self.x_min[j])
        xy_train = np.concatenate((xx_train, yy_train), axis=1)

        self._build_KD_Tree(xy_train)
        pass

    def __call__(self, x_input):
        #assert(x_input.shape[1])
        # normalize x_input
        xx_input = np.zeros_like(x_input)
        for j in range(xx_input.shape[1]):
            xx_input[:, j] = (x_input[:, j] -
                              self.x_min[j])/(self.x_max[j] - self.x_min[j])

        # predict with kd tree
        y_output = np.zeros((x_input.shape[0], 1))

        for i in range(xx_input.shape[0]):
            _x = xx_input[i, :]
            _y = self._KNN_predict(_x)
            y_output[i, :] = _y

        return y_output

