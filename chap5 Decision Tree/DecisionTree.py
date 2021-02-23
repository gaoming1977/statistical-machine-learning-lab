"""
  decision tree
  CART for classification and regression tree of decision tree
  using IRIS as training data
"""
import numpy as np
import PlotUtil


class CART:

    class TreeNode:
        tree_type = True

        def __init__(self, parent=None, condition=None, value=None):
            self.__left = None
            self.__right = None
            self.__parent = parent

            """
            # condition is the decision condition.
            condition format is 
            {'xi': value} 
            left child is 'xi == value',
            right child is 'xi != value'
            """
            self.__condition = condition
            self.__c_value = value

        def __str__(self):
            assert len(self.__condition) == 1
            key = list(self.__condition.keys())[0]
            if self.isLeaf():
                strText = f"Y={self.__c_value}"
            else:
                if self.tree_type:
                    strText = f"A{key}={self.__condition[key]}"
                else:
                    strText = f"A{key}<={self.__condition[key]}"
            return strText

        def isLeft(self):
            return self.__parent.__left == self

        def isRight(self):
            return self.__parent.__right == self

        def __recursive_calc_TreeHeight(self):
            if self.isLeaf():
                return 1
            l_height = self.get_Left().__recursive_calc_TreeHeight()
            r_height = self.get_Right().__recursive_calc_TreeHeight()
            return max(l_height, r_height) + 1

        def __recursive_calc_TreeWidth(self):
            if self.isLeaf():
                return 1
            l_width = self.get_Left().__recursive_calc_TreeWidth()
            r_width = self.get_Right().__recursive_calc_TreeWidth()
            return l_width + r_width

        def calc_Treesize(self):
            width = self.__recursive_calc_TreeWidth()
            height = self.__recursive_calc_TreeHeight()
            return width, height

        def isLeaf(self):
            return self.__c_value is not None

        def add_Left(self, node):
            self.__left = node

        def add_Right(self, node):
            self.__right = node

        def set_Condition(self, condition):
            self.__condition = condition

        def set_Parent(self, parent):
            self.__parent = parent

        def get_Left(self):
            return self.__left

        def get_Right(self):
            return self.__right

        def get_Parent(self):
            return self.__parent

        def get_Condition(self):
            return self.__condition

        def get_Value(self):
            return self.__c_value

        def set_Value(self, value):
            self.__c_value = value

    def __init__(self, c_r=True, epsilon=1e-4):
        self.__root = None
        self.__x_dim = 0
        self.__cr = c_r
        self.__epsilon = epsilon
        if self.__cr:
            print('The (C)ART model is created')
        else:
            print('The CA(R)T model is created')
        pass

    def train(self, x_train, y_train):
        self.__root = None
        self.__x_dim = x_train.shape[1]
        if self.__cr:
            self.TreeNode.tree_type = True
            self.__build_C_Tree(self.__root, x_train, y_train)
        else:
            self.TreeNode.tree_type = False
            self.__build_R_Tree(self.__root, x_train, y_train)
        pass

    def display_Tree(self):
        initial_node = self.__root
        _w, _h = initial_node.calc_Treesize()
        PlotUtil.pos_xoffset = 1.0/float(_w)
        PlotUtil.pos_yoffset = 1.05/float(_h)
        initial_pos = (0.5, 0.98)
        self.__recursive_display_Tree(initial_node, initial_pos, parent_pos=initial_pos)
        pass

    def __recursive_display_Tree(self, node, pos, parent_pos):
        node_text = str(node)
        node_type = PlotUtil.nodestyle_leaf if node.isLeaf() else PlotUtil.nodestyle_internal
        _w_p, _ = node.calc_Treesize()

        # draw left child
        if node.get_Left() is not None:
            node_left = node.get_Left()
            _w_l, _ = node_left.calc_Treesize()
            x_l = pos[0] - _w_p*PlotUtil.pos_xoffset/2.0
            pos_child = ((x_l + _w_l*PlotUtil.pos_xoffset/2.0), pos[1]-PlotUtil.pos_yoffset)
            self.__recursive_display_Tree(node_left, pos_child, parent_pos=pos)

        if node.get_Right() is not None:
            node_right = node.get_Right()
            _w_r, _ = node_right.calc_Treesize()
            x_r = pos[0] + _w_p*PlotUtil.pos_xoffset/2.0
            pos_child = ((x_r - _w_r*PlotUtil.pos_xoffset/2.0), pos[1]-PlotUtil.pos_yoffset)
            self.__recursive_display_Tree(node_right, pos_child, parent_pos=pos)

        # draw myself
        PlotUtil.draw_Node(node_text, pos, parent_pos, node_type)

    def prune(self, val_x, val_y):
        """
        recursive prune the CART tree, using validation dataset
        :param val_x:
        :param val_y:
        :return:
        """
        pass

    def predict(self, x):
        assert x.shape[1] == self.__x_dim
        row_num = x.shape[0]
        y_p = np.zeros((row_num, 1), dtype=float)
        for i in range(row_num):
            x_i = x[i, :]
            if self.__cr:
                y_p[i] = self.__predict_c(x_i, self.__root)
            else:
                y_p[i] = self.__predict_r(x_i, self.__root)

        return y_p

    def __predict_c(self, x, node):
        if node.isLeaf():
            y_c = node.get_Value()
        else:
            con = node.get_Condition()
            a_i = list(con.keys())[0]
            a_i_v = con[a_i]
            if x[int(a_i)] == a_i_v:
                node = node.get_Left()
            else:
                node = node.get_Right()
            y_c = self.__predict_c(x, node)
        return y_c

    def __predict_r(self, x, node):
        if node.isLeaf():
            y_r = node.get_Value()
        else:
            con = node.get_Condition()
            a_i = list(con.keys())[0]
            a_i_v = con[a_i]
            if x[int(a_i)] <= a_i_v:
                node = node.get_Left()
            else:
                node = node.get_Right()
            y_r = self.__predict_r(x, node)
        return y_r

    def __calc_Gini(self, y):
        if y.shape[0] == 0:
            return 0
        _, c_counts = np.unique(y, return_counts=True)
        counts = y.shape[0]
        sum = np.sum((c_i / counts) ** 2 for c_i in c_counts)
        y_gini = 1.0 - sum

        return y_gini

    def __build_C_Tree(self, parent, x, y):
        """
        build Classify Tree, using Gini as loss evaluation
        :param x:
        :param y:
        :return:
        """
        # step1 prepare dataset
        if x.shape[0] != y.shape[0]:
            print(f'error: x dimension({x.shape[0]} is not equal to y({y.shape[0]})')
            return None
        y = np.reshape(y, (-1, 1))
        D_A = np.concatenate((x, y), axis=1)
        print(f"D size is {D_A.shape[0]} ")

        y_u = np.unique(y)
        if y_u.shape[0] < 2:  # y can not be divide into more than 2 group
            node = CART.TreeNode(parent=parent,
                                 condition=parent.get_Condition(),
                                 value=int(y_u[0]))
            return node

        if D_A.shape[0] == 2:
            if(x[0, :].all() == x[1, :].all()) and \
                    (y_u.shape[0] == 2):
                print('WARN: error in train data, y value is ambiguous')
                _D_A = np.delete(D_A, [1], axis=0)
                return self.__build_C_Tree(parent, _D_A[:, :-1], _D_A[:, -1])

        # step1 select x feature
        # calc x Gini
        A_num = x.shape[1]
        D_c = D_A.shape[0]
        min_gini = 1
        min_A_val = -1
        min_A_index = -1
        D1_A = None
        D2_A = None

        for i in range(A_num):  # foreach x column for different feature
            xi = x[:, i]
            xi_u = np.unique(xi)
            for xii in xi_u:
                D1_xii = D_A[np.where(xi == xii)]
                y1_xii = D1_xii[:, -1]
                gini_D1_xii = self.__calc_Gini(y1_xii)

                D2_xii = D_A[np.where(xi != xii)]
                y2_xii = D2_xii[:, -1]
                gini_D2_xii = self.__calc_Gini(y2_xii)

                D1_c = D1_xii.shape[0]
                D2_c = D2_xii.shape[0]

                gini_D_xii = D1_c /D_c * gini_D1_xii + D2_c /D_c * gini_D2_xii

                if min_gini > gini_D_xii:
                    min_gini = gini_D_xii
                    min_A_val = xii
                    min_A_index = i
                    D1_A = D1_xii
                    D2_A = D2_xii

        # select min gini
        condition = {f"{min_A_index}": int(min_A_val)}
        node = CART.TreeNode(parent=parent,
                             condition=condition,
                             value=None)
        if parent is None:
            self.__root = node

        l_child = self.__build_C_Tree(node, D1_A[:, :-1], D1_A[:, -1])
        node.add_Left(l_child)
        r_child = self.__build_C_Tree(node, D2_A[:, :-1], D2_A[:, -1])
        node.add_Right(r_child)

        return node

    def __calc_MSE(self, y):
        if y.shape[0] == 0:
            return 0
        y_mse = np.var(y)
        return y_mse

    def __build_R_Tree(self, parent, x, y):
        """
        build Regression Tree, using MSE as loss evaluation
        :param x:
        :param y:
        :return:
        """
        # step1 prepare dataset
        if x.shape[0] != y.shape[0]:
            print(f'error: x dimension({x.shape[0]} is not equal to y({y.shape[0]})')
            return None

        mse_y = self.__calc_MSE(y)
        if mse_y <= self.__epsilon:
            node = CART.TreeNode(parent=parent,
                       condition=parent.get_Condition(),
                       value=np.mean(y).round(2))
            return node

        # non leaf node
        y = np.reshape(y, (-1, 1))
        D_A = np.concatenate((x, y), axis=1)
        print(f"D size is {D_A.shape[0]} ")

        A_num = x.shape[1]
        min_mse = mse_y
        min_A_val = None
        min_A_index = 0
        D1_A = None
        D2_A = None

        for i in range(A_num):
            xi = x[:, i]
            xi_u = np.unique(xi)
            for xii in xi_u:
                D1_xii = D_A[np.where(xi <= xii)]
                y1_xii = D1_xii[:, -1]
                D1_mse = self.__calc_MSE(y1_xii)

                D2_xii = D_A[np.where(xi > xii)]
                y2_xii = D2_xii[:, -1]
                D2_mse = self.__calc_MSE(y2_xii)

                if (D1_xii.shape[0] < 1) or (D2_xii.shape[0] < 1):  # invalid split, skip to next A
                    continue

                mse_D_xii = D1_mse + D2_mse

                if min_mse >= mse_D_xii:
                    min_mse = mse_D_xii
                    min_A_val = xii
                    min_A_index = i
                    D1_A = D1_xii
                    D2_A = D2_xii

        if D1_A is None or D2_A is None:  # can not split D, create leaf node, terminate recursion.
            node = CART.TreeNode(parent=parent,
                       condition=parent.get_Condition(),
                       value=np.mean(y).round(2))
            return node

        # select min mse
        condition = {f"{min_A_index}": float(min_A_val)}
        node = CART.TreeNode(parent=parent,
                             condition=condition,
                             value=None)
        if parent is None:
            self.__root = node

        l_child = self.__build_R_Tree(node, D1_A[:, :-1], D1_A[:, -1])
        node.add_Left(l_child)

        r_child = self.__build_R_Tree(node, D2_A[:, :-1], D2_A[:, -1])
        node.add_Right(r_child)

        return node

