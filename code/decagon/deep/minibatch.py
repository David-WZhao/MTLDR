from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold
from ..utility import preprocessing
#from test import Loss_w


np.random.seed(123)


class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    val_edges_false: dict

    def __init__(self, adj_mats, feat, edge_types, batch_size=128, data_set='DataSet', val_test_size=0.01,seed = 0):
        self.adj_mats = adj_mats   # 各个关联
        self.feat = feat   # 特征
        self.data_set = data_set
        self.edge_types = edge_types   # 边的类型

        self.batch_size = batch_size  # 128
        self.val_test_size = val_test_size  # 0.1
        self.num_edge_types = sum(self.edge_types.values())  # 12
        #print('self.num_edge_types:',self.num_edge_types)
        self.iter = 0

        self.freebatch_edge_types = list(range(self.num_edge_types))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #print('freebatch_edge_types:',self.freebatch_edge_types)
        self.batch_num = [0]*self.num_edge_types
        #print(self.batch_num)  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.current_edge_type_idx = 0

        self.edge_type2idx = {}
        self.idx2edge_type = {}

        r = 0
        for i, j in self.edge_types:
            #print('i,j:',self.edge_types[i,j])  # 字典中的值 2 1 1 ...
            for k in range(self.edge_types[i,j]):
                self.edge_type2idx[i, j, k] = r   # 0,1,2,3,4,5,6,7,8,9,10,11
                #print('edge_type2idx:',self.edge_type2idx[i,j,k])
                self.idx2edge_type[r] = i, j, k        # (0,0,0) (0,0,1),(0,1,0).....
                #print(self.idx2edge_type[r])
                r += 1

        self.train_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        # self.train_edges: {(0, 0): [None, None], (0, 1): [None], (0, 2): [None], (1, 0): [None], (1, 1): [None, None]......
        self.val_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        #  {(0, 0): [None, None], (0, 1): [None], (0, 2): [None], (1, 0): [None], (1, 1): [None, None], (1, 2): [None], (1, 3): [None], (2, 0): [None], (2, 1): [None], (3, 1): [None]}
        self.test_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        # 同上
        self.test_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        # 同上
        self.val_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}

        # Function to build test and val sets with val_test_size positive links
        self.adj_train = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                print("Minibatch edge type:", "(%d, %d, %d)" % (i, j, k))
                #(0,0,0) (0,0,1) (0,1,0) (0,2,0)....

                self.mask_test_edges((i, j), k,seed)   # 划分数据集
                # print('seed:',seed)

                print("Train edges=", "%04d" % len(self.train_edges[i,j][k]))
                print("Val edges=", "%04d" % len(self.val_edges[i,j][k]))
                print("Test edges=", "%04d" % len(self.test_edges[i,j][k]))

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        #print('adj:',adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])  # 同质信息网络主对角线为1
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo() # 对称归一化
        else:
            rowsum = np.array(adj.sum(1)) # 每行为1的个数
            colsum = np.array(adj.sum(0)) # 每列为1的个数
            #print('adj.sum(1):',adj.sum(1))
            #print('adj.sum(0):', adj.sum(0))
            #print('rowsum:',rowsum.shape) #  (1512, 1)
            #print('colsum:',colsum.shape) #  (1, 708)
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            #print('power:',rowdegree_mat_inv)
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()   #  对称归一化
            #print('1:',rowdegree_mat_inv.shape)  # 1512*1512
            #print('2:',adj) # 1512*708
            #print('3:',coldegree_mat_inv.shape)  # 708*708
            #print('1:',rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv))
            #print('2:',adj_normalized)
            #print('1:',adj_normalized)
            #print('2:',preprocessing.sparse_to_tuple(adj_normalized))
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)  # axis=1表示每行元素操作
        return np.any(rows_close)

    def mask_test_edges(self, edge_type, type_idx,seed):  # 类型的id
        if edge_type==(1,2) or edge_type==(2,1):  # 边的类型是药物-蛋白质
            data_set = self.data_set  #  luo
            ratio='kflod'

            train_edges = np.loadtxt('../MTHGCN-demo2/'+data_set+'/'+ratio+'/train_index_(1,2)'+str(seed)+'.txt',dtype=int)
            if edge_type == (2, 1):
                for ii in range(len(train_edges)):
                    temp = train_edges[ii][0]
                    train_edges[ii][0] = train_edges[ii][1]
                    train_edges[ii][1] = temp

            test_edges_false = np.loadtxt('../MTHGCN-demo2/'+data_set+'/'+ratio+'/index_test_false(1,2)'+str(seed)+'.txt',dtype=int)
            if edge_type == (2,1):
                for ii in range(len(test_edges_false)):
                    temp = test_edges_false[ii][0]
                    test_edges_false[ii][0] = test_edges_false[ii][1]
                    test_edges_false[ii][1] = temp
            val_edges_false = test_edges_false

            test_edges = np.loadtxt('../MTHGCN-demo2/'+data_set+'/'+ratio+'/test_index_(1,2)'+str(seed)+'.txt',dtype=int)
            if edge_type == (2, 1):
                for ii in range(len(test_edges)):
                    temp = test_edges[ii][0]
                    test_edges[ii][0] = test_edges[ii][1]
                    test_edges[ii][1] = temp
            val_edges = test_edges

            # Re-build adj matrices
            data = np.ones(train_edges.shape[0])
            #print('data:',data) # 1730个1
            #print('train.shape[0]:',train_edges.shape[0]) # 1730
            #print(data)
            adj_train = sp.csr_matrix(
                (data, (train_edges[:, 0], train_edges[:, 1])),
                shape=self.adj_mats[edge_type][type_idx].shape)
            #print('adj_train',adj_train)
            #print('train_edges[:, 0]:',train_edges[:, 0]) # [   2    2    6 ... 1506 1507 1508] train_index(0,1)0  第一列
            #print('train_edges[:, 1]:', train_edges[:, 1]) # [546 622  23 ... 299 299 299]  train_index(0,1)0  第二列
            #print('self.adj_mats[edge_type][type_idx].shape',self.adj_mats[edge_type][type_idx].shape) # (1512, 708) 是(0,1)的维度
            #print('adj_train:',adj_train)  # (2, 546)	1.0
            #print("adj:",adj_train.shape[0]) #(0,1)  1512*708
            #print("adj:", adj_train.shape[1]) # (1,0)  708*1512
            self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)  # 经过归一化  ： DAD

            self.train_edges[edge_type][type_idx] = train_edges
            self.val_edges[edge_type][type_idx] = val_edges
            self.val_edges_false[edge_type][type_idx] = np.array(val_edges_false)
            self.test_edges[edge_type][type_idx] = test_edges
            self.test_edges_false[edge_type][type_idx] = np.array(test_edges_false)
        else:
            edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[edge_type][type_idx])
            train_edges = edges_all
            num_test = 1
            num_val = 1
            all_edge_idx = list(range(edges_all.shape[0]))
            #print('edge_idx:',edges_all.shape[0]) # 有多少条已知的边
            np.random.shuffle(all_edge_idx)  # 打乱

            val_edge_idx = all_edge_idx[:num_val]
            #print('all_edge_idx:', all_edge_idx)
            #print('all_edge_idx[:num_val]:',all_edge_idx[:num_val])
            val_edges = edges_all[val_edge_idx]

            #
            test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
            test_edges = edges_all[test_edge_idx]

            test_edges_false = []

            while len(test_edges_false) < len(test_edges):
                if len(test_edges_false) % 1000 == 0:
                    print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))
                # 使用随机函数随机选取一个样本的行列，
                idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
                idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
                print('idx_i:',idx_i)
                print('idx_j:',idx_j)
                # 如果该行列已经存在正样本
                if self._ismember([idx_i, idx_j], edges_all):
                    continue
                if test_edges_false:
                    if self._ismember([idx_i, idx_j], test_edges_false):
                        continue
                test_edges_false.append([idx_i, idx_j])
                #print('test_edges_false:',len(test_edges_false))

            val_edges_false = []
            while len(val_edges_false) < len(val_edges):
                if len(val_edges_false) % 1000 == 0:
                    print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
                idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
                idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
                if self._ismember([idx_i, idx_j], edges_all):
                    continue
                if val_edges_false:
                    if self._ismember([idx_i, idx_j], val_edges_false):
                        continue
                val_edges_false.append([idx_i, idx_j])


            # Re-build adj matrices
            data = np.ones(train_edges.shape[0])
            adj_train = sp.csr_matrix(
                (data, (train_edges[:, 0], train_edges[:, 1])),
                shape=self.adj_mats[edge_type][type_idx].shape)
            #print('adj_train:',adj_train) # (0, 259)	1.0
            self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)  # 拉普拉斯矩阵

            self.train_edges[edge_type][type_idx] = train_edges
            self.val_edges[edge_type][type_idx] = val_edges
            self.val_edges_false[edge_type][type_idx] = np.array(val_edges_false)
            self.test_edges[edge_type][type_idx] = test_edges
            self.test_edges_false[edge_type][type_idx] = np.array(test_edges_false)


    def end(self):
        '''
        a = self.edge_type2idx[1, 2, 0] not in self.freebatch_edge_types  # a表示(1,2)类型的关联不在  药物-疾病
        b = self.edge_type2idx[2, 1, 0] not in self.freebatch_edge_types  # b表示(2,1)类型的关联不在

        if a & b:
            finished = 1  # 如果a，b都没有了，表示计算结束
        else:
            finished = 0
        return finished
        '''
        a = self.edge_type2idx[1, 2, 0] not in self.freebatch_edge_types  # a表示(1,2)类型的关联不在  药物-疾病
        b = self.edge_type2idx[2, 1, 0] not in self.freebatch_edge_types  # b表示(2,1)类型的关联不在
        c = self.edge_type2idx[0, 1, 0] not in self.freebatch_edge_types  # c表示(0,1)类型的关联不在  药物-蛋白质
        d = self.edge_type2idx[1, 0, 0] not in self.freebatch_edge_types  # d表示(1,0)类型的关联不在
        e = self.edge_type2idx[1, 3, 0] not in self.freebatch_edge_types  # e表示(1,3)类型的关联不在  药物-mirna
        f = self.edge_type2idx[3, 1, 0] not in self.freebatch_edge_types  # f表示(3,1)类型的关联不在
        g = self.edge_type2idx[2, 3, 0] not in self.freebatch_edge_types  # g表示(2,3)类型的关联不在  疾病-mirna
        h = self.edge_type2idx[3, 2, 0] not in self.freebatch_edge_types  # h表示(3,2)类型的关联不在
        # i = self.edge_type2idx[0, 2, 0] not in self.freebatch_edge_types  # h表示(0,2)类型的关联不在
        # j = self.edge_type2idx[2, 0, 0] not in self.freebatch_edge_types  # h表示(2,0)类型的关联不在
        if a & b & c & d & e & f & g & h:  # & i & j:
            finished = 1  # 如果a，b都没有了，表示计算结束
        else:
            finished = 0
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):   # 更新占位符，更新内容为经过归一化的邻接矩阵以及特征表示，丢失率   进行模型中的表示学习
        # construct feed dictionary
        feed_dict.update({
            placeholders['adj_mats_%d,%d,%d' % (i,j,k)]: self.adj_train[i,j][k]
            for i, j in self.edge_types for k in range(self.edge_types[i,j])})
        feed_dict.update({placeholders['feat_%d' % i]: self.feat[i] for i, _ in self.edge_types})
        feed_dict.update({placeholders['dropout']: dropout})
        return feed_dict



    def batch_feed_dict(self, batch_edges, batch_edge_type,loss_w, placeholders):  # 取一个batch;取batch的类型;占位符
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        #print('batch_edges:',batch_edges)
        feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
        #print('batch_edge_type_idx', batch_edge_type)   # 2和4的其中一个的随机取值
        feed_dict.update({placeholders['batch_row_edge_type']: self.idx2edge_type[batch_edge_type][0]})
        #print('batch_row_edge_type:',self.idx2edge_type[batch_edge_type][0])  # 第一个位置:[0]上的值，表示该节点的类型
        feed_dict.update({placeholders['batch_col_edge_type']: self.idx2edge_type[batch_edge_type][1]})
        #print('batch_col_edge_type:', self.idx2edge_type[batch_edge_type][1]) # 第二个位置:[1]上的值，也表示该节点的类型

        # loss weight
        # loss weight
        feed_dict.update({placeholders['loss_w']: loss_w})

        return feed_dict

    # Needs to be changed
    def next_minibatch_feed_dict(self, placeholders):
        # next_minibatch_feed_dict()返回的是下一个edge minibatch的placeholders信息
        """Select a random edge type and a batch of edges of the same type"""  # 随机选择一个边的类型，并选择一个batch是该类型的边  注意"随机"两个字
                # random side effect relation
                #print('len(self.freebatch_edge_types):',len(self.freebatch_edge_types)) # 第一次遍历 : 2
                #print('self.freebatch_edge_types:',self.freebatch_edge_types)  #  第一次遍历  :   [2,4]
        if len(self.freebatch_edge_types) > 0:
            self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)  # 从freebatch_edge_types中随机选取
                    #print('self.current_edge_type_idx:',self.current_edge_type_idx)

        i, j, k = self.idx2edge_type[self.current_edge_type_idx]   # idx2edge_type[2] 就是 (0,1,0)
            #print('self.idx2edge_type[self.current_edge_type_idx]:',self.idx2edge_type[self.current_edge_type_idx])
        if self.batch_num[self.current_edge_type_idx] * self.batch_size <= len(self.train_edges[i,j][k]) - self.batch_size + 1:  # 判断是否还有数据
                #print('self.batch_num[self.current_edge_type_idx]:',self.batch_num[self.current_edge_type_idx])
                #print('len(self.train_edges[i,j][k]):',len(self.train_edges[i,j][k]))
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size  # 开始
            self.batch_num[self.current_edge_type_idx] += 1  # 加1
            batch_edges = self.train_edges[i, j][k][start: start + self.batch_size]  # 取一个batch
        else:
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size  # 开始
            self.batch_num[self.current_edge_type_idx] += 1  # 加1
            #batch_edges = self.train_edges[i, j][k][start: len(self.train_edges[i,j][k])]  # 取一个batch
            batch_edges = self.train_edges[i, j][k][start-(self.batch_size-(len(self.train_edges[i,j][k])-start+1))-1: len(self.train_edges[i, j][k])]  # 取最后一个batch
            self.freebatch_edge_types.remove(self.current_edge_type_idx)  # 当某类型边计算完后就删除该类型的边的id

        #self.iter += 1
        #print('iter:',self.iter)
        #print('self.current_edge_type_idx:',self.current_edge_type_idx)

        if self.current_edge_type_idx == 2:  # 药物-蛋白
            loss_w = Loss_w[1]
        if self.current_edge_type_idx == 4:
            loss_w = Loss_w[1]
        if self.current_edge_type_idx == 7:
            loss_w = Loss_w[0]
        if self.current_edge_type_idx == 10:
            loss_w = Loss_w[0]
        if self.current_edge_type_idx == 8:  # 药物-mirna
            loss_w = Loss_w[1]
        if self.current_edge_type_idx == 14:
            loss_w = Loss_w[1]
        if self.current_edge_type_idx == 13:
            loss_w = Loss_w[2]
        if self.current_edge_type_idx == 15:  # 疾病-mirna
            loss_w = Loss_w[2]
        return self.batch_feed_dict(batch_edges, self.current_edge_type_idx,loss_w,placeholders)


    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        for edge_type in self.edge_types:
            #print('edge_type:',edge_type)  #(0, 0) 也就是取关系类型字典中的值
            for k in range(self.edge_types[edge_type]):
                #print('edge_types:',self.edge_types) # edge_types: {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (1, 3): 1, (2, 0): 1, (2, 1): 1, (3, 1): 1}
                #print('edge_types[edge_type]:',self.edge_types[edge_type])  # 2,2,1...也就是根据(0,0)在字典中所对应的值
                self.train_edges[edge_type][k] = np.random.permutation(self.train_edges[edge_type][k]) # permutation : 打乱

                self.batch_num[self.edge_type2idx[edge_type[0], edge_type[1], k]] = 0
        self.current_edge_type_idx = 0

        # Notice: the two dataset is different.
        # edge_types {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
        self.freebatch_edge_types = list(range(self.num_edge_types))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #print('self.freebatch_edge_types:',self.freebatch_edge_types)
        if self.data_set=='DataSet':
            self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 1])
            #self.freebatch_edge_types.remove(self.edge_type2idx[0, 1, 0])  #2
            self.freebatch_edge_types.remove(self.edge_type2idx[0, 2, 0])
            #self.freebatch_edge_types.remove(self.edge_type2idx[1, 0, 0])  #4
            self.freebatch_edge_types.remove(self.edge_type2idx[1, 1, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[1, 1, 1])
            #self.freebatch_edge_types.remove(self.edge_type2idx[1, 2, 0]) # 7
            #self.freebatch_edge_types.remove(self.edge_type2idx[1, 3, 0]) #8

            self.freebatch_edge_types.remove(self.edge_type2idx[2, 0, 0])
            #self.freebatch_edge_types.remove(self.edge_type2idx[2, 1, 0]) #10
            self.freebatch_edge_types.remove(self.edge_type2idx[2, 2, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[2, 2, 1])
            #self.freebatch_edge_types.remove(self.edge_type2idx[2, 3, 0]) # 13

            #self.freebatch_edge_types.remove(self.edge_type2idx[3, 1, 0]) # 14
            #self.freebatch_edge_types.remove(self.edge_type2idx[3, 2, 0])#15
            self.freebatch_edge_types.remove(self.edge_type2idx[3, 3, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[3, 3, 1])
            #print('self.edge_type2idx:', self.edge_type2idx[3, 1, 0])
            #print('self.freebatch_edge_types:', self.freebatch_edge_types)  # freebatch_edge_types 中只保留(0,1) (1,0)
            # 这里删除的边是不具有'val  test'属性的
            self.iter = 0
        else:
            self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 1])
            # self.freebatch_edge_types.remove(self.edge_type2idx[0, 1, 0])

            # self.freebatch_edge_types.remove(self.edge_type2idx[1, 0, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[1, 1, 0])
            self.freebatch_edge_types.remove(self.edge_type2idx[1, 1, 1])

            self.iter = 0




    def num_training_batches(self, edge_type, type_idx):
        return len(self.train_edges[edge_type][type_idx]) // self.batch_size + 1

    def val_feed_dict(self, edge_type, type_idx, placeholders, size=None):
        edge_list = self.val_edges[edge_type][type_idx]
        if size is None:
            return self.batch_feed_dict(edge_list, edge_type, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, edge_type, placeholders)




        '''
            # Needs to be changed
    def next_minibatch_feed_dict2(self, placeholders):
        # next_minibatch_feed_dict()返回的是下一个edge minibatch的placeholders信息
        """Select a random edge type and a batch of edges of the same type"""  # 随机选择一个边的类型，并选择一个batch是该类型的边  注意"随机"两个字
                # random side effect relation
                #print('len(self.freebatch_edge_types):',len(self.freebatch_edge_types)) # 第一次遍历 : 2
                #print('self.freebatch_edge_types:',self.freebatch_edge_types)  #  第一次遍历  :   [2,4]
        while True:
            if self.iter % 4 == -1:   # 取模是为了保证每每个batchsize都为128
                # gene-gene relation
                #self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
                pass
            elif self.iter % 4 == 1:
                # gene-drug relation
                self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
            elif self.iter % 4 == 2:
                # drug-gene relation
                self.current_edge_type_idx = self.edge_type2idx[1, 0, 0]
            else:
                # random side effect relation
                # print('len(self.freebatch_edge_types):',len(self.freebatch_edge_types)) # 第一次遍历 : 2
                # print('self.freebatch_edge_types:',self.freebatch_edge_types)  #  第一次遍历  :   [2,4]
                if len(self.freebatch_edge_types) > 0:
                    self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)  # 从freebatch_edge_types中随机选取
                    # print('self.current_edge_type_idx:',self.current_edge_type_idx)
                else:
                    self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
                    self.iter = 1

            i, j, k = self.idx2edge_type[self.current_edge_type_idx]  # idx2edge_type[2] 就是 (0,1,0)
            # print('self.idx2edge_type[self.current_edge_type_idx]:',self.idx2edge_type[self.current_edge_type_idx])
            if self.batch_num[self.current_edge_type_idx] * self.batch_size <= len(self.train_edges[i, j][k]) - self.batch_size + 1:
                # print('self.batch_num[self.current_edge_type_idx]:',self.batch_num[self.current_edge_type_idx])
                # print('len(self.train_edges[i,j][k]):',len(self.train_edges[i,j][k]))
                break
            else:
                if self.iter % 4 in [1, 2]:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:  # 表示其中一个已经训练完毕
                    self.freebatch_edge_types.remove(self.current_edge_type_idx)


        self.iter += 1
        print('iter:',self.iter)
        print('self.current_edge_type_idx:',self.current_edge_type_idx)
        start = self.batch_num[self.current_edge_type_idx] * self.batch_size  # 开始
        self.batch_num[self.current_edge_type_idx] += 1  # 加1
        batch_edges = self.train_edges[i, j][k][start: start + self.batch_size]  # 取一个batch
        print('len:',len(batch_edges))

        return self.batch_feed_dict(batch_edges, self.current_edge_type_idx, placeholders)

        '''


        '''
                while True:
            if self.iter % 4 == -1:
                # gene-gene relation
                self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
            elif self.iter % 4 == 1:
                # gene-drug relation
                self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
            elif self.iter % 4 == 2:
                # drug-gene relation
                self.current_edge_type_idx = self.edge_type2idx[1, 0, 0]
            else:
                # random side effect relation
                #print('len(self.freebatch_edge_types):',len(self.freebatch_edge_types)) # 第一次遍历 : 2
                #print('self.freebatch_edge_types:',self.freebatch_edge_types)  #  第一次遍历  :   [2,4]
                if len(self.freebatch_edge_types) > 0:
                    self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)  # 从freebatch_edge_types中随机选取
                    #print('self.current_edge_type_idx:',self.current_edge_type_idx)
                else:
                    self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
                    self.iter = 1

            i, j, k = self.idx2edge_type[self.current_edge_type_idx]   # idx2edge_type[2] 就是 (0,1,0)
            #print('self.idx2edge_type[self.current_edge_type_idx]:',self.idx2edge_type[self.current_edge_type_idx])
            if self.batch_num[self.current_edge_type_idx] * self.batch_size <= len(self.train_edges[i,j][k]) - self.batch_size + 1:
                #print('self.batch_num[self.current_edge_type_idx]:',self.batch_num[self.current_edge_type_idx])
                #print('len(self.train_edges[i,j][k]):',len(self.train_edges[i,j][k]))
                break
            else:
                if self.iter % 4 in [1,2]:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:  # 表示其中一个已经训练完毕
                    self.freebatch_edge_types.remove(self.current_edge_type_idx)

        '''

        '''
                        # random side effect relation
                #print('len(self.freebatch_edge_types):',len(self.freebatch_edge_types)) # 第一次遍历 : 2
                #print('self.freebatch_edge_types:',self.freebatch_edge_types)  #  第一次遍历  :   [2,4]
        if len(self.freebatch_edge_types) > 0:
            self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)  # 从freebatch_edge_types中随机选取
                    #print('self.current_edge_type_idx:',self.current_edge_type_idx)

        i, j, k = self.idx2edge_type[self.current_edge_type_idx]   # idx2edge_type[2] 就是 (0,1,0)
            #print('self.idx2edge_type[self.current_edge_type_idx]:',self.idx2edge_type[self.current_edge_type_idx])
        if self.batch_num[self.current_edge_type_idx] * self.batch_size <= len(self.train_edges[i,j][k]) - self.batch_size + 1:  # 判断是否还有数据
                #print('self.batch_num[self.current_edge_type_idx]:',self.batch_num[self.current_edge_type_idx])
                #print('len(self.train_edges[i,j][k]):',len(self.train_edges[i,j][k]))
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size  # 开始
            self.batch_num[self.current_edge_type_idx] += 1  # 加1
            batch_edges = self.train_edges[i, j][k][start: start + self.batch_size]  # 取一个batch
        else:
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size  # 开始
            self.batch_num[self.current_edge_type_idx] += 1  # 加1
            batch_edges = self.train_edges[i, j][k][start: len(self.train_edges[i,j][k])-start]  # 取一个batch
            

            self.freebatch_edge_types.remove(self.current_edge_type_idx)


        #self.iter += 1
        #print('iter:',self.iter)
        #print('self.current_edge_type_idx:',self.current_edge_type_idx)


        return self.batch_feed_dict(batch_edges, self.current_edge_type_idx, placeholders)
        '''