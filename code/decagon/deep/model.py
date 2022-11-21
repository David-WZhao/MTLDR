from collections import defaultdict

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from .layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, \
    DistMultDecoder, InnerProductDecoder, DEDICOMDecoder, BilinearDecoder

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class DecagonModel(Model):
    def __init__(self, data_set,placeholders, num_feat, nonzero_feat, edge_types, decoders, **kwargs):
        super(DecagonModel, self).__init__(**kwargs)
        self.edge_types = edge_types
        #for i,j in self.edge_types:
         #   print('num_types:',self.edge_types[i,j]) # num_types : 2 1 1 ..也就是关联字典中的值

        self.data_set = data_set  # 数据集的名称
        self.num_edge_types = sum(self.edge_types.values()) # num_edge_types : 12
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1  # 4

        self.decoders = decoders
        self.att = tf.Variable(tf.constant([0.5, 0.25,0.15,0.1]))
        #self.att = tf.Variable(tf.constant([0.5, 0.33, 0.25]))


        self.inputs = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types}   #  输入 ： 每种类型的节点特征
        print('inputs:',self.inputs)
        # self.extra_inputs = {i: placeholders['feat_extra_%d' % i] for i in range(2)}
        self.input_dim = num_feat  # 输入的维度
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']

        self.adj_mats = {et: [
            placeholders['adj_mats_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types.items()}    # 关联   et=(0,0)...

        #print('self.adj_mats:',self.adj_mats)
        self.build()

    def _build(self):
#inputs: {0: <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001FEE201E460>, 1: <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001FEE2022580>, 2: <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001FEE2022DF0>, 3: <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x000001FEE2027190>}

        # Layer1.......
        self.hidden1 = defaultdict(list)  # 创建表示的存储列表
        for i, j in self.edge_types:  # 遍历每种关联  i:0~3  j:0~3
            #print('i___:',i)
            #print('j___:',j)
            #print('inputs1:',self.inputs[j])
            #print('edge_types[i,j]:',self.edge_types[i,j]) # self.edge_types[i,j]为2 1 1 1 2 1 1 1 1 1
            self.hidden1[i].append(GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1,
                edge_type=(i,j), num_types=self.edge_types[i,j],   # edge_types =10
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=lambda x: x, dropout=self.dropout,
                logging=self.logging)(self.inputs[j]))
            #print('input[j]:',self.inputs[j])
        print('hidden1:',self.hidden1)
#hidden1: defaultdict(<class 'list'>, {0: [<tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_1/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_2/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_3/l2_normalize:0' shape=(None, 64) dtype=float32>], 1: [<tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_4/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_5/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_6/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_7/l2_normalize:0' shape=(None, 64) dtype=float32>], 2: [<tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_8/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_9/l2_normalize:0' shape=(None, 64) dtype=float32>], 3: [<tf.Tensor 'decagonmodel/graphconvolutionsparsemulti_10/l2_normalize:0' shape=(None, 64) dtype=float32>]})
        self.embeddings0 = [None] * self.num_obj_types
        print('type:',type(self.embeddings0))
        for i, hid1 in self.hidden1.items():
            self.embeddings0[i] = tf.nn.relu(tf.add_n(hid1))  # 获得各个节点的第一隐藏层的表示
            #print('hid1:',hid1)
            #print('tf.add(hid1):',tf.add_n(hid1))
            #print('self.hidden1[i]:', self.hidden1[i])
        print('self.hidden1.items():',self.hidden1.items())
        #print('self.hidden1:', self.hidden1)
        print('self.embeddings0:',self.embeddings0)
        #print('type_input:',type(self.inputs))  # <class  'dict'>
        #self.inputs[0] = list(self.inputs[0])
        #self.embeddings0[0] = self.embeddings0[0] + self.inputs[0]
#self.hidden1.items(): dict_items([(0, <tf.Tensor 'decagonmodel/Relu:0' shape=(None, 64) dtype=float32>), (1, <tf.Tensor 'decagonmodel/Relu_1:0' shape=(None, 64) dtype=float32>), (2, <tf.Tensor 'decagonmodel/Relu_2:0' shape=(None, 64) dtype=float32>), (3, <tf.Tensor 'decagonmodel/Relu_3:0' shape=(None, 64) dtype=float32>)])
#self.hidden1: defaultdict(<class 'list'>, {0: <tf.Tensor 'decagonmodel/Relu:0' shape=(None, 64) dtype=float32>, 1: <tf.Tensor 'decagonmodel/Relu_1:0' shape=(None, 64) dtype=float32>, 2: <tf.Tensor 'decagonmodel/Relu_2:0' shape=(None, 64) dtype=float32>, 3: <tf.Tensor 'decagonmodel/Relu_3:0' shape=(None, 64) dtype=float32>})
#self.embeddings0: [<tf.Tensor 'decagonmodel/Relu:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_1:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_2:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_3:0' shape=(None, 64) dtype=float32>]
        #self.hidden1 = self.hidden1 + self.inputs
        # Layer2.......
        self.embeddings_reltyp = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings0[j]))
        self.embeddings1 = [None] * self.num_obj_types

        for i, embeds in self.embeddings_reltyp.items():
            self.embeddings1[i] = tf.nn.relu(tf.add_n(embeds))

        self.embeddings1 = self.embeddings0 + self.embeddings1  # 行
        print(' self.embeddings_reltyp:', self.embeddings_reltyp)
        print(' self.embeddings_reltyp.items:', self.embeddings_reltyp.items())
        print('self.embeddings1:', self.embeddings1)
        print('type:',type(self.embeddings1))  # <class  'list'>
#self.embeddings_reltyp: defaultdict(<class 'list'>, {0: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_1/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_2/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_3/l2_normalize:0' shape=(None, 64) dtype=float32>], 1: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_4/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_5/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_6/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_7/l2_normalize:0' shape=(None, 64) dtype=float32>], 2: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_8/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_9/l2_normalize:0' shape=(None, 64) dtype=float32>], 3: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_10/l2_normalize:0' shape=(None, 64) dtype=float32>]})
#self.embeddings_reltyp.items: dict_items([(0, [<tf.Tensor 'decagonmodel/graphconvolutionmulti_1/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_2/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_3/l2_normalize:0' shape=(None, 64) dtype=float32>]), (1, [<tf.Tensor 'decagonmodel/graphconvolutionmulti_4/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_5/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_6/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_7/l2_normalize:0' shape=(None, 64) dtype=float32>]), (2, [<tf.Tensor 'decagonmodel/graphconvolutionmulti_8/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_9/l2_normalize:0' shape=(None, 64) dtype=float32>]), (3, [<tf.Tensor 'decagonmodel/graphconvolutionmulti_10/l2_normalize:0' shape=(None, 64) dtype=float32>])])
#self.embeddings1: [<tf.Tensor 'decagonmodel/Relu_4:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_5:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_6:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_7:0' shape=(None, 64) dtype=float32>]
        # Layer3.......
        self.embeddings_reltyp2 = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp2[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings1[j]))

        self.embeddings2 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp2.items():
            self.embeddings2[i] = tf.nn.relu(tf.add_n(embeds))

        #print('self.embeddings_reltyp2:',self.embeddings_reltyp2)
        #print('self.embeddings2:', self.embeddings2)
#self.embeddings_reltyp2: defaultdict(<class 'list'>, {0: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_11/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_12/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_13/l2_normalize:0' shape=(None, 64) dtype=float32>], 1: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_14/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_15/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_16/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_17/l2_normalize:0' shape=(None, 64) dtype=float32>], 2: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_18/l2_normalize:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/graphconvolutionmulti_19/l2_normalize:0' shape=(None, 64) dtype=float32>], 3: [<tf.Tensor 'decagonmodel/graphconvolutionmulti_20/l2_normalize:0' shape=(None, 64) dtype=float32>]})
#self.embeddings2 : [<tf.Tensor 'decagonmodel/Relu_8:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_9:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_10:0' shape=(None, 64) dtype=float32>, <tf.Tensor 'decagonmodel/Relu_11:0' shape=(None, 64) dtype=float32>]


        self.embeddings2 = self.embeddings2 + self.embeddings1  # 这一行可以相加


        self.embeddings_reltyp3 = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp3[i].append(GraphConvolutionMulti(
            input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
            edge_type=(i, j), num_types=self.edge_types[i, j],
            adj_mats=self.adj_mats, act=lambda x: x,
            dropout=self.dropout, logging=self.logging)(self.embeddings2[j]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp3.items():
             self.embeddings[i] = tf.add_n(embeds)

        #self.embeddings = self.embeddings2 + self.embeddings


        #for i, embeds in self.embeddings.items():
            #self.embeddings = tf.add_n(embeds)
        '''
        # Layer4.......
        self.embeddings_reltyp3 = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp3[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings2[j]))

        self.embeddings3 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp3.items():
            self.embeddings3[i] = tf.nn.relu(tf.add_n(embeds))


        # Layer5.......
        self.embeddings_reltyp4 = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp4[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings3[j]))

        self.embeddings4 = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp4.items():
            self.embeddings4[i] = tf.nn.relu(tf.add_n(embeds))
        # Layer6.......
        self.embeddings_reltyp5 = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp5[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden2, output_dim=FLAGS.hidden2,
                edge_type=(i, j), num_types=self.edge_types[i, j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.embeddings4[j]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp4.items():
            self.embeddings[i] = tf.add_n(embeds)
        '''


        # concat operation
        if self.data_set == 'DataSet':
            # examples: 5 layers
            # self.embeddings[0] = tf.concat([self.embeddings[0]],1)
            # self.embeddings[1] = tf.concat([self.embeddings[1]],1)
            # self.embeddings[2] = tf.concat([self.embeddings[2]],1)
            # self.embeddings[3] = tf.concat([self.embeddings[3]],1)

            # 注意力机制
            self.embeddings[0] = self.embeddings0[0]* self.att[0] + self.embeddings1[0]*self.att[1]+self.embeddings2[0]*self.att[2]+self.embeddings[0]*self.att[3]#+self.embeddings[0]*self.att[4]
            self.embeddings[1] = self.embeddings0[1]* self.att[0]+ self.embeddings1[1]*self.att[1]+ self.embeddings2[1]*self.att[2]+self.embeddings[1]*self.att[3]#+self.embeddings[1]*self.att[4]
            self.embeddings[2] = self.embeddings0[2]* self.att[0]+ self.embeddings1[2]*self.att[1]+ self.embeddings2[2]*self.att[2]+self.embeddings[2]*self.att[3]#+self.embeddings[2]*self.att[4]
            self.embeddings[3] = self.embeddings0[3]* self.att[0]+ self.embeddings1[3]*self.att[1]+ self.embeddings2[3]*self.att[2]+self.embeddings[3]*self.att[3]#+self.embeddings[3]*self.att[4]
            #self.embeddings =self.embeddings0 * self.att[0]+self.embeddings*self.att[1]+self.embeddings2*self.att[2]
            print('self.embeddings0[0]:',self.embeddings0[0])
            print('self.embeddings1[0]:', self.embeddings1[0])
            print('self.embeddings2[0]:', self.embeddings[0])

        else:
            # example:3 layers
            self.embeddings[0] = tf.concat([self.embeddings0[0], self.embeddings1[0], self.embeddings[0]], 1)
            self.embeddings[1] = tf.concat([self.embeddings0[1], self.embeddings1[1], self.embeddings[1]], 1)
        #print(' self.embeddings[0]:',  self.embeddings[0].shape.as_list)







        self.edge_type2decoder = {}
        for i, j in self.edge_types:
            # Important notice:
            # you need to change the num (it represents the number of GCN layers)
            num = 1
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=FLAGS.hidden2*num, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)(self.embeddings)
                #print('self.embedding:',self.embeddings[i])
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=FLAGS.hidden2, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)#(self.embeddings)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=FLAGS.hidden2*5, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)(self.embeddings)
            elif decoder == 'dedicom':
                self.edge_type2decoder[i, j] = DEDICOMDecoder(
                    input_dim=FLAGS.hidden2, logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)#(self.embeddings)
            else:
                raise ValueError('Unknown decoder type')

        self.latent_inters = []
        self.latent_varies = []
        for edge_type in self.edge_types:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types[edge_type]):
                if decoder == 'innerproduct':
                    glb = tf.eye(FLAGS.hidden2*num , FLAGS.hidden2*num)
                    loc = tf.eye(FLAGS.hidden2*num, FLAGS.hidden2*num)
                elif decoder == 'distmult':
                    glb = tf.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = tf.eye(FLAGS.hidden2*3, FLAGS.hidden*3)
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = tf.eye(FLAGS.hidden2*5, FLAGS.hidden2*5)
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = tf.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')

                self.latent_inters.append(glb)
                self.latent_varies.append(loc)
