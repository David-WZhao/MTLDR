#import tensorflow as tf

from . import inits
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems] # 非空元素的个数，如果x中非空元素的个数为4,则num_nonzero_elems为[4]

    random_tensor = keep_prob + tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), num_types=-1, **kwargs):
        self.edge_type = edge_type
        self.num_types = num_types
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout  # 丢失率
        self.adj_mats = adj_mats # 各个关联矩阵
        self.act = act   # 激活函数
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name): # 用来指定变量(参数)的作用域，因为每一层都有不同的参数
            for k in range(self.num_types):  # num_types 是关联矩阵的字典值，比如(0,0)对应2,那么k为2
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim[self.edge_type[1]], output_dim, name='weights_%d' % k)   # 定义相应的权重矩阵
        #print('edge_type_dim:',self.edge_type[1])  edge_type[1]: 0 1 2 0 1 2 3 0 1 1

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):   # 同上
            #print('inputs2:',inputs)
            x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat[self.edge_type[1]]) # 对初始输入数据进行稀疏化处理
            #print('x:',x.size())
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)
            outputs.append(self.act(x))

        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)  # dim=1  按行进行正则化
        #print('outputs:',outputs.shape)
        return outputs


class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['weights_%d' % k] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights_%d' % k)

    def _call(self, inputs):
        outputs = []
        for k in range(self.num_types):
            x = tf.nn.dropout(inputs, 1-self.dropout)

            x = tf.matmul(x, self.vars['weights_%d' % k])
            x = tf.sparse_tensor_dense_matmul(self.adj_mats[self.edge_type][k], x)

            outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)

        #print('ssssssssssssssssssssssssssssssssssssssssssssssss%d')  # 能够运行到这一步
        return outputs


class DEDICOMDecoder(MultiLayer):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
  
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = inits.weight_variable_glorot(
                input_dim, input_dim, name='global_interaction')
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='local_variation_%d' % k)
                self.vars['local_variation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)


            relation = tf.diag(self.vars['local_variation_%d' % k])
            product1 = tf.matmul(inputs_row, relation)
            product2 = tf.matmul(product1, self.vars['global_interaction'])
            product3 = tf.matmul(product2, relation)
            rec = tf.matmul(product3, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs

class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='relation_%d' % k)
                self.vars['relation_%d' % k] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)

            relation = tf.diag(self.vars['relation_%d' % k])
            intermediate_product = tf.matmul(inputs_row, relation)
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb%d')
        return outputs

class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            for k in range(self.num_types):
                self.vars['relation_%d' % k] = inits.weight_variable_glorot(
                    input_dim, input_dim, name='relation_%d' % k)

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []
        for k in range(self.num_types):
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)

            intermediate_product = tf.matmul(inputs_row, self.vars['relation_%d' % k])
            rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        print('hhhh',outputs)
        print('cccccccccccccccccccccccccccccccccccccccccccccccccc%d')
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

        #print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')  # 能够运行到这一步
    def _call(self, inputs):
        i, j = self.edge_type  #
        outputs = []
        for k in range(self.num_types):
            #print('inputs-----------------------:',inputs)
            #print('inputs[i]-----------------------:', inputs[i])
            #print('inputs[j]-----------------------:', inputs[j])
            inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)

            rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        #print('ssssssssssssssssssssssssssssssssssssssssssssssss%d')  #print('outputs:',outputs)  #print('outputs2:',len(outputs))
        return outputs
