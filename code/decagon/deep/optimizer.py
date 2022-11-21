#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
# -*- coding: utf-8 -*-

class DecagonOptimizer(object):
    def __init__(self, embeddings, latent_inters, latent_varies,
                 degrees, edge_types, edge_type2dim, placeholders,
                 margin=0.1, neg_sample_weights=1., batch_size=100):
        self.embeddings= embeddings

        self.latent_inters = latent_inters
        self.latent_varies = latent_varies

        self.edge_types = edge_types
        self.degrees = degrees  #
        self.edge_type2dim = edge_type2dim  # {(0, 0): [(1512, 1512), (1512, 1512)], (0, 1): [(1512, 708)], (0, 2): [(1512, 5603)],...
        self.obj_type2n = {i: self.edge_type2dim[i,j][0][0] for i, j in self.edge_types} # {0: 1512, 1: 708, 2: 5603, 3: 4192}
        #print('self.edge_type2dim:',self.edge_type2dim)
        #print('self.obj_type2n:',self.obj_type2n)

        self.margin = margin  # 0.1
        self.batch_size = batch_size  # 128

        self.neg_sample_weights = neg_sample_weights  #

        self.inputs = placeholders['batch']#print('self.inputs:',self.inputs)
        self.batch_edge_type_idx = placeholders['batch_edge_type_idx']
        self.batch_row_edge_type = placeholders['batch_row_edge_type']
        self.batch_col_edge_type = placeholders['batch_col_edge_type']
        self.loss_w = placeholders['loss_w']



        print('self.batch_edge_type_idx:',self.batch_edge_type_idx)


        self.row_inputs = tf.squeeze(gather_cols(self.inputs, [0]))  # shape=(None, None)说明是二维张量
        #print('gather_cols(self.inputs, [0]):',gather_cols(self.inputs, [0]))
        #Tensor("optimizer/gather_cols_1/Reshape_3:0", shape=(None, None), dtype=int64)
        #print('self.row:',self.row_inputs) # Tensor("optimizer/Squeeze:0", dtype=int64)
        self.col_inputs = tf.squeeze(gather_cols(self.inputs, [1]))
        #print('gather_cols(self.inputs, [1]):', gather_cols(self.inputs, [1]))
        #Tensor("optimizer/gather_cols_2/Reshape_3:0", shape=(None, None), dtype=int64)

        obj_type_n = [self.obj_type2n[i] for i in range(len(self.embeddings))] # obj_type_n: [1512, 708, 5603, 4192]
        #print('obj_type_n:',obj_type_n)
        #print('len(self.embeddings)',len(self.embeddings))  # 4

        self.obj_type_lookup_start = tf.cumsum([0] + obj_type_n[:-1])  # obj_type_n[:-1] : [1512, 708, 5603]
        print('obj:',[0] + obj_type_n[:-1])  # obj: [0, 1512, 708, 5603]

        sess = tf.Session()
        print('self.obj_type_lookup_start:',sess.run(self.obj_type_lookup_start))  # [   0 1512 2220 7823]

        self.obj_type_lookup_end = tf.cumsum(obj_type_n)
        print('obj_type:', sess.run(self.obj_type_lookup_end))  # [ 1512  2220  7823 12015]
        labels = tf.reshape(tf.cast(self.row_inputs, dtype=tf.int64), [self.batch_size, 1])  # cast是执行张量数据类型的转换
        #print('labels:',labels)  # Tensor("optimizer/Reshape:0", shape=(128, 1), dtype=int64)


        neg_samples_list = []
        for i, j in self.edge_types:  # 进行负采样
            for k in range(self.edge_types[i,j]):
                print('k:',k)
                neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                    true_classes=labels,   # 目标类
                    num_true=1,   # 每个训练示例的目标类数
                    num_sampled=self.batch_size,  # 随机抽样的类数
                    unique=False,
                    range_max=len(self.degrees[i][k]),  # 可能的类数
                    distortion=0.75,
                    unigrams=self.degrees[i][k].tolist())
                neg_samples_list.append(neg_samples)
        self.neg_samples = tf.gather(neg_samples_list, self.batch_edge_type_idx)  # 只获得任务的负样本


        self.preds = self.batch_predict(self.row_inputs, self.col_inputs)
        # 若想执行下一步的对角线，那么每一个batch_size的预测值(是不同类型节点的情况下)在对角线上。那么就需要对原始batch_size进行改造
        # 也就是将每一个batch_size大小中对应的样本对应成二维张量的对角线
        self.outputs = tf.diag_part(self.preds)  # tf.diag_part的两维必须相等,取对角线上的值
        self.outputs = tf.reshape(self.outputs, [-1])  # tf.reshape(t,[-1]) 表示的是将张量t展成一维

        self.neg_preds = self.batch_predict(self.neg_samples, self.col_inputs)
        self.neg_outputs = tf.diag_part(self.neg_preds)
        self.neg_outputs = tf.reshape(self.neg_outputs, [-1])

        self.predict()

        self._build()

    def batch_predict(self, row_inputs, col_inputs):
        concatenated = tf.concat(self.embeddings, 0)   # shape=(None, 320)

        #print('ind_start:', concatenated)
        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)  # batch_row_edge_type 节点类型
        #sess = tf.Session()
        #print('ind_start:',sess.run(ind_start))

        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)   #
        row_embeds = tf.gather(row_embeds, row_inputs)  #从该类型的所有 选出每次batch_size中的表示


        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)
        col_embeds = tf.gather(col_embeds, col_inputs)


        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        #print('latent_inter:', latent_inter) # shape=(320*320)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)  # product3:(128,320)
        preds = tf.matmul(product3, tf.transpose(col_embeds))   # preds:(128,128)
        #print('product3:',product3)

        return preds

    def predict(self):
        concatenated = tf.concat(self.embeddings, 0)

        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_row_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_row_edge_type)
        indices = tf.range(ind_start, ind_end)
        row_embeds = tf.gather(concatenated, indices)

        ind_start = tf.gather(self.obj_type_lookup_start, self.batch_col_edge_type)
        ind_end = tf.gather(self.obj_type_lookup_end, self.batch_col_edge_type)
        indices = tf.range(ind_start, ind_end)
        col_embeds = tf.gather(concatenated, indices)


        latent_inter = tf.gather(self.latent_inters, self.batch_edge_type_idx)
        latent_var = tf.gather(self.latent_varies, self.batch_edge_type_idx)

        product1 = tf.matmul(row_embeds, latent_var)
        product2 = tf.matmul(product1, latent_inter)
        product3 = tf.matmul(product2, latent_var)
        self.predictions = tf.matmul(product3, tf.transpose(col_embeds))

    def _build(self):

        self.cost = self._xent_loss(self.outputs, self.neg_outputs) * self.loss_w # 经过交叉熵计算正采样和负采样损失
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)



    def _xent_loss(self, aff, neg_aff):   # 交叉熵损失优化
        """Cross-entropy optimization."""
        # l2_loss = tf.add_n(tf.get_collection("l2_reg"))
        # tf.add_to_collection('l2_reg', tf.contrib.layers.l2_regularizer(1.0)(self.vars['weights_%d' % k]))
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss


def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.
    # indices : [0],[1]
    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input  : 检查输入数据,将其转换张量
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        #print('params:',params)  # Tensor("batch:0", dtype=int64)
        #print('indices:',indices) # Tensor("optimizer/gather_cols/indices:0", shape=(1,), dtype=int32)

        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)  # shape=(None,)  : 第一维是128，因为第一维是行的个数
        #print('p_shape:',p_shape) # Tensor("optimizer/gather_cols/Shape:0", shape=(None,), dtype=int32)
        p_flat = tf.reshape(params, [-1])  # tf.reshape(t,[-1]) 表示的是将张量t展成一维
        # 注意，此时input变为256维，排一列

        #print('p_flat:',p_flat)  # shape=(None,)
        #print('p_flat:', p_flat.shape)

        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],[-1, 1]) + indices, [-1])

        #print('tf.range(0, p_shape[0]) * p_shape[1]',tf.range(0, p_shape[0]) * p_shape[1])
        #tf.reshape(tensor,[-1,1])将张量变为一维列向量
        #shape[0]是计算行的个数    shape[1]是计算列的个数

        return tf.reshape(tf.gather(p_flat, i_flat), [p_shape[0], -1])


