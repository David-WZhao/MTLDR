from __future__ import division
from __future__ import print_function
from operator import itemgetter

import time
import os

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import scipy.sparse as sp

from sklearn import metrics
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
from decagon.utility import loadData

import sys
import datetime


# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')



data_set = 'DataSet'
print('data_set',data_set)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1',64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')



#01
AUROC_01_list = []
AUPR_01_list  = []
APatK_01_list = []
ACC_01_list   = []
F1_01_list    = []
MSE_01_list   = []
MAE_01_list   = []

#10

AUROC_10_list = []
AUPR_10_list  = []
APatK_10_list = []
ACC_10_list   = []
F1_10_list    = []
MSE_10_list   = []
MAE_10_list   = []


#/home/ysw/work/MTHGCN-demo/DataSet/mat_drug_drug.txt
# about drug-drug
drug_drug_path = '../MTHGCN-demo2/DataSet/mat_drug_drug.txt' # 药物-药物相互作用
drug_drug_sim_chemical_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_Drugs_chemical.txt'
drug_drug_sim_interaction_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_drug_drug_interaction.txt'
drug_drug_sim_disease_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_drug_disease.txt'

# 1interaction + 3sim +1 for drug 5 networks
# About Protein
protein_protein_path = '../MTHGCN-demo2/DataSet/mat_protein_protein.txt' # 蛋白质-蛋白质相互作用矩阵
protein_protein_sim_sequence_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_Proteins.txt' # 根据序列构造蛋白质-蛋白质相似矩阵
protein_protein_sim_disease_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_protein_disease.txt'
protein_protein_sim_interaction_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mat_protein_protein.txt'

# about mirna
mirna_mirna_sim_function_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mirna_mirna.txt'
mirna_mirna_sim_drug_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mirna_mirna(drug).txt'
mirna_mirna_sim_disease_path = '../MTHGCN-demo2/DataSet/sim_mat/sim_mirna(disease).txt'

# about disease
disease_disease_path = '../MTHGCN-demo2/DataSet/disease_disease_adj.txt'

# About associations...
drug_disease_path = '../MTHGCN-demo2/DataSet/mat_drug_disease.txt'
drug_mirna_path = '../MTHGCN-demo2/DataSet/mat_drug_mirna.txt'
drug_protein_path = '../MTHGCN-demo2/DataSet/mat_drug_protein.txt'
protein_disease_path = '../MTHGCN-demo2/DataSet/mat_protein_disease.txt'
mirna_disease_path = '../MTHGCN-demo2/DataSet/mat_mirna_disease.txt'


# 经过稀疏处理后的数据表示形式
drug_drug_adj = loadData.Load_Drug_Adj_Togerther(drug_drug_path=drug_drug_path,
                                                 drug_drug_sim_chemical_path=drug_drug_sim_chemical_path,
                                                 drug_drug_sim_interaction_path=drug_drug_sim_interaction_path,
                                                 drug_drug_sim_disease_path=drug_drug_sim_disease_path)

protein_protein_adj = loadData.Load_Protein_Adj_Togerther(protein_protein_path=protein_protein_path,
                                                          protein_protein_sim_sequence_path=protein_protein_sim_sequence_path,
                                                          protein_protein_sim_interaction_path=protein_protein_sim_interaction_path,
                                                          protein_protein_sim_disease_path=protein_protein_sim_disease_path)

mirna_mirna_adj = loadData.Load_miRNA_Adj_Togerther(mirna_mirna_sim_function_path=mirna_mirna_sim_function_path,
                                                    mirna_mirna_sim_drug_path=mirna_mirna_sim_drug_path,
                                                    mirna_mirna_sim_disease_path=mirna_mirna_sim_disease_path)

disease_disease_adj = loadData.Load_Disease_Adj_Togerther(disease_disease_path=disease_disease_path)

# 药物-疾病关联
drug_disease_adj,disease_drug_adj = loadData.load_drug_disease_adj(path=drug_disease_path)
#药物-蛋白关联
drug_protein_adj,protein_drug_adj = loadData.load_drug_protein_adj(path=drug_protein_path)
#药物-mirna关联
drug_mirna_adj,mirna_drug_adj = loadData.load_drug_mirna_adj(path=drug_mirna_path)
#mirna_疾病关联
mirna_disease_adj,disease_mirna_adj = loadData.load_mirna_disease_adj(path=mirna_disease_path)
#蛋白质-疾病关联
protein_disease_adj,disease_protein_adj = loadData.load_protein_disease_adj(path=protein_disease_path)

# 10 fold cross-validation
for seed in range(0, 10):
    val_test_size = 0.1
    print('Current seed is :', seed)


    def get_accuracy_scores(edges_pos, edges_neg, edge_type):
        feed_dict.update({placeholders['dropout']: 0})
        feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
        feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
        feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
        rec = sess.run(opt.predictions, feed_dict=feed_dict)

        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        actual = []
        predicted = []
        edge_ind = 0

        # pos
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            # print('edge_type:',edge_type)   # edge_type: (1, 0, 0)  或者  edge_type: (0, 1, 0)
            # print('edge_type[:2]:', edge_type[:2])  # (1, 0)  或者  (0,1)
            # print('edge_type:[2]', edge_type[2])  # 0
            # print('u:',u)
            # print('v:',v)
            score = sigmoid(rec[u, v])
            preds.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

            actual.append(edge_ind)
            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_neg = []

        # neg
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds_neg.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_all = np.hstack([preds, preds_neg])
        preds_all = np.nan_to_num(preds_all)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

        # evatalution.....
        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)
        apk_sc = rank_metrics.apk(actual, predicted, k=10)

        return roc_sc, aupr_sc, apk_sc


    def get_final_accuracy_scores(edges_pos, edges_neg, edge_type):
        feed_dict.update({placeholders['dropout']: 0})
        feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
        feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
        feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
        rec = sess.run(opt.predictions, feed_dict=feed_dict)

        def sigmoid(x):
            return 1. / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        actual = []
        predicted = []
        edge_ind = 0

        # pos
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

            actual.append(edge_ind)
            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_neg = []
        # neg
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            score = sigmoid(rec[u, v])
            preds_neg.append(score)
            assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

            predicted.append((score, edge_ind))
            edge_ind += 1

        preds_all = np.hstack([preds, preds_neg])
        preds_all = np.nan_to_num(preds_all)
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

        # evatalution.....
        roc_sc = metrics.roc_auc_score(labels_all, preds_all)
        aupr_sc = metrics.average_precision_score(labels_all, preds_all)
        apk_sc = rank_metrics.apk(actual, predicted, k=50)
        FPR, TPR, thresholds = metrics.roc_curve(labels_all, preds_all)

        precision, recall, _ = metrics.precision_recall_curve(labels_all, preds_all)

        mse = metrics.mean_squared_error(labels_all, preds_all)
        mae = metrics.median_absolute_error(labels_all, preds_all)
        r2 = metrics.r2_score(labels_all, preds_all)
        #np.savetxt('./log/' + data_set + str(nowTime) + instruction + '/' + str(seed) + '' + str(
         #   edge_type) + '_' + '_true.txt', labels_all, fmt='%d')
        #np.savetxt('./log/' + data_set + str(nowTime) + instruction + '/' + str(seed) + '' + str(
         #   edge_type) + '_' + '_pred.txt', preds_all, fmt='%.3f')
        preds_all[preds_all >= 0.5] = 1
        preds_all[preds_all < 0.5] = 0
        acc = metrics.accuracy_score(labels_all, preds_all)
        f1 = metrics.f1_score(labels_all, preds_all, average='macro')
        return FPR, TPR, roc_sc, \
               precision, recall, aupr_sc, \
               apk_sc, thresholds, mse, mae, r2, acc, f1


    def construct_placeholders(edge_types):
        placeholders = {
            'batch': tf.placeholder(tf.int64, name='batch'),
            'batch_edge_type_idx': tf.placeholder(tf.int64, shape=(), name='batch_edge_type_idx'),
            'batch_row_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_row_edge_type'),
            'batch_col_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_col_edge_type'),
            'loss_w': tf.placeholder(tf.float32),
            'degrees': tf.placeholder(tf.int64),
            'dropout': tf.placeholder_with_default(0., shape=()),
        }  # dtype:数据类型, shape:数据形状， name:名称
        placeholders.update({  # 字典(Dictionary) update() 函数把字典 dict2 的键/值对更新到 dict 里。
            'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
            for i, j in edge_types for k in range(edge_types[i, j])})  # i,j表示(0,0),(0,1)...;k表示长度，比如(0,0)有两个，k就是2
        placeholders.update({
            'feat_%d' % i: tf.sparse_placeholder(tf.float32) for i, _ in edge_types})

        return placeholders


    #
    # data representation
    # 0 for protein / 1 for drug / 2 for disease / 3 for side-effect
    adj_mats_orig = {
        (0, 0): [protein_protein_adj, protein_protein_adj],  # type1
        # 为什么蛋白质-蛋白质是两个？蛋白质-蛋白质网络不是同质的吗？
        (0, 1): [protein_drug_adj],  # type2
        (0, 2): [protein_disease_adj],

        (1, 0): [drug_protein_adj],
        (1, 1): [drug_drug_adj, drug_drug_adj],  # type3
        (1, 2): [drug_disease_adj],
        (1, 3): [drug_mirna_adj],

        (2, 0): [disease_protein_adj],
        (2, 1): [disease_drug_adj],
        (2, 2): [disease_disease_adj,disease_disease_adj],
        (2 ,3): [disease_mirna_adj],

        (3, 1): [mirna_drug_adj],
        (3, 2): [mirna_disease_adj],
        (3, 3): [mirna_mirna_adj,mirna_mirna_adj],
    }  # 各种类型的边的表示

    # print(adj_mats_orig.get((0,0)))

    #
    protein_degrees = np.array(protein_protein_adj.sum(axis=0)).squeeze()
    # 对于二维数组，axis=1表示按行相加，0表示按列相加;  从数组的形状中删除单维条目：把1的维度去掉()  降维操作
    drug_degrees = np.array(drug_drug_adj.sum(axis=0)).squeeze()
    disease_degrees = np.array(disease_disease_adj.sum(axis=0)).squeeze()
    mirna_degrees = np.array(mirna_mirna_adj.sum(axis=0)).squeeze()

    degrees = {
        0: [protein_degrees, protein_degrees],
        1: [drug_degrees, drug_degrees],
        2: [disease_degrees,disease_degrees],
        3: [mirna_degrees,mirna_degrees]
    }

    # 各类型节点的特征表示，使用one-hot编码

    # # featureless (genes)
    protein_feat = sp.identity(1512)  #
    protein_nonzero_feat, protein_num_feat = protein_feat.shape
    protein_feat = preprocessing.sparse_to_tuple(protein_feat.tocoo())

    #
    # # features (drugs)
    drug_feat = sp.identity(708)
    # drug_feat = Drug_Drug_adj
    drug_nonzero_feat, drug_num_feat = drug_feat.shape
    drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

    # data representation
    diease_feat = sp.identity(5603)
    diease_nonzero_feat, diease_num_feat = diease_feat.shape
    diease_feat = preprocessing.sparse_to_tuple(diease_feat.tocoo())
    # NOTICE

    mirna_feat = sp.identity(1023)
    mirna_nonzero_feat, mirna_num_feat = mirna_feat.shape
    mirna_feat = preprocessing.sparse_to_tuple(mirna_feat.tocoo())
    #extra_mirna_feat = mirna_feat
    # NOTICE

    num_feat = {
        0: protein_num_feat,  # 1512
        1: drug_num_feat,  # 708
        2: diease_num_feat,  # 5603
        3: mirna_num_feat,  # 1023
    }
    nonzero_feat = {
        0: protein_nonzero_feat,  # 1512
        1: drug_nonzero_feat,  # 708
        2: diease_nonzero_feat,  # 5603
        3: mirna_nonzero_feat  # 1023
    }
    feat = {
        0: protein_feat,
        1: drug_feat,
        2: diease_feat,
        3: mirna_feat
    }

    edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}  # 不同类型边的维度，也就是每种类型有多少条边

    # edge_types
    # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
    edge_type2decoder = {
        # 点积
        (0, 0): 'innerproduct',  # type1
        (0, 1): 'innerproduct',  # type2
        (0, 2): 'innerproduct',

        (1, 0): 'innerproduct',
        (1, 1): 'innerproduct',  # type3
        (1, 2): 'innerproduct',  # *
        (1, 3): 'innerproduct',

        (2, 0): 'innerproduct',
        (2, 1): 'innerproduct',# *
        (2, 2): 'innerproduct',
        (2, 3): 'innerproduct',

        (3, 1): 'innerproduct',
        (3, 2): 'innerproduct',
        (3, 3): 'innerproduct'
    }

    edge_types = {k: len(v) for k, v in
                  adj_mats_orig.items()}  # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (1, 3): 1, (2, 0): 1, (2, 1): 1, (3, 1): 1}
    print('edge_types', edge_types)
    num_edge_types = sum(edge_types.values())
    print("Edge types:", "%d" % num_edge_types)  #
    print('edge_types_item:', edge_types.items())
    # dict_items([((0, 0), 2), ((0, 1), 1), ((0, 2), 1), ((1, 0), 1), ((1, 1), 2), ((1, 2), 1), ((1, 3), 1), ((2, 0), 1), ((2, 1), 1), ((3, 1), 1)])

    # Important -- Do not evaluate/print validation performance every iteration as it can take
    # substantial amount of time    不要每次迭代都评估，打印验证的性能，因为占用时间
    PRINT_PROGRESS_EVERY = 20

    print("Defining placeholders")  # 定义占位符
    placeholders = construct_placeholders(edge_types)

    print("Create minibatch iterator")
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,  # 关系
        feat=feat,  #
        seed=seed,  # 10折交叉验证的fold
        data_set=data_set,
        edge_types=edge_types,  # 边的关系类型
        batch_size=FLAGS.batch_size,  # 
        val_test_size=val_test_size  # 0.1
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        data_set=data_set,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,  # degrees 在优化器
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}

    # Train model

    print("Train model")
    for epoch in range(FLAGS.epochs):

        minibatch.shuffle()
        itr = 0
        while not minibatch.end():

            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)  # 将训练集划分以batch_size大小

            feed_dict = minibatch.update_feed_dict(feed_dict=feed_dict, dropout=FLAGS.dropout,
                                                   placeholders=placeholders)

            t = time.time()

            # Training step: run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
            train_cost = outs[1]
            batch_edge_type = outs[2]

            if itr % PRINT_PROGRESS_EVERY == 0:
                print('itr:',itr)
            itr += 1

    print("Optimization finished!")

    for et in range(num_edge_types):
        print('et=', et)
        PRINT = 1
        if PRINT == 1:
            FPR, TPR, roc_score, \
            precision, recall, auprc_score, \
            apk_score, \
            thresholds, mse, mae, r2, acc, f1 = get_final_accuracy_scores(minibatch.test_edges,
                                                                          minibatch.test_edges_false,
                                                                          minibatch.idx2edge_type[et])
            # if et==1 or et==2:
            # edge_types
            # {(0, 0): 2, (0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 1): 2, (1, 2): 1, (2, 0): 1, (2, 1): 1, (2, 2): 1}
            if et == 7:
                AUROC_01_list.append(roc_score)
                AUPR_01_list.append(auprc_score)
                APatK_01_list.append(apk_score)
                ACC_01_list.append(acc)
                F1_01_list.append(f1)
                MSE_01_list.append(mse)
                MAE_01_list.append(mae)
            if et == 10:
                AUROC_10_list.append(roc_score)
                AUPR_10_list.append(auprc_score)
                APatK_10_list.append(apk_score)
                ACC_10_list.append(acc)
                F1_10_list.append(f1)
                MSE_10_list.append(mse)
                MAE_10_list.append(mae)

            print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
            print("Edge type:", "%04d" % et, "Test AUROC score", "{:.5f}".format(roc_score))
            print("Edge type:", "%04d" % et, "Test AUPRC score", "{:.5f}".format(auprc_score))
            print("Edge type:", "%04d" % et, "Test AP@k score", "{:.5f}".format(apk_score))
            print("Edge type:", "%04d" % et, "Test acc score", "{:.5f}".format(acc))
            print("Edge type:", "%04d" % et, "Test f1 score", "{:.5f}".format(f1))
            print("Edge type:", "%04d" % et, "Test mse score", "{:.5f}".format(mse))
            print("Edge type:", "%04d" % et, "Test mae score", "{:.5f}".format(mae))
            print("Edge type:", "%04d" % et, "Test r2 score", "{:.5f}".format(r2))
            print()

print('10-Flod-cross-val-result')

print('-----01------')
print('AUROC_01_list', AUROC_01_list)
print('AUPR_01_list', AUPR_01_list)
print('APatK_01_list', APatK_01_list)
print('ACC_01_list', ACC_01_list)
print('F1_01_list', F1_01_list)
print('MSE_01_list', MSE_01_list)
print('MAE_01_list', MAE_01_list)
print('AVG_AUROC_01_list', np.mean(AUROC_01_list).round(4))
print('AVG_AUPR_01_list', np.mean(AUPR_01_list).round(4))
print('AVG_APatK_01_list', np.mean(APatK_01_list).round(4))
print('AVG_ACC_01_list', np.mean(ACC_01_list).round(4))
print('AVG_F1_01_list', np.mean(F1_01_list).round(4))
print('AVG_MSE_01_list', np.mean(MSE_01_list).round(4))
print('AVG_MAE_01_list', np.mean(MAE_01_list).round(4))

print('-----10------')
print('AUROC_10_list', AUROC_10_list)
print('AUPR_10_list', AUPR_10_list)
print('APatK_10_list', APatK_10_list)
print('ACC_10_list', ACC_10_list)
print('F1_10_list', F1_10_list)
print('MSE_10_list', MSE_10_list)
print('MAE_10_list', MAE_10_list)
print('AVG_AUROC_10_list', np.mean(AUROC_10_list).round(4))
print('AVG_AUPR_10_list', np.mean(AUPR_10_list).round(4))
print('AVG_APatK_10_list', np.mean(APatK_10_list).round(4))
print('AVG_ACC_10_list', np.mean(ACC_10_list).round(4))
print('AVG_F1_10_list', np.mean(F1_10_list).round(4))
print('AVG_MSE_10_list', np.mean(MSE_10_list).round(4))
print('AVG_MAE_10_list', np.mean(MAE_10_list).round(4))

