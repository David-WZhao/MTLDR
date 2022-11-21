import numpy as np
from matplotlib import pyplot
import scipy.sparse as sp
import time
import sklearn.preprocessing as prep


def minmax_scale(x_train):
    '''
    do min-max normalization
    '''
    preprocessor = prep.MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train


def standrad_scale(x_train):
    '''
    do standard normalization
    '''
    preprocessor = prep.StandardScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train


def drawHist(mat):
    pyplot.hist(mat, 100)  # 绘制直方图
    pyplot.xlabel('Numbers Distribution between 0 and 1')
    pyplot.ylabel('Frequency')
    pyplot.title('Cell-Gene Edges Total')
    pyplot.savefig('./sc_ppi/zeisel/cell_gene_distribution.png',dpi=300)


def drawHist_gene_gene(mat):
    pyplot.hist(mat, 100)
    pyplot.xlabel('Numbers Distribution between 0 and 1')
    pyplot.ylabel('Frequency')
    pyplot.title('Gene-Gene Edges Total')


def print_count(mat):
    print('[0.0~0.1)=', sum(sum(mat >= 0.0)) - sum(sum(mat >= 0.1))) # sum(mat>=0.0) 将一行的值>=0的个数累加;  本行的意思是确定元素在0.0~0.1的个数
    print('[0.1~0.2)=', sum(sum(mat >= 0.1)) - sum(sum(mat >= 0.2)))
    print('[0.2~0.3)=', sum(sum(mat >= 0.2)) - sum(sum(mat >= 0.3)))
    print('[0.3~0.4)=', sum(sum(mat >= 0.3)) - sum(sum(mat >= 0.4)))
    print('[0.4~0.5)=', sum(sum(mat >= 0.4)) - sum(sum(mat >= 0.5)))
    print('[0.5~0.6)=', sum(sum(mat >= 0.5)) - sum(sum(mat >= 0.6)))
    print('[0.6~0.7)=', sum(sum(mat >= 0.6)) - sum(sum(mat >= 0.7)))
    print('[0.7~0.8)=', sum(sum(mat >= 0.7)) - sum(sum(mat >= 0.8)))
    print('[0.8~0.9)=', sum(sum(mat >= 0.8)) - sum(sum(mat >= 0.9)))
    print('[0.9~1.0]=', sum(sum(mat >= 0.9)) - sum(sum(mat > 1.0)))



def load_drug_disease_adj(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: drug-disease=============')
    drug_disease_adjs = np.loadtxt(path)
    print(drug_disease_adjs.shape)

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist(drug_disease_adjs)

    print('Before:')
    print_count(drug_disease_adjs)

    # If change by the threshold?
    if threshold==0:pass
    else:drug_disease_adjs[drug_disease_adjs < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:drug_disease_adjs[(drug_disease_adjs !=0)] = 1  # 不为0的值为1(经过阈值筛选)

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    disease_drug_adjs = drug_disease_adjs.transpose() # transpose()转置
    print('COUNT:   After threshold chose num_count = ',sum(sum(drug_disease_adjs>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(drug_disease_adjs>threshold))/(drug_disease_adjs.shape[0]*drug_disease_adjs.shape[1]),3))+'%')  # round函数表示四舍五入
    print('COUNT:   Drug    numbers = ',drug_disease_adjs.shape[0])
    print('COUNT:   Disease numbers = ', drug_disease_adjs.shape[1])

    # Sparse process.....
    drug_disease_adj = sp.csr_matrix(drug_disease_adjs)  # 稀疏表示
    disease_drug_adj = sp.csr_matrix(disease_drug_adjs)  # 稀疏表示
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  drug_disease_adj,disease_drug_adj  # 药物-疾病邻接构造完毕



# load_drug_protein_interactions and drug protein
def load_drug_protein_adj(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: drug-protein=============')
    drug_protein_adjs = np.loadtxt(path)
    print(drug_protein_adjs.shape)

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist(drug_protein_adjs)

    print('Before:')
    print_count(drug_protein_adjs)

    # If change by the threshold?
    if threshold==0:pass
    else:drug_protein_adjs[drug_protein_adjs < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:drug_protein_adjs[(drug_protein_adjs !=0)] = 1  # 不为0的值为1(经过阈值筛选)

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    protein_drug_adjs = drug_protein_adjs.transpose() # transpose()转置
    print('COUNT:   After threshold chose num_count = ',sum(sum(drug_protein_adjs>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(drug_protein_adjs>threshold))/(drug_protein_adjs.shape[0]*drug_protein_adjs.shape[1]),3))+'%')  # round函数表示四舍五入
    print('COUNT:   Drug    numbers = ',drug_protein_adjs.shape[0])
    print('COUNT:   Protein numbers = ', drug_protein_adjs.shape[1])

    # Sparse process.....
    protein_drug_adj = sp.csr_matrix(protein_drug_adjs)  # 稀疏表示
    drug_protein_adj = sp.csr_matrix(drug_protein_adjs)  # 稀疏表示
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  drug_protein_adj,protein_drug_adj  # 药物-蛋白质邻接构造完毕

# load_drug_protein_interactions and drug protein
def load_drug_mirna_adj(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: drug-mirna=============')
    drug_mirna_adjs = np.loadtxt(path)
    print(drug_mirna_adjs.shape)

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist(drug_mirna_adjs)

    print('Before:')
    print_count(drug_mirna_adjs)

    # If change by the threshold?
    if threshold==0:pass
    else:drug_mirna_adjs[drug_mirna_adjs < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:drug_mirna_adjs[(drug_mirna_adjs !=0)] = 1  # 不为0的值为1(经过阈值筛选)

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    mirna_drug_adjs = drug_mirna_adjs.transpose() # transpose()转置
    print('COUNT:   After threshold chose num_count = ',sum(sum(drug_mirna_adjs>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(drug_mirna_adjs>threshold))/(drug_mirna_adjs.shape[0]*drug_mirna_adjs.shape[1]),3))+'%')  # round函数表示四舍五入
    print('COUNT:   Drug    numbers = ',drug_mirna_adjs.shape[0])
    print('COUNT:   mirna numbers = ', drug_mirna_adjs.shape[1])

    # Sparse process.....
    mirna_drug_adj = sp.csr_matrix(mirna_drug_adjs)  # 稀疏表示
    drug_mirna_adj = sp.csr_matrix(drug_mirna_adjs)  # 稀疏表示
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  drug_mirna_adj,mirna_drug_adj  # 药物-蛋白质邻接构造完毕


def load_mirna_disease_adj(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: mirna-disease=============')
    mirna_disease_adjs = np.loadtxt(path)
    print(mirna_disease_adjs.shape)

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist(mirna_disease_adjs)

    print('Before:')
    print_count(mirna_disease_adjs)

    # If change by the threshold?
    if threshold==0:pass
    else:mirna_disease_adjs[mirna_disease_adjs < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:mirna_disease_adjs[(mirna_disease_adjs !=0)] = 1  # 不为0的值为1(经过阈值筛选)

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    disease_mirna_adjs = mirna_disease_adjs.transpose() # transpose()转置
    print('COUNT:   After threshold chose num_count = ',sum(sum(mirna_disease_adjs>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(mirna_disease_adjs>threshold))/(mirna_disease_adjs.shape[0]*mirna_disease_adjs.shape[1]),3))+'%')  # round函数表示四舍五入
    print('COUNT:   mirna    numbers = ',mirna_disease_adjs.shape[0])
    print('COUNT:   disease numbers = ', mirna_disease_adjs.shape[1])

    # Sparse process.....
    mirna_disease_adj = sp.csr_matrix(mirna_disease_adjs)  # 稀疏表示
    disease_mirna_adj = sp.csr_matrix(disease_mirna_adjs)  # 稀疏表示
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  mirna_disease_adj,disease_mirna_adj  # 药物-蛋白质邻接构造完毕


def load_protein_disease_adj(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: protein-disease=============')
    protein_disease_adjs = np.loadtxt(path)
    print(protein_disease_adjs.shape)

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist(protein_disease_adjs)

    print('Before:')
    print_count(protein_disease_adjs)

    # If change by the threshold?
    if threshold==0:pass
    else:protein_disease_adjs[protein_disease_adjs < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:protein_disease_adjs[(protein_disease_adjs !=0)] = 1  # 不为0的值为1(经过阈值筛选)

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    disease_protein_adjs = protein_disease_adjs.transpose() # transpose()转置
    print('COUNT:   After threshold chose num_count = ',sum(sum(protein_disease_adjs>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(protein_disease_adjs>threshold))/(protein_disease_adjs.shape[0]*protein_disease_adjs.shape[1]),3))+'%')  # round函数表示四舍五入
    print('COUNT:   Protein    numbers = ',protein_disease_adjs.shape[0])
    print('COUNT:   disease numbers = ', protein_disease_adjs.shape[1])

    # Sparse process.....
    protein_disease_adj = sp.csr_matrix(protein_disease_adjs)  # 稀疏表示
    disease_protein_adj = sp.csr_matrix(disease_protein_adjs)  # 稀疏表示
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  protein_disease_adj,disease_protein_adj  # 药物-蛋白质邻接构造完毕


def Load_Drug_Adj_Togerther(drug_drug_path = '',drug_drug_sim_chemical_path = '',
                            drug_drug_sim_interaction_path = '',\
                            drug_drug_sim_disease_path=''):
    #b1 = np.loadtxt(drug_drug_sim_chemical_path)  #此时b的值还没变
    print('Load_Drug_Adj_Togerther')
    th = 0.5
    a = np.loadtxt(drug_drug_path)
    b = np.loadtxt(drug_drug_sim_chemical_path) # 大于 0.5为1  小于为0
    c = np.loadtxt(drug_drug_sim_interaction_path)
    d = np.loadtxt(drug_drug_sim_disease_path)

    print('-------------Before--------------')
    print_count(a)
    # H = sum(sum(a>=0.0))
    # J = sum(sum(a>=0.1))

    #th = 0.5
    b[b >= th] = 1
    b[b < th] = 0

    c[c >= th] = 1
    c[c < th] = 0

    d[d >= 0.8] = 1
    d[d < 0.8] = 0


    Final = a+b+c+d
    # Final = a
    Final[Final >= 1] = 1

    for i in range(Final.shape[0]):  # shape[0] 输出行数
        for j in range(Final.shape[1]):  # shape[1] 输出列数
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1: # 将邻接矩阵表示为对称矩阵的形式
                Final[j][i] = 1
    print('-------------After--------------')
    print_count(Final)
    Final = sp.csr_matrix(Final) # 处理稀疏矩阵   节省空间

    return Final  # drug-drug的邻接矩阵构造完毕

def Load_Protein_Adj_Togerther(protein_protein_path='', protein_protein_sim_sequence_path='',
                             protein_protein_sim_interaction_path='',protein_protein_sim_disease_path=''):
    print('Load_Protein_Adj_Togerther')
    th = 0.5
    a = np.loadtxt(protein_protein_path)
    b = np.loadtxt(protein_protein_sim_sequence_path)
    b = b/100
    c = np.loadtxt(protein_protein_sim_interaction_path)
    d = np.loadtxt(protein_protein_sim_disease_path)
    print('Before')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i][j] = 0
    print_count(a)
    b[b >= th] = 1
    b[b < th] = 0
    d[d >= 0.85] = 1
    d[d < 0.85] = 0
    c[c >= th] = 1
    c[c < th] = 0
    Final = a + b + c +d

    Final[Final >= 1] = 1

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1:
                Final[j][i] = 1
    print('After')

    print_count(Final)
    Final = sp.csr_matrix(Final)

    return Final  # 蛋白质-蛋白质邻接矩阵构造完毕


def Load_miRNA_Adj_Togerther(mirna_mirna_sim_function_path='', mirna_mirna_sim_drug_path='',
                             mirna_mirna_sim_disease_path=''):
    print('Load_miRNA_Adj_Togerther')
    th = 0.5
    a = np.loadtxt(mirna_mirna_sim_function_path)
    b = np.loadtxt(mirna_mirna_sim_drug_path)
    c = np.loadtxt(mirna_mirna_sim_disease_path)
    print('Before')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i][j] = 0
    print_count(a)
    b[b >= th] = 1
    b[b < th] = 0
    c[c >= th] = 1
    c[c < th] = 0

    Final = a + b + c

    Final[Final >= 1] = 1

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1:
                Final[j][i] = 1
    print('After')
    print_count(Final)
    Final = sp.csr_matrix(Final)

    return Final  #

def Load_Disease_Adj_Togerther(disease_disease_path=''):
    print('Load_Disease_Adj_Togerther')
    #th = 0.5
    Final = np.loadtxt(disease_disease_path)
    print('After')
    print_count(Final)
    Final = sp.csr_matrix(Final)

    return Final  #


'''
    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1:
                Final[j][i] = 1
'''


def load_Adj_adj(threshold=0,toone=0,draw=0,sim_path=''):
    print('=========Loading.....: and feat=========='+sim_path)
    print('threshold = ', threshold)
    time1 = time.time()
    Adj = np.loadtxt(sim_path) # 加载数据

    # If draw the histPic?  是否需要绘制直方图
    if draw==0:pass
    else:drawHist_gene_gene(Adj)

    print('Before:')
    print_count(Adj)

    # If change by the threshold?
    if threshold==0:pass
    else:Adj[Adj < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:Adj[(Adj !=0)] = 1
    print('COUNT:    numbers=',Adj.shape[0])

    # get adj of cell_gene and feature
    # Protein_feature = np.loadtxt(feat_path)
    # print('COUNT:   protein feature=', Protein_feature.shape)
    print('COUNT:   After threshold chose num_count is ',sum(sum(Adj>0)))
    print('COUNT:   sparse rate is ',str(round(100*sum(sum(Adj>0))/(Adj.shape[0]*Adj.shape[1]),3))+'%')
    # print(Protein_Protein_sim)

    Adj =  sp.csr_matrix(Adj)
    # Protein_feature = sp.csr_matrix(Protein_feature)
    time2 = time.time()
    print('load time is = ',round(time2-time1,3))
    return Adj  # 蛋白质-疾病 /药物-疾病 /药物-副作用 邻接矩阵构造完毕

def load_Adj_adj_transpose(threshold=0,toone=0,draw=0,sim_path=''):
    print('=========Loading.....: and feat=========='+sim_path)
    print('threshold = ', threshold)
    time1 = time.time()
    Adj = np.loadtxt(sim_path)
    Adj = Adj.transpose()
    # If draw the histPic?
    if draw==0:pass
    else:drawHist_gene_gene(Adj)

    print('Before:')
    print_count(Adj)

    # If change by the threshold?
    if threshold==0:pass
    else:Adj[Adj < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:Adj[(Adj !=0)] = 1
    print('COUNT:    numbers=',Adj.shape[0])

    # get adj of cell_gene and feature
    # Protein_feature = np.loadtxt(feat_path)
    # print('COUNT:   protein feature=', Protein_feature.shape)
    print('COUNT:   After threshold chose num_count is ',sum(sum(Adj>0)))
    print('COUNT:   sparse rate is ',str(round(100*sum(sum(Adj>0))/(Adj.shape[0]*Adj.shape[1]),3))+'%')
    # print(Protein_Protein_sim)

    Adj =  sp.csr_matrix(Adj) # 处理稀疏
    # Protein_feature = sp.csr_matrix(Protein_feature)
    time2 = time.time()
    print('load time is = ',round(time2-time1,3))
    return Adj  # 疾病-蛋白质 /疾病-药物 /副作用-药物 邻接构造完毕
