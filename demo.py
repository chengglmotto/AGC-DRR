import random

import logging
from torch_scatter import scatter
import opt as opt
import utils
from opt import args
import torch
import torch.nn.functional as F
import numpy as np
from GAE import IGAE,IGAE_encoder

from utils import setup_seed,load_plantiod
from train import Train_gae
from sklearn.decomposition import PCA
from load_data import *


import warnings

from view_learner import ViewLearner

warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
setup_seed(np.random.randint(1000))


import pandas as pd

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.model_save_path = 'model/model_save_gae/{}_gae.pkl'.format(opt.args.name)



print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

# x = np.loadtxt(opt.args.data_path, dtype=float)
# y = np.loadtxt(opt.args.label_path, dtype=int)
#
# adj = torch.load('adj')
# adj = adj.to_dense()
# edge_index1=np.genfromtxt(opt.args.graph_save_path, dtype=np.int32)
# edge_index1 = edge_index1.transpose()
#
#

# dataset = LoadDataset(x1)
# data = torch.Tensor(dataset.x).to(device)

data = load_plantiod(opt.args.name)
x = data.x
pca1 = PCA(n_components=opt.args.n_components)
x = torch.tensor(pca1.fit_transform(x),dtype=torch.float32).to(device)
y = data.y
edge_index = data.edge_index
adj = utils.to_sparse_tensor(edge_index,x.shape[0]).to_dense()
adj = F.normalize(adj,dim=1)

model_gae = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1,
        gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_enc_3=opt.args.gae_n_enc_3,
        n_input=x.shape[1]
    ).to(device)

view_learner = ViewLearner(
        IGAE_encoder(gae_n_enc_1=opt.args.gae_n_enc_1,
                     gae_n_enc_2=opt.args.gae_n_enc_2,
                     gae_n_enc_3=opt.args.gae_n_enc_3,
                     n_input=x.shape[1]),
    ).to(device)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []
for i in range(10):
    best_acc,best_nmi,best_ari,best_f1 = Train_gae(model_gae,view_learner,x.to(device),adj.to(device), np.array(y), edge_index.to(device))
    acc_list.append(best_acc)
    nmi_list.append(best_nmi)
    ari_list.append(best_ari)
    f1_list.append(best_f1)

acc_list = np.array(acc_list)
nmi_list = np.array(nmi_list)
ari_list = np.array(ari_list)
f1_list = np.array(f1_list)
print(acc_list.mean(), "±", acc_list.std())
print(nmi_list.mean(), "±", nmi_list.std())
print(ari_list.mean(), "±", ari_list.std())
print(f1_list.mean(), "±", f1_list.std())

