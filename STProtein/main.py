import pandas as pd
import numpy as np
import scanpy as sc
import os
import torch
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
from model.utils import fix_seed
from model.utils import pca
from model.utils import clr_normalize_each_cell
from model.utils import Cal_Spatial_Net
from model.train import train_STProtein,train_Transfer_STProtein
from sklearn import metrics
from utils import clustering
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score

fix_seed(2025)


def RMSE(true,pred):
    true_array = np.array(true,dtype=np.float32).flatten()
    pred_array = np.array(pred,dtype=np.float32).flatten() 
    rmse = metrics.mean_squared_error(true_array, pred_array)**0.5
    return rmse

def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)
    return list

warnings.filterwarnings('ignore')
# Environment configuration. SpatialGlue pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load data
file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# name = 'Mouse_Spleen'
# name = 'Mouse_Thymus'
name = 'Human_Lymph_Node'
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/adata_all_mouse_spleen_rep2.h5ad')
ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/adata_all_human_lymph_node_D1.h5ad')
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/adata_all_mouse_thymus2.h5ad')
# adata_omics1 = sc.read_h5ad(file_fold + 'Dataset1_Mouse_Spleen1/adata_RNA.h5ad')
# adata_omics2 = sc.read_h5ad(file_fold + 'Dataset1_Mouse_Spleen1/adata_ADT.h5ad')
# adata_omics1 = sc.read_h5ad(file_fold + 'Dataset5_Mouse_Thymus3/adata_RNA.h5ad')
# adata_omics2 = sc.read_h5ad(file_fold + 'Dataset5_Mouse_Thymus3/adata_ADT.h5ad')
adata_omics1 = sc.read_h5ad(file_fold + 'Dataset11_Human_Lymph_Node_A1/adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'Dataset11_Human_Lymph_Node_A1/adata_ADT.h5ad')
adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
# test_adata_omics1 = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_RNA.h5ad') 
# test_adata_omics2 = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_ADT.h5ad')
# test_adata_omics1 = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_RNA.h5ad')
# test_adata_omics2 = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# ground = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
test_adata_omics1 = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_RNA.h5ad')
test_adata_omics2 = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_ADT.h5ad')
test_adata_omics1.var_names_make_unique()
test_adata_omics2.var_names_make_unique()
# ground.var_names_make_unique()

#Normalization
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=4000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)
sc.pp.highly_variable_genes(test_adata_omics1, flavor="seurat_v3", n_top_genes=4000)
sc.pp.normalize_total(test_adata_omics1, target_sum=1e4)
sc.pp.log1p(test_adata_omics1)
sc.pp.scale(test_adata_omics1)

clr_normalize_each_cell(adata_omics2)
# sc.pp.normalize_total(adata_omics2)
# sc.pp.log1p(adata_omics2)
sc.pp.scale(adata_omics2)
clr_normalize_each_cell(test_adata_omics2)
# sc.pp.log1p(test_adata_omics2)
# sc.pp.normalize_total(test_adata_omics2)
# sc.pp.log1p(test_adata_omics2)
sc.pp.scale(test_adata_omics2)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

test_adata_omics1_high = test_adata_omics1[:, test_adata_omics1.var['highly_variable']]
test_adata_omics1.obsm['feat'] = pca(test_adata_omics1_high, n_comps=test_adata_omics2.n_vars-1)

adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() 
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
test_adata_omics2 = test_adata_omics2[test_adata_omics1.obs_names].copy() 
test_adata_omics2.obsm['feat'] = pca(test_adata_omics2, n_comps=test_adata_omics2.n_vars-1)

# if name == 'Mouse_Thymus':
#     Cal_Spatial_Net(adata_omics1, model="KNN", n_neighbors=6)
#     Cal_Spatial_Net(adata_omics2, model="KNN", n_neighbors=6)
#     Cal_Spatial_Net(test_adata_omics1, model="KNN", n_neighbors=6)
#     Cal_Spatial_Net(test_adata_omics2, model="KNN", n_neighbors=6)
# else:
# Radius
# Cal_Spatial_Net(adata_omics1, model="Radius", radius=2)
# Cal_Spatial_Net(adata_omics2, model="Radius", radius=2)
# Cal_Spatial_Net(test_adata_omics1, model="Radius", radius=2)
# Cal_Spatial_Net(test_adata_omics2, model="Radius", radius=2)
Cal_Spatial_Net(adata_omics1, model="KNN", n_neighbors=3)
Cal_Spatial_Net(adata_omics2, model="KNN", n_neighbors=3)
Cal_Spatial_Net(test_adata_omics1, model="KNN", n_neighbors=3)
Cal_Spatial_Net(test_adata_omics2, model="KNN", n_neighbors=3)

out, model = train_STProtein(adata = adata_omics1,
                  ground_truth= torch.FloatTensor(adata_omics2.X),
                  feature_key="feat",
                  edge_key="edgeList",
                  weights=[5,3], # 5:3
                  n_epochs=12000, # 12000/15000
                  weight_decay=0.001
                  )

target_1 = torch.FloatTensor(adata_omics2.X).to(device)

# test_z, _, _ = train_Transfer_STProtein(
#         pretrained_model= model, 
#         adata=test_adata_omics1, 
#         target_emb=torch.FloatTensor(test_adata_omics2.X).shape[1],  
#         n_epochs=100
#     )

test_x1, test_edge_index1 = torch.FloatTensor(test_adata_omics1.obsm["feat"]), torch.LongTensor(test_adata_omics1.uns["edgeList"])
test_z,test_out = model(test_x1.to(device), test_edge_index1.to(device))
target_2 = torch.FloatTensor(test_adata_omics2.X).to(device)
rmse = RMSE(target_2.to('cpu').detach().numpy(), test_z.to('cpu').detach().numpy())
print(rmse)
# mae = MAE(target_2.to('cpu').detach().numpy(), test_z.to('cpu').detach().numpy())
# print(mae)
# r2 = R2(target_2.to('cpu').detach().numpy(), test_z.to('cpu').detach().numpy())
# print(r2)
# pcc = PCC(target_2.to('cpu').detach().numpy(), test_z.to('cpu').detach().numpy())
# print(pcc)
# print(test_z.to('cpu').detach().numpy())
# print(target_2)

predict = test_adata_omics2.copy()
predict.X = test_z.to('cpu').detach().numpy()
prediction = predict.copy()
clr_normalize_each_cell(prediction)
sc.pp.scale(prediction)
prediction.obsm['embedding'] = prediction.X


# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/opt/miniforge/envs/STAligner/lib/R'
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
tool = 'mclust' # mclust, leiden, and louvain
if name == 'Mouse_Thymus':
    clustering(prediction, key='embedding', add_key='embedding', n_clusters=6, method=tool, use_pca=False)
    intersection = prediction.obs_names.intersection(ground_truth.obs_names)
    prediction = prediction[intersection]
elif name == 'Mouse_Spleen':
    clustering(prediction, key='embedding', add_key='embedding', n_clusters=5, method=tool, use_pca=True)
else:
    clustering(prediction, key='embedding', add_key='embedding', n_clusters=6, method=tool, use_pca=True)


label = ground_truth.obs['Pro_label_origi']
list = label.tolist()
path_1 = name+'_GT_list'
with open(path_1, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


label = prediction.obs[tool]
list = label.tolist()
path_2 = name+'_Our_list'
with open(path_2, 'w') as f:
    for num in list:
        f.write(f"{num}\n")

GT_list = read_list_from_file(path_1)
Our_list = read_list_from_file(path_2)

Our_Jaccard = jaccard(Our_list, GT_list)
print(f"our         jaccard: {Our_Jaccard*100:.6f}")
Our_F_measure = F_measure(Our_list, GT_list)
print(f"our         F_measure: {Our_F_measure*100:.6f}")
Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"our         Mutual Information: {Our_mutual_info*100:.6f}")
Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"Our         (NMI): {Our_nmi*100:.6f}")
Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"Our         (AMI): {Our_ami*100:.6f}")
Our_V = v_measure_score(GT_list, Our_list)
print(f"Our         V-measure: {Our_V*100:.6f}")
Our_homogeneity = homogeneity_score(GT_list, Our_list)
Our_completeness = completeness_score(GT_list, Our_list)
print(f"Our         Homogeneity: {Our_homogeneity*100:.6f} Completeness: {Our_completeness*100:.6f}")
Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"Our         (ARI): {Our_ari*100:.6f}")
Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"Our         (FMI): {Our_fmi*100:.6f}")
print(f"{Our_nmi*100:.2f}&{Our_ami*100:.2f}&{Our_fmi*100:.2f}&{Our_ari*100:.2f}&{Our_V*100:.2f}&{Our_F_measure*100:.2f}&{Our_Jaccard*100:.2f}&{Our_completeness*100:.2f}")