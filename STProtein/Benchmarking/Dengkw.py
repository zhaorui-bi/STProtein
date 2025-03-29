#!/usr/bin/env python
# coding: utf-8

# In[271]:


import logging
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

logging.basicConfig(level=logging.INFO)


# In[272]:


import logging
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import scanpy as sc

from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge

logging.basicConfig(level=logging.INFO)
# name = 'Mouse_Spleen'
name = 'Mouse_Thymus'
# name = 'Human_Lymph_Node'
# path_root = "Results/Mouse_Spleen/" 
path_root = "Results/Mouse_Thymus/" 
# path_root = "Results/Human_Lymph_Node/" 
output_path = path_root
path = '/data/hulei/ZhaoruiJiang/Data/STProtein/'
logging.info('Reading data files...')
# input_train_mod1 = sc.read_h5ad(path + "Mouse_Spleen1.h5ad")
# input_test_mod1 = sc.read_h5ad(path + "Mouse_Spleen2.h5ad")
input_train_mod1 = sc.read_h5ad(path + "Mouse_Thymus3.h5ad")
input_test_mod1 = sc.read_h5ad(path + "Mouse_Thymus2.h5ad")
# input_train_mod1 = sc.read_h5ad(path + "Human_Lymph_Node_A1.h5ad")
# input_test_mod1 = sc.read_h5ad(path + "Human_Lymph_Node_D1.h5ad")
# train = sc.read_h5ad(data_path+"dataset"+train_id+"_adata.h5ad")
# train_data_path = "Dataset/adataset1.h5ad"
# test_data_path =  "Dataset/adataset2.h5ad"

# input_train_mod1 = sc.read(train_data_path)

# input_test_mod1 = sc.read(test_data_path)
file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
ground = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')


# In[273]:


if input_train_mod1.obs.index.duplicated().any():
    input_train_mod1.obs.reset_index(drop=True, inplace=True)
if input_test_mod1.obs.index.duplicated().any():
    input_test_mod1.obs.reset_index(drop=True, inplace=True)
if input_train_mod1.var.index.duplicated().any():
    input_train_mod1.var.reset_index(drop=True, inplace=True)
if input_test_mod1.var.index.duplicated().any():
    input_test_mod1.var.reset_index(drop=True, inplace=True)


# In[274]:


genes_inter = input_train_mod1.var_names.intersection(input_test_mod1.var_names)
input_train_mod1 = input_train_mod1[:,genes_inter].copy()
input_test_mod1 = input_test_mod1[:,genes_inter].copy()


# In[275]:


input_train_mod1.obs['batch'] = pd.Categorical(len(input_train_mod1.obs_names)*['batch1'])
input_test_mod1.obs['batch'] = pd.Categorical(len(input_test_mod1.obs_names)*['batch2'])


# In[276]:


pred_dimx = input_test_mod1.shape[0] #细胞数目
pred_dimy = input_train_mod1.obsm['protein_expression'].shape[1]  #蛋白质数目


# In[277]:


feature_obs = input_train_mod1.obs
# gs_obs = input_train_mod2.obs
gs_obs = input_train_mod1.obs


# In[278]:


batches = input_train_mod1.obs.batch.unique().tolist()
batch_len = len(batches)


# In[279]:


index = input_test_mod1.obsm['protein_expression'].index
columns = input_train_mod1.obsm['protein_expression'].columns


# In[280]:


input_train = ad.concat(
    {"train": input_train_mod1, "test": input_test_mod1},
    axis=0,
    join="outer",
    label="group",
    fill_value=0,
    index_unique="-"
)


# In[281]:


logging.info('Determine parameters by the modalities')
# mod1_type = input_train_mod1.var.feature_types[0]
# mod1_type = mod1_type.upper()
# mod2_type = input_train_mod2.var.feature_types[0]
# mod2_type = mod2_type.upper()
mod1_type = "GEX"
mod2_type = "ADT"
n_comp_dict = {
        ("GEX", "ADT"): (300, 70, 10, 0.2),
        ("ADT", "GEX"): (None, 50, 10, 0.2),
        ("GEX", "ATAC"): (1000, 50, 10, 0.1),
        ("ATAC", "GEX"): (100, 70, 10, 0.1)
        }
logging.info(f"{mod1_type}, {mod2_type}")
n_mod1, n_mod2, scale, alpha = n_comp_dict[(mod1_type, mod2_type)]
logging.info(f"{n_mod1}, {n_mod2}, {scale}, {alpha}")

# Do PCA on the input data
logging.info('Models using the Truncated SVD to reduce the dimension')


# In[282]:


if n_mod1 is not None and n_mod1 < input_train.shape[1]:
    embedder_mod1 = TruncatedSVD(n_components=n_mod1)
    mod1_pca = embedder_mod1.fit_transform(input_train.X).astype(np.float32)
    train_matrix = mod1_pca[input_train.obs['group'] == 'train']
    test_matrix = mod1_pca[input_train.obs['group'] == 'test']
else:
    train_matrix = input_train_mod1.to_df().values.astype(np.float32)
    test_matrix = input_test_mod1.to_df().values.astype(np.float32)

# if n_mod2 is not None and n_mod2 < input_train_mod2.shape[1]:
#     embedder_mod2 = TruncatedSVD(n_components=n_mod2)
#     train_gs = embedder_mod2.fit_transform(input_train_mod2.X).astype(np.float32)
# else:
#     train_gs = input_train_mod2.to_df().values.astype(np.float32)
if n_mod2 is not None and n_mod2 < input_train_mod1.obsm['protein_expression'].shape[1]:
    embedder_mod2 = TruncatedSVD(n_components=n_mod2)
    train_gs = embedder_mod2.fit_transform(input_train_mod1.obsm['protein_expression'].values).astype(np.float32)
else:
    embedder_mod2 = None 
    train_gs = input_train_mod1.obsm['protein_expression'].values.astype(np.float32)


# In[283]:


del input_train
del input_train_mod1
# del input_train_mod2
del input_test_mod1


# In[284]:


logging.info('Running normalization ...')
train_sd = np.std(train_matrix, axis=1).reshape(-1, 1)
train_sd[train_sd == 0] = 1
train_norm = (train_matrix - np.mean(train_matrix, axis=1).reshape(-1, 1)) / train_sd
train_norm = train_norm.astype(np.float32)
del train_matrix


# In[285]:


test_sd = np.std(test_matrix, axis=1).reshape(-1, 1)
test_sd[test_sd == 0] = 1
test_norm = (test_matrix - np.mean(test_matrix, axis=1).reshape(-1, 1)) / test_sd
test_norm = test_norm.astype(np.float32)
del test_matrix


# In[286]:


logging.info('Running KRR model ...')
y_pred = np.zeros((pred_dimx, pred_dimy), dtype=np.float32)
np.random.seed(2025)


# In[287]:


for _ in range(5):
    np.random.shuffle(batches)
    for batch in [batches[:batch_len//2], batches[batch_len//2:]]:
        # for passing the test
        if not batch:
            batch = [batches[0]]

        logging.info(batch)
        kernel = RBF(length_scale = scale)
        krr = KernelRidge(alpha=alpha, kernel=kernel)
        logging.info('Fitting KRR ... ')
        krr.fit(train_norm[feature_obs.batch.isin(batch)], 
                train_gs[gs_obs.batch.isin(batch)])
        # y_pred += (krr.predict(test_norm) @ embedder_mod2.components_)
        if embedder_mod2 is not None:
            y_pred += (krr.predict(test_norm).dot(embedder_mod2.components_))
        else:
            y_pred += krr.predict(test_norm)


# In[288]:


np.clip(y_pred, a_min=0, a_max=None, out=y_pred)
if mod2_type == "ATAC":
    np.clip(y_pred, a_min=0, a_max=1, out=y_pred)

logging.info('Storing annotated data...')
y_pred /= 10
protein_pred = pd.DataFrame(y_pred,index = index,columns = columns)
# output_path = "Results/" #path to results
protein_pred.to_csv(path_root+name+"_Dengkw.csv")


# In[289]:


file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# prediction = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_ADT.h5ad')
prediction = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# prediction = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_ADT.h5ad')
prediction.var_names_make_unique()
prediction.X= protein_pred.to_numpy()


# In[290]:


def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()


# In[291]:


clr_normalize_each_cell(prediction)
sc.pp.scale(prediction)


# In[292]:


prediction.obsm['embedding'] = prediction.to_df().to_numpy()


# In[293]:


import os
# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/opt/miniforge/envs/STAligner/lib/R'
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
tool = 'mclust' # mclust, leiden, and louvain
from utils import clustering
# clustering(prediction, key='embedding', add_key='embedding', n_clusters=5, method=tool, use_pca=True)


# In[294]:


# visualization
import matplotlib.pyplot as plt
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(prediction, use_rep='embedding', n_neighbors=10)
# sc.tl.umap(prediction)

# sc.pl.umap(prediction, color='embedding', ax=ax_list[0], title='Dengkw', s=20, show=False)
# sc.pl.embedding(prediction, basis='spatial', color='embedding', ax=ax_list[1], title='Dengkw', s=25, show=False)

# plt.tight_layout(w_pad=0.3)
# plt.show()


# In[295]:


# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + "adata_all_mouse_spleen_rep2.h5ad")
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_human_lymph_node_D1.h5ad')
ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_mouse_thymus2.h5ad')
# In[296]:

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
path_1 = output_path+'_GT_list'
with open(path_1, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


# In[297]:


label = prediction.obs['mclust']
list = label.tolist()
path_2 = output_path + 'Dengkw_list'
with open(path_2, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


# In[298]:


from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score


# In[299]:


def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)
    return list


# In[300]:


GT_list = read_list_from_file(path_1)
Our_list = read_list_from_file(path_2)


# In[301]:


Our_Jaccard = jaccard(Our_list, GT_list)
print(f"Dengkw         jaccard: {Our_Jaccard*100:.6f}")
Our_F_measure = F_measure(Our_list, GT_list)
print(f"Dengkw         F_measure: {Our_F_measure*100:.6f}")
Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"Dengkw         Mutual Information: {Our_mutual_info*100:.6f}")
Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"Dengkw         (NMI): {Our_nmi*100:.6f}")
Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"Dengkw         (AMI): {Our_ami*100:.6f}")
Our_V = v_measure_score(GT_list, Our_list)
print(f"Dengkw         V-measure: {Our_V*100:.6f}")
Our_homogeneity = homogeneity_score(GT_list, Our_list)
Our_completeness = completeness_score(GT_list, Our_list)
print(f"Dengkw         Homogeneity: {Our_homogeneity*100:.6f} Completeness: {Our_completeness*100:.6f}")
Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"Dengkw         (ARI): {Our_ari*100:.6f}")
Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"Dengkw         (FMI): {Our_fmi*100:.6f}")
print(f"{Our_nmi*100:.2f}&{Our_ami*100:.2f}&{Our_fmi*100:.2f}&{Our_ari*100:.2f}&{Our_V*100:.2f}&{Our_F_measure*100:.2f}&{Our_Jaccard*100:.2f}&{Our_completeness*100:.2f}")

# In[ ]:




