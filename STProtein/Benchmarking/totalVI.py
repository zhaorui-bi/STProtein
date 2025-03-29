#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotnine as p9
import anndata as ad
import scanpy as sc
import scvi
# name = "Mouse_Spleen"
# name = "Human_Lymph_Node"
name = "Mouse_Thymus"
# output_path = "Results/Mouse_Spleen/" #path to results
# output_path = "Results/Human_Lymph_Node/" 
output_path = "Results/Mouse_Thymus/" #path to results


# In[2]:

file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# adata_batch1 = sc.read_h5ad("Dataset/adataset1.h5ad")
# adata_batch2 = sc.read_h5ad("Dataset/adataset2.h5ad")
path = '/data/hulei/ZhaoruiJiang/Data/STProtein/'
# adata_batch1 = sc.read_h5ad(path + "Human_Lymph_Node_A1.h5ad")
# adata_batch2 = sc.read_h5ad(path + "Human_Lymph_Node_D1.h5ad")
adata_batch1 = sc.read_h5ad(path + "Mouse_Thymus3.h5ad")
adata_batch2 = sc.read_h5ad(path + "Mouse_Thymus2.h5ad")
# adata_batch1 = sc.read_h5ad(path + "Mouse_Spleen1.h5ad")
# adata_batch2 = sc.read_h5ad(path + "Mouse_Spleen2.h5ad")
# adata_batch1 = sc.read_h5ad(train_data_path)
# adata_batch2 = sc.read_h5ad(test_data_path)

ground = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# In[3]:


# adata_batch1.obsm['protein_expression']


# In[4]:


# adata_batch2.obsm['protein_expression']


# In[5]:


# 假设 adata_batch1 和 adata_batch2 是你的 AnnData 对象, 检查并处理重复索引
if adata_batch1.obs.index.duplicated().any():
    adata_batch1.obs.reset_index(drop=True, inplace=True)
if adata_batch2.obs.index.duplicated().any():
    adata_batch2.obs.reset_index(drop=True, inplace=True)
if adata_batch1.var.index.duplicated().any():
    adata_batch1.var.reset_index(drop=True, inplace=True)
if adata_batch2.var.index.duplicated().any():
    adata_batch2.var.reset_index(drop=True, inplace=True)


# In[6]:


batch_idx = 1
adata = ad.concat([adata_batch1, adata_batch2],axis=0)
adata.obs_keys


# 

# In[7]:


# adata.to_df()


# In[8]:


modality = ['batch1']*adata_batch1.shape[0]+['batch2']*adata_batch2.shape[0]
adata.obs['batch']=modality
adata.obs['batch']


# In[9]:


batch = modality


# In[10]:


batch_set = ['batch1', 'batch2']
hold_out_batch = batch_set[batch_idx]
held_out_proteins = adata.obsm["protein_expression"][[batch[i] == hold_out_batch for i in range(len(batch))]].copy()
adata.obsm["protein_expression"].loc[[batch[i] == hold_out_batch for i in range(len(batch))]] = np.zeros_like(adata.obsm["protein_expression"][[batch[i] == hold_out_batch for i in range(len(batch))]])


# In[11]:


sc.pp.highly_variable_genes(
    adata,
    batch_key="batch",
    flavor="seurat_v3",
    n_top_genes=4000,
    subset=True
)


# In[12]:


# adata.to_df()


# In[13]:


scvi.model.TOTALVI.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key="protein_expression")

model = scvi.model.TOTALVI(
    adata,
    latent_distribution="normal",
    n_layers_decoder=1,
    n_layers_encoder=1
)
# model.train(lr=1e-3)
model.train(lr=1e-3)


# In[14]:


adata.obsm["X_totalVI"] = model.get_latent_representation()
adata.obsm["protein_fg_prob"] = model.get_protein_foreground_probability(transform_batch=batch_set[int(batch_idx==0)])


# In[15]:


_, protein_means = model.get_normalized_expression(
    n_samples=2,
    transform_batch=batch_set[int(batch_idx==0)],
    include_protein_background=True,
    sample_protein_mixing=True,
    return_mean=True,
)


# In[16]:


protein = protein_means.loc[[batch[i]==hold_out_batch for i in range(len(batch))]]#path to results
protein.to_csv(output_path+ name +"_totalVI.csv")


# In[17]:


protein


# In[18]:


protein.to_numpy().shape


# In[95]:


file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# prediction = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_ADT.h5ad')
prediction = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# prediction = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_ADT.h5ad')

prediction.var_names_make_unique()


# In[96]:


prediction.X= protein.to_numpy()


# In[97]:


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


# In[132]:


clr_normalize_each_cell(prediction)
# sc.pp.scale(prediction)


# In[133]:


prediction.obsm['embedding'] = prediction.to_df().to_numpy()


# In[134]:


import os
# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/opt/miniforge/envs/STAligner/lib/R'
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
tool = 'mclust' # mclust, leiden, and louvain
# stp.pl.utils.clustering(true_adata, key='embedding', add_key='embedding', n_clusters=5, method=tool, use_pca=True)


# In[135]:


from utils import clustering


# In[136]:


# visualization
# import matplotlib.pyplot as plt
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(prediction, use_rep='embedding', n_neighbors=10)
# sc.tl.umap(prediction)

# sc.pl.umap(prediction, color='embedding', ax=ax_list[0], title='totalVI', s=20, show=False)
# sc.pl.embedding(prediction, basis='spatial', color='embedding', ax=ax_list[1], title='totalVI', s=25, show=False)

# plt.tight_layout(w_pad=0.3)
# plt.show()


# In[104]:




# In[105]:


# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + "adata_all_mouse_spleen_rep2.h5ad")
ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + "adata_all_mouse_thymus2.h5ad")
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_human_lymph_node_D1.h5ad')

# In[106]:


ground_truth.obs['Pro_label_origi']


# In[107]:
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
path_1 = output_path+'_GT_list'
with open(path_1, 'w') as f:
    for num in list:
        f.write(f"{num}\n")

# In[137]:


label = prediction.obs[tool]
list = label.tolist()
path_2 = output_path + 'totalVI_list'
with open(path_2, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


# In[109]:


from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score


# In[110]:


def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)
    return list


# In[127]:


GT_list = read_list_from_file(path_1)
Our_list = read_list_from_file(path_2)


# In[138]:


Our_Jaccard = jaccard(Our_list, GT_list)
print(f"totalVI         jaccard: {Our_Jaccard*100:.6f}")
Our_F_measure = F_measure(Our_list, GT_list)
print(f"totalVI         F_measure: {Our_F_measure*100:.6f}")
Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"totalVI         Mutual Information: {Our_mutual_info*100:.6f}")
Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"totalVI         (NMI): {Our_nmi*100:.6f}")
Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"totalVI         (AMI): {Our_ami*100:.6f}")
Our_V = v_measure_score(GT_list, Our_list)
print(f"totalVI         V-measure: {Our_V*100:.6f}")
Our_homogeneity = homogeneity_score(GT_list, Our_list)
Our_completeness = completeness_score(GT_list, Our_list)
print(f"totalVI         Homogeneity: {Our_homogeneity*100:.6f} Completeness: {Our_completeness*100:.6f}")
Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"totalVI         (ARI): {Our_ari*100:.6f}")
Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"totalVI         (FMI): {Our_fmi*100:.6f}")
print(f"{Our_nmi*100:.2f}&{Our_ami*100:.2f}&{Our_fmi*100:.2f}&{Our_ari*100:.2f}&{Our_V*100:.2f}&{Our_F_measure*100:.2f}&{Our_Jaccard*100:.2f}&{Our_completeness*100:.2f}")




