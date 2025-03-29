import scanpy as sc
import anndata as ad
import torch
import scarches as sca
import matplotlib.pyplot as plt
import numpy as np
import scvi as scv
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from scipy.io import mmread, mmwrite

# name = "Mouse_Spleen"
# name = "Human_Lymph_Node"
name = "Mouse_Thymus"
# output_path = "Results/Mouse_Spleen/
# output_path = "Results/Human_Lymph_Node/" 
output_path = "Results/Mouse_Thymus/"
path = '/data/hulei/ZhaoruiJiang/Data/STProtein/'
adata_ref = sc.read_h5ad(path + "Mouse_Thymus3.h5ad")
adata_query = sc.read_h5ad(path + "Mouse_Thymus2.h5ad")
# adata_ref  = sc.read_h5ad(path + "Human_Lymph_Node_A1.h5ad")
# adata_query =  sc.read_h5ad(path + "Human_Lymph_Node_D1.h5ad")
# adata_ref = sc.read_h5ad(path + "Mouse_Spleen1.h5ad")
# adata_query = sc.read_h5ad(path + "Mouse_Spleen2.h5ad")
# adata_ref = sc.read(train_data_path)
# adata_query = sc.read(test_data_path)
file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
ground = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')

if adata_ref.obs.index.duplicated().any():
    adata_ref .obs.reset_index(drop=True, inplace=True)
if adata_query.obs.index.duplicated().any():
    adata_query.obs.reset_index(drop=True, inplace=True)
if adata_ref .var.index.duplicated().any():
    adata_ref .var.reset_index(drop=True, inplace=True)
if adata_query.var.index.duplicated().any():
    adata_query.var.reset_index(drop=True, inplace=True)


# In[7]:


genes_inter = adata_ref.var_names.intersection(adata_query.var_names)
adata_ref = adata_ref[:,genes_inter].copy()
adata_query = adata_query[:,genes_inter].copy()


# In[8]:


adata_ref.obs["batch"] = "train"
adata_query.obs["batch"] = "test"


# In[9]:


# put matrix of zeros for protein expression (considered missing)
pro_exp = adata_ref.obsm["protein_expression"]
data = np.zeros((adata_query.n_obs, pro_exp.shape[1]))
adata_query.obsm["protein_expression"] = pd.DataFrame(columns=pro_exp.columns, index=adata_query.obs_names, data = data)


# In[10]:


adata_full = ad.concat([adata_ref, adata_query])

batch = ['train']*adata_ref.shape[0]+['test']*adata_query.shape[0]


# In[11]:


sc.pp.highly_variable_genes(
    adata_ref,
    n_top_genes=4000,
    flavor="seurat_v3",
    batch_key="batch",
    subset=True,
)


# In[12]:


adata_query = adata_query[:, adata_ref.var_names].copy()


# In[13]:


sca.models.TOTALVI.setup_anndata(
    adata_ref,
    batch_key="batch",
    protein_expression_obsm_key="protein_expression"
)


# In[14]:


arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
)


# In[15]:


vae_ref = sca.models.TOTALVI(
    adata_ref, 
    **arches_params
)


# In[16]:


vae_ref.train()
# vae_ref.train(lr=1e-4)


# In[17]:


# vae_ref.get_latent_representation()


# In[18]:


adata_ref.obsm["X_totalVI"] = vae_ref.get_latent_representation()
# adata_ref.obsm["X_totalVI"] = torch.load("/data/hulei/ZhaoruiJiang/STProtein/Embedding/emb_SpatialGlue.pth").numpy()
sc.pp.neighbors(adata_ref, use_rep="X_totalVI")
sc.tl.umap(adata_ref, min_dist=0.4)


# In[19]:


dir_path = "Results/scArches/Twobatches/saved_model/"+"dataset"+" to "+name+"/"
vae_ref.save(dir_path, overwrite=True)


# In[20]:


vae_q = sca.models.TOTALVI.load_query_data(
    adata_query, 
    dir_path, 
    freeze_expression=True
)


# In[21]:


vae_q.train(3)


# In[22]:


adata_query.obsm["X_totalVI"] = vae_q.get_latent_representation()
sc.pp.neighbors(adata_query, use_rep="X_totalVI")
sc.tl.umap(adata_query, min_dist=0.4)


# In[23]:


_, imputed_proteins = vae_q.get_normalized_expression(
    adata_query,
    n_samples=25,
    include_protein_background=True,
    sample_protein_mixing=False,
    return_mean=True,
    transform_batch=["test"],
)


# In[24]:


adata_query.obs = pd.concat([adata_query.obs, imputed_proteins], axis=1)
imputed_proteins.to_csv(output_path+ name +"_scArches.csv")


# In[25]:


file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# prediction = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_ADT.h5ad')
prediction = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# prediction = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_ADT.h5ad')
prediction.var_names_make_unique()
prediction.X= imputed_proteins.to_numpy()


# In[27]:


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
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )


# In[28]:


clr_normalize_each_cell(prediction)
sc.pp.scale(prediction)


# In[29]:


prediction.obsm['embedding'] = prediction.to_df().to_numpy()


# In[42]:


import os
# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/opt/miniforge/envs/STAligner/lib/R'
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
tool = 'mclust' # mclust, leiden, and louvain
from utils import clustering
# clustering(prediction, key='embedding', add_key='embedding', n_clusters=6, method=tool, use_pca=True)


# In[43]:


# visualization
# import matplotlib.pyplot as plt
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(prediction, use_rep='embedding', n_neighbors=10)
# sc.tl.umap(prediction)

# sc.pl.umap(prediction, color='embedding', ax=ax_list[0], title='scArches', s=20, show=False)
# sc.pl.embedding(prediction, basis='spatial', color='embedding', ax=ax_list[1], title='scArches', s=25, show=False)

# plt.tight_layout(w_pad=0.3)
# plt.show()


# In[44]:


# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + "adata_all_mouse_spleen_rep2.h5ad")
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_human_lymph_node_D1.h5ad')
ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_mouse_thymus2.h5ad')
# In[ ]:

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


# In[46]:


label = prediction.obs[tool]
list = label.tolist()
path_2 = output_path + 'scArches_list'
with open(path_2, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


# In[47]:


from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score


# In[48]:


def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)
    return list


# In[49]:


GT_list = read_list_from_file(path_1)
Our_list = read_list_from_file(path_2)


# In[50]:


Our_Jaccard = jaccard(Our_list, GT_list)
print(f"scArches         jaccard: {Our_Jaccard*100:.6f}")
Our_F_measure = F_measure(Our_list, GT_list)
print(f"scArches         F_measure: {Our_F_measure*100:.6f}")
Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"scArches         Mutual Information: {Our_mutual_info*100:.6f}")
Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"scArches         (NMI): {Our_nmi*100:.6f}")
Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"scArches         (AMI): {Our_ami*100:.6f}")
Our_V = v_measure_score(GT_list, Our_list)
print(f"scArches         V-measure: {Our_V*100:.6f}")
Our_homogeneity = homogeneity_score(GT_list, Our_list)
Our_completeness = completeness_score(GT_list, Our_list)
print(f"scArches         Homogeneity: {Our_homogeneity*100:.6f} Completeness: {Our_completeness*100:.6f}")
Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"scArches         (ARI): {Our_ari*100:.6f}")
Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"scArches         (FMI): {Our_fmi*100:.6f}")

# In[ ]:


print(f"{Our_nmi*100:.2f}&{Our_ami*100:.2f}&{Our_fmi*100:.2f}&{Our_ari*100:.2f}&{Our_V*100:.2f}&{Our_F_measure*100:.2f}&{Our_Jaccard*100:.2f}&{Our_completeness*100:.2f}")

