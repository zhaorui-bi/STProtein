#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[20]:


import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.optim as optim
import sys
from scipy import stats
import scanpy as sc
interactive = True
# name = 'Mouse_Spleen'
name = 'Mouse_Thymus'
# name = 'Human_Lymph_Node'
# output_path = "Results/Mouse_Spleen/" 
output_path = "Results/Mouse_Thymus/" 
# output_path = "Results/Human_Lymph_Node/" 
path = '/data/hulei/ZhaoruiJiang/Data/STProtein/'
# train_data = sc.read_h5ad(path + "Human_Lymph_Node_A1.h5ad")
# test_data = sc.read_h5ad(path + "Human_Lymph_Node_D1.h5ad")
train_data = sc.read_h5ad(path + "Mouse_Thymus3.h5ad")
test_data = sc.read_h5ad(path + "Mouse_Thymus2.h5ad")
# train_data = sc.read_h5ad(path + "Mouse_Spleen1.h5ad")
# test_data = sc.read_h5ad(path + "Mouse_Spleen2.h5ad")
# train_data = sc.read("Dataset/adataset1.h5ad")
# test_data = sc.read("Dataset/adataset2.h5ad")
file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
ground = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')

# In[21]:


if train_data.obs.index.duplicated().any():
    train_data.obs.reset_index(drop=True, inplace=True)
if test_data.obs.index.duplicated().any():
    test_data.obs.reset_index(drop=True, inplace=True)
if train_data.var.index.duplicated().any():
    train_data.var.reset_index(drop=True, inplace=True)
if test_data.var.index.duplicated().any():
    test_data.var.reset_index(drop=True, inplace=True)


# In[22]:


genes_inter = train_data.var_names.intersection(test_data.var_names)
train_data = train_data[:,genes_inter].copy()
test_data = test_data[:,genes_inter].copy()


# In[23]:


X_file = train_data.to_df().T
y_file = train_data.obsm['protein_expression'].T


# In[24]:


header_list = ['']
loc = output_path+'model_optimal/'+ name +'_'


# In[25]:


repi = 0
n_batches = 32
protein_list = None 
gene_list = None
X_final = None
y_final = None
max_epochs=4


# In[27]:


x_batch2 = test_data.to_df().T
y_list_savePath = output_path + name +"_cTPnet.csv"


# In[28]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(14505, 1000)
        self.fc1 = nn.Linear(X.shape[1], 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.ModuleDict({})
        for p in protein_list:
            self.fc3[p] = nn.Linear(256, 64)
        self.fc4 = nn.ModuleDict({})
        for p in protein_list:
            self.fc4[p] = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = {}
        for p in protein_list:
            outputs[p] = self.fc4[p](F.relu(self.fc3[p](x)))
        return outputs


# In[29]:


X = X_file
y = y_file
if interactive:
    X=X.transpose().sample(frac=0.1,random_state=4905).transpose()
	# Dealing with X's dimensionality
gene = set(X.index.tolist())
if gene_list is None:
    gene_list=gene
else:
    gene_list=set(gene_list).intersection(gene)
    
gene_list=list(gene_list)
gene_list.sort()
X=X.loc[gene_list,]

if not X_final is None:
    X_final=X_final.loc[gene_list,]
	
    # Dealing with Y's dimensionality
protein = set(y.index)
if protein_list is None:
    protein_list = protein
else:
    protein_list=set(protein_list).intersection(protein)
protein_list = list(protein_list)   
	# print(protein)
	# y.index=protein
y = y[X.columns]
y = y.loc[protein_list,]
	# Add header to cell
X.columns = list(map(lambda x: header_list[0]+'-'+x, X.columns.tolist()))
y.columns = list(map(lambda x: header_list[0]+'-'+x, y.columns.tolist()))
    
X_final = X
y_final = y   


# In[30]:


# Normalize y
shared_proteins = y_final.apply(lambda x: not x.isna().any(),axis=1)
y = y_final.apply(lambda x: np.log((x+1.0)/stats.gmean(x[shared_proteins]+1.0)), axis=0)
del(y_final)
# Use direct UMI count is hard to transfer across experiment
# Let's try normalize the data with seurat like method
X=X_final.apply(lambda x: np.log((x*10000.0/sum(x))+1.0), axis=0)
del(X_final)
# random cell order
X=X.T
X=X.sample(frac=1,random_state=4905)
# separate test data from train data
if interactive:
	X_test=X.sample(frac=0.15,random_state=4905)# Need change after test
else:
	X_test=X.sample(frac=0.15,random_state=4905)# Need change after test

y_test=y[X_test.index]

X=X.drop(X_test.index)
y=y.drop(columns=y_test.columns)
y=y[X.index]


# In[31]:


# covert to tenor
X=torch.tensor(X.values)
X=X.type(torch.FloatTensor)
y=torch.tensor(y.values)
y=y.type(torch.FloatTensor)
y=torch.t(y)

X_test=torch.tensor(X_test.values)
X_test=X_test.type(torch.FloatTensor)
y_test=torch.tensor(y_test.values)
y_test=y_test.type(torch.FloatTensor)
y_test=torch.t(y_test)


# In[32]:


net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001,amsgrad=True, weight_decay=0.001)
# optimizer = optim.Adagrad(net.parameters(), lr=lr_vec[repi], lr_decay=0.001)


# In[33]:


train_loss=pd.DataFrame(np.zeros(shape=(len(protein_list),max_epochs)),index=protein_list)
test_loss=pd.DataFrame(np.zeros(shape=(len(protein_list),max_epochs)),index=protein_list)


# In[34]:


# Init early stop
patience=30
best_score=None
Dy=len(protein_list)
estop_counter=pd.Series(np.zeros(Dy),index=protein_list)
early_stop=pd.Series([False]*Dy,index=protein_list)


# In[35]:


for epoch in range(max_epochs):
	if all(early_stop):
		break
	running_loss=pd.Series(np.zeros(Dy),index=protein_list)
	for i in range(int(y.shape[0]/n_batches)):
		# Local batches and labels
		local_X, local_y = X[i*n_batches:min((i+1)*n_batches,X.shape[0]-1),], y[i*n_batches:min((i+1)*n_batches,y.shape[0]-1),]
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs_dict = net(local_X)
		loss=None
		loss_count=0.0
		for p in protein_list:
			notNaN=(local_y[:,protein_list.index(p):(protein_list.index(p)+1)]==local_y[:,protein_list.index(p):(protein_list.index(p)+1)])
			loss_p=criterion(outputs_dict[p][notNaN],local_y[:,protein_list.index(p):(protein_list.index(p)+1)][notNaN])
			if not torch.isnan(loss_p):
				loss_count+=1.0
				running_loss[p]+=loss_p.item()
				if loss is None:
					loss=loss_p
				else:
					loss=loss+loss_p
		loss.backward()
		optimizer.step()
		if(i==(int(y.shape[0]/n_batches)-1)):
			train_loss.iloc[:,epoch]=(running_loss / 150)
		if i % 150 == 149:    # print every mini-batches
			# print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, sum(running_loss / 150)))
			running_loss=pd.Series(np.zeros(Dy),index=protein_list)
			sys.stdout.flush()
	test_outputs = net(X_test)
	test_outputs = [test_outputs[p] for p in protein_list]
	test_outputs=torch.transpose(torch.stack(test_outputs),0,1).view(X_test.shape[0],-1)
	test_loss_i=pd.Series([criterion(test_outputs[:,pi][y_test[:,pi]==y_test[:,pi]], y_test[:,pi][y_test[:,pi]==y_test[:,pi]]).item() for pi in range(Dy)],index=protein_list)
	test_loss.iloc[:,epoch]=test_loss_i
	# if epoch % 10 == 9:
	# 	f,ax=plt.subplots(figsize=(6,6))
	# 	ax.scatter(y_test.detach().numpy(),test_outputs.detach().numpy())
	# 	ax.plot([-2,5],[-2,5],ls='--',c='.3')
	# 	# ax.text(3,-2,'correlation: '+str(np.corrcoef(test_outputs.detach().numpy().flatten(),y_test.detach().numpy().flatten())[1,0]))
	# 	df = pd.DataFrame({"y_pred":test_outputs.detach().numpy().flatten(),'y_truth':y_test.detach().numpy().flatten()})
	# 	ax.text(3,-2,'correlation: '+str(round(df.corr().values[1,0],4)))
	# 	fig = ax.get_figure()
	# 	fig.savefig(loc+'figure_rep'+str(repi)+'_ep'+str(epoch)+'.pdf')
	# 	sys.stdout.flush()
	# 	plt.close(fig)
	if epoch % 5 == 4:
		torch.save(net.state_dict(), loc+'model_rep'+str(repi)+'_ep'+str(epoch))
	# Implement early stopping
	if best_score is None:
		best_score=test_loss_i
	else:
		for p in protein_list:
			if test_loss_i[p]>(best_score[p]-0.001) and (not early_stop[p]):
				estop_counter[p]+=1
				if estop_counter[p]>=patience:
					early_stop[p]=True
			else:
				best_score[p]=test_loss_i[p]
				estop_counter[p]=0
		# print(estop_counter)

print('Finished Training')


# In[36]:


torch.save(net.state_dict(), loc+'model_rep'+str(repi)+'_ep'+str(epoch))

train_loss.index=['train_'+p for p in protein_list]
test_loss.index=['test_'+p for p in protein_list]
log=pd.concat([train_loss,test_loss])

log.to_csv(loc+'log_rep'+str(repi)+'.csv')


# In[37]:


X_pred_raw = x_batch2
# X_pred_raw = pd.read_csv(X_list_batch2, index_col=0)
X_pred = X_pred_raw.loc[gene_list,]
X_pred = X_pred.apply(lambda x: np.log((x*10000.0/sum(x))+1.0), axis=0).T
X_pred = torch.tensor(X_pred.values)
X_pred = X_pred.type(torch.FloatTensor)
test_outputs_ = net(X_pred)
def fun(x):
    test_outputs_[x] = test_outputs_[x].detach().numpy().flatten()
ing = [fun(i) for i in test_outputs_]
test_outputs_ = pd.DataFrame.from_dict(test_outputs_, orient='index')
test_outputs_.columns = X_pred_raw.columns
test_outputs_.to_csv(y_list_savePath)


# In[40]:


file_fold = '/data/hulei/ZhaoruiJiang/Data/SpatialGlue/'
# prediction = sc.read_h5ad(file_fold + 'Dataset2_Mouse_Spleen2/adata_ADT.h5ad')
prediction = sc.read_h5ad(file_fold + 'Dataset4_Mouse_Thymus2/adata_ADT.h5ad')
# prediction = sc.read_h5ad(file_fold + 'Dataset12_Human_Lymph_Node_D1/adata_ADT.h5ad')
prediction.var_names_make_unique()
prediction.X= test_outputs_.T.to_numpy()


# In[42]:


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


# In[43]:


clr_normalize_each_cell(prediction)
sc.pp.scale(prediction)


# In[44]:


prediction.obsm['embedding'] = prediction.to_df().to_numpy()


# In[45]:


import os
# the location of R, which is required for the 'mclust' algorithm. Please replace the path below with local R installation path
os.environ['R_HOME'] = '/opt/miniforge/envs/STAligner/lib/R'
# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
tool = 'mclust' # mclust, leiden, and louvain
from utils import clustering
# clustering(prediction, key='embedding', add_key='embedding', n_clusters=5, method=tool, use_pca=True)


# In[46]:


# visualization
# import matplotlib.pyplot as plt
# fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
# sc.pp.neighbors(prediction, use_rep='embedding', n_neighbors=10)
# sc.tl.umap(prediction)

# sc.pl.umap(prediction, color='embedding', ax=ax_list[0], title='cTP-net', s=20, show=False)
# sc.pl.embedding(prediction, basis='spatial', color='embedding', ax=ax_list[1], title='cTP-net', s=25, show=False)

# plt.tight_layout(w_pad=0.3)
# plt.show()


# In[47]:

# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + "adata_all_mouse_spleen_rep2.h5ad")
# ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_human_lymph_node_D1.h5ad')
ground_truth = sc.read_h5ad('/data/hulei/ZhaoruiJiang/Data/SpatialGlue/Ground-Truth/' + 'adata_all_mouse_thymus2.h5ad')

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

# In[49]:


label = prediction.obs[tool]
list = label.tolist()
path_2 = output_path + 'cTP-net_list'
with open(path_2, 'w') as f:
    for num in list:
        f.write(f"{num}\n")


# In[50]:


from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from metric import jaccard, Dice, F_measure
from sklearn.metrics import silhouette_score


# In[51]:


def read_list_from_file(path):
    list = []
    with open(path, 'r') as f:
        for line in f:
            num = int(line.strip())
            list.append(num)
    return list


# In[52]:


GT_list = read_list_from_file(path_1)
Our_list = read_list_from_file(path_2)


# In[53]:


Our_Jaccard = jaccard(Our_list, GT_list)
print(f"cTP-net         jaccard: {Our_Jaccard*100:.6f}")
Our_F_measure = F_measure(Our_list, GT_list)
print(f"cTP-net         F_measure: {Our_F_measure*100:.6f}")
Our_mutual_info = mutual_info_score(GT_list, Our_list)
print(f"cTP-net         Mutual Information: {Our_mutual_info*100:.6f}")
Our_nmi = normalized_mutual_info_score(GT_list, Our_list)
print(f"cTP-net         (NMI): {Our_nmi*100:.6f}")
Our_ami = adjusted_mutual_info_score(GT_list, Our_list)
print(f"cTP-net         (AMI): {Our_ami*100:.6f}")
Our_V = v_measure_score(GT_list, Our_list)
print(f"cTP-net         V-measure: {Our_V*100:.6f}")
Our_homogeneity = homogeneity_score(GT_list, Our_list)
Our_completeness = completeness_score(GT_list, Our_list)
print(f"cTP-net         Homogeneity: {Our_homogeneity*100:.6f} Completeness: {Our_completeness*100:.6f}")
Our_ari = adjusted_rand_score(GT_list, Our_list)
print(f"cTP-net         (ARI): {Our_ari*100:.6f}")
Our_fmi = fowlkes_mallows_score(GT_list, Our_list)
print(f"cTP-net         (FMI): {Our_fmi*100:.6f}")
print(f"{Our_nmi*100:.2f}&{Our_ami*100:.2f}&{Our_fmi*100:.2f}&{Our_ari*100:.2f}&{Our_V*100:.2f}&{Our_F_measure*100:.2f}&{Our_Jaccard*100:.2f}&{Our_completeness*100:.2f}")




