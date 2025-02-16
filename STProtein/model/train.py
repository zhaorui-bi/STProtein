import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model import STProtein


def train_STProtein(adata, ground_truth , feature_key="feat", edge_key="edgeList", weights=None, n_epochs=600,
               lr=0.0001,weight_decay=1e-5,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):      
    x1, edge_index1 = torch.FloatTensor(adata.obsm[feature_key]), torch.LongTensor(adata.uns[edge_key])
    emb_dim = ground_truth.shape[1]
    target = ground_truth.to(device)

    model = STProtein(hidden_dims=[x1.shape[1], emb_dim])

    x1, edge_index1= x1.to(device), edge_index1.to(device)
    model.to(device)
    
    n_epochs = n_epochs
    loss_list = []
    w1, w2 = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()
        z, x1_rec= model(x1, edge_index1)
        loss = w1 * F.mse_loss(x1, x1_rec) + w2*F.mse_loss(target, z)
        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    
    model.eval()
    x1, edge_index1 = x1.to(device), edge_index1.to(device)
    z, x1_rec = model(x1, edge_index1)

    return z, model