import torch
import torch.nn.functional as F
from tqdm import tqdm
from .model import STProtein,Transfer_STProtein


def train_STProtein(adata, ground_truth, feature_key="feat", edge_key="edgeList", weights=None, n_epochs=600,
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
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        loss = w1 * F.mse_loss(x1, x1_rec) + w2*F.mse_loss(target, z)
        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    
    model.eval()
    x1, edge_index1 = x1.to(device), edge_index1.to(device)
    z, x1_rec= model(x1, edge_index1)

    return z, model


def train_Transfer_STProtein(pretrained_model, adata, target_emb=None, 
                             feature_key="feat", edge_key="edgeList", 
                             n_epochs=600, lr=0.0001, weight_decay=1e-5, 
                             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    x1 = torch.FloatTensor(adata.obsm[feature_key]).to(device)
    edge_index1 = torch.LongTensor(adata.uns[edge_key]).to(device)

    model = Transfer_STProtein(pretrained_model=pretrained_model, 
                              target_in_dim=x1.shape[1], 
                              target_emb=target_emb).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Transfer_STProtein"):
        model.train()
        optimizer.zero_grad()
        x_emb, x2, x_rec = model(x1, edge_index1)
        loss = F.mse_loss(x_rec, x1)
        loss_list.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_emb, x2, x_rec = model(x1, edge_index1)

    return x_emb, model, loss_list

  