

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import *
from torch_geometric.nn import GATConv,GCNConv,SAGEConv,GATv2Conv,GCN2Conv,TransformerConv

    
class SAGEConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # x = self.fc(x)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

class GCNConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(out_channels, out_channels, normalize=True)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # x = self.fc(x)
        return x


class GCNConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True)
        self.conv2 = GCNConv(in_channels, out_channels, normalize=True)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x
      
        
class GATv2Conv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATv2Conv_Encoder, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.conv1 = GATv2Conv(in_channels, out_channels,heads=5,concat=False)
        self.conv2 = GATv2Conv(out_channels, out_channels,heads=5,concat=False)
        # self.mu = nn.Linear(out_channels, in_channels)
        # self.log_var = nn.Linear(out_channels, in_channels)
        self.fc=nn.Linear(out_channels, out_channels)
      
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = x.tanh()
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # x = x.tanh()
        x = torch.relu(x)
        # x = self.fc(x)
        # mu = self.mu(x)
        # log_var = self.log_var(x)
        return x


class GATv2Conv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATv2Conv_Decoder, self).__init__()
        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.conv1 = GATv2Conv(in_channels, in_channels,heads=5,concat=False)
        self.conv2 = GATv2Conv(in_channels, out_channels,heads=5,concat=False)
        self.fc=nn.Linear(out_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = x.tanh()
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # x = x.tanh()
        x = torch.relu(x)
        x = self.fc(x)
        return x


class GATConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Decoder, self).__init__()
        self.conv1 = GATConv(in_channels, in_channels,heads=5,concat=False)
        self.conv2 = GATConv(in_channels, out_channels,heads=5,concat=False)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

class GATConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATConv_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels,heads=5,concat=False)
        self.conv2 = GATConv(out_channels, out_channels,heads=5,concat=False)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # x = self.fc(x)
        return x

class TransformerConv_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerConv_Decoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, in_channels,heads=5,concat=False)
        self.conv2 = TransformerConv(in_channels, out_channels,heads=5,concat=False)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.fc(x)
        return x

class TransformerConv_Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransformerConv_Encoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, out_channels,heads=5,concat=False)
        self.conv2 = TransformerConv(out_channels, out_channels,heads=5,concat=False)
        self.fc=nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # x = self.fc(x)
        return x



class STProtein(torch.nn.Module):
    def __init__(self, hidden_dims,Conv_Encoder=SAGEConv_Encoder,Conv_Decoder=SAGEConv_Decoder):
        super(STProtein, self).__init__()

        [in_dim1, out_dim] = hidden_dims
        self.conv1_enc = Conv_Encoder(in_dim1, out_dim)
        # self.fc=nn.Linear(out_dim, out_dim)
        self.conv1_dec = Conv_Decoder(out_dim, in_dim1)

    def forward(self, features1, edge_index1):
        x1 = self.conv1_enc(features1, edge_index1)
        # x =self.fc(x1)
        x1_rec = self.conv1_dec(x1, edge_index1)

        return x1, x1_rec



class Transfer_STProtein(nn.Module):
    def __init__(self, pretrained_model, target_in_dim, target_emb=None, 
                 Conv_Encoder=GATv2Conv_Encoder, Conv_Decoder=GATv2Conv_Decoder):
        super(Transfer_STProtein, self).__init__()
        self.conv1_enc = pretrained_model.conv1_enc
        hidden_dim = self.conv1_enc.out_channels  
        in_channels = self.conv1_enc.in_channels  
        self.adapter = nn.Linear(target_in_dim, in_channels) if target_in_dim != in_channels else None
        target_emb = hidden_dim if target_emb is None else target_emb
        self.fc1 = nn.Linear(hidden_dim, target_emb)
        self.fc2 = nn.Linear(target_emb, hidden_dim)
        self.conv1_dec = Conv_Decoder(in_channels=hidden_dim, out_channels=target_in_dim)

    def forward(self, x, edge_index):
        if self.adapter is not None:
            x = self.adapter(x).relu()
        x1 = self.conv1_enc(x, edge_index)
        x_emb = self.fc1(x1).relu()
        x2 = self.fc2(x_emb).relu()
        x_rec = self.conv1_dec(x2, edge_index)
        return x_emb, x2, x_rec