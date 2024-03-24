import torch
import torch.nn.functional as F
from torch.nn import Linear


class SimpleLinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()



"""
we want a decoder that conditions on the projected time for prediction
"""
class TimeProjDecoder(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 time_dim: int, 
                 hidden_channels: int, 
                 out_channels: int, 
                 num_layers: int,
                 dropout: float):
        r"""
        initialization for the time projected decoder
        takes input both node embeddings and the delta time
        for now just concatenate the cosine similarity of node embeddings and time embeddings
        Parameters:
            in_channels: int, the size of node embeddings
            time_dim: int, the size of time embeddings
            hidden_channels: int, the size of hidden layers
            out_channels: int, the size of output
            num_layers: int, the number of layers
            dropout: float, the dropout rate
        """
        super(TimeProjDecoder, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels+time_dim, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, time_embed):
        x = x_i * x_j
        x = torch.cat([x, time_embed], dim=-1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)







"""
this is a static link prediction which doesn't takes time into account
"""
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

