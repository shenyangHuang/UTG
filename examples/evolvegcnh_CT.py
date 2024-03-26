from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric_temporal.nn.recurrent import EvolveGCNH

import timeit 
import numpy as np
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import TemporalDataLoader
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def generate_random_negatives(pos_edges: torch.Tensor,
                              num_nodes: int, 
                              num_neg_samples: int =1):
    r"""
    generate random negative samples for training
    Parameters:
        pos_edges: positive edges
        num_nodes: number of nodes in the graph
        num_neg_samples: number of negative samples per positive edge
    Return:
        neg_edges: negative edges
    """
    neg_edges = negative_sampling(pos_edges, num_nodes=num_nodes, num_neg_samples=(pos_edges.size(1)*num_neg_samples))
    return neg_edges


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_count, node_features, hidden_dim):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class LinkPredictor(torch.nn.Module):
    """
    A Link Predictor
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=2, dropout=0.1):
        super(LinkPredictor, self).__init__()
        
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.linears.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.linears.append(torch.nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout
        
    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()

    def forward(self, z_src, z_dst):
        x = z_src * z_dst
        for layer in self.linears[: -1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return torch.sigmoid(x)


def test_step(data_loader, neg_sampler, evaluator, metric, split_mode, x=None):
    """
    DTDG inference step
    NOTE: Inference/Evaluation happens in a Continuous-Time manner
    """
    model.eval()
    decoder.eval()
    
    perf_list = []
    # edge_weight = None  # TODO: edge_features should be loaded!
    
    for pos_batch in data_loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=args.device,
            )
            edge_index = torch.stack((query_src, query_dst), dim=0)
            edge_weight = torch.ones((edge_index.shape[1], )).to(args.device)
            
            z = model(x=x, edge_index=edge_index, edge_weight=edge_weight)
            y_pred = decoder(z[edge_index[:, 0]], z[edge_index[:, 1]])
            
            input_dict = {
                "y_pred_pos": y_pred[0].detach().cpu().numpy(),
                "y_pred_neg": y_pred[1:].detach().cpu().numpy(),
                "eval_metric": [metric],
                }
            perf_list.append(evaluator.eval(input_dict)[metric])
            
    perf_metric = float(np.mean(perf_list))
    
    return perf_metric


def train_step(data, x=None):
    r"""
    DTDG train step 
    NOTE: training happens in a Discrete-Time manner
    """
    model.train()
    decoder.train()
    
    # edge_weight = None
    snapshot_list = data['edge_index']
    total_loss = 0 
    
    for snapshot_idx in range(data['time_length']):
        
        pos_index = snapshot_list[snapshot_idx]
        pos_index = pos_index.long().to(args.device)

        # generate random samples for training
        neg_index = generate_random_negatives(pos_index, num_nodes=args.num_nodes, num_neg_samples=1)
        edge_weight = torch.ones((pos_index.shape[1], )).to(args.device)
        
        # pass positive edge indices through the model
        z = model(x=x, edge_index=pos_index, edge_weight=edge_weight)  # TODO: `x` should take the current memory of the nodes when nodes don't have features
        
        pos_out = decoder(z[pos_index[:, 0]], z[pos_index[:, 1]])
        neg_out = decoder(z[neg_index[:, 0]], z[neg_index[:, 1]])
        
        total_loss += criterion(pos_out, torch.ones_like(pos_out))
        total_loss += criterion(neg_out, torch.zeros_like(neg_out))
        
    total_loss = total_loss / (snapshot_idx + 1)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss
            
               

if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    set_random(args.seed)
    
    # set initial features
    HID_DIM = EMB_DIM = args.nfeat  # default value for `nfeat`
    BATCH_SIZE = args.bs
    
    # Data Loadings --- TGB
    dataset_tgb = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
    train_mask = dataset_tgb.train_mask
    val_mask = dataset_tgb.val_mask
    test_mask = dataset_tgb.test_mask
    data_tgb = dataset_tgb.get_TemporalData()
    data_tgb = data_tgb.to(args.device)
    metric = dataset_tgb.eval_metric

    train_data_tgb = data_tgb[train_mask]
    val_data_tgb = data_tgb[val_mask]
    test_data_tgb = data_tgb[test_mask]

    # train_loader = TemporalDataLoader(train_data_tgb, batch_size=BATCH_SIZE)  # not needed
    val_loader = TemporalDataLoader(val_data_tgb, batch_size=BATCH_SIZE)
    test_loader = TemporalDataLoader(test_data_tgb, batch_size=BATCH_SIZE)
    
    neg_sampler = dataset_tgb.negative_sampler
    
    # set TGB evaluator
    evaluator = Evaluator(name=args.dataset)
    
    # loading data with innate loader
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data = data['train_data']
    # val_data = data['val_data']  # not needed
    # test_data = data['test_data']  # not needed
    args.num_nodes = int(1.1 * train_data['num_nodes'])  # allocate larger than training set to accomodate inductive nodes
    
    # x = torch.rand(args.num_nodes, args.nfeat).to(args.device)  # TODO: should load initial node features!
    x = torch.ones((args.num_nodes, args.nfeat)).to(args.device)
    
    # set the embedding model
    model = RecurrentGCN(node_count=args.num_nodes, node_features=args.nfeat, hidden_dim=HID_DIM).to(args.device)
    
    # set the decoder
    decoder = LinkPredictor(in_channels=EMB_DIM, hidden_channels=EMB_DIM, out_channels=1, 
                            num_layers=2, dropout=0.1).to(args.device)
    
    # set the optimizer
    optimizer = torch.optim.Adam(set(model.parameters()) | set(decoder.parameters()), lr=args.lr)
    
    # set the loss
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # -----------------
    # -----------------
    # Learning...
    
    for epoch in range(1, args.max_epoch + 1):
        # =======================
        # ======== Train ========
        # =======================
        epoch_start_time = timeit.default_timer()
        loss = train_step(train_data, x)
        print(f"Epoch: {epoch:02d}, Loss: {loss: .4f}, Training elapsed Time (s): {timeit.default_timer() - epoch_start_time: .4f}")
        
        # ============================
        # ======== Validation ========
        # ============================
        start_val = timeit.default_timer()
        dataset_tgb.load_val_ns()
        perf_metric_val = test_step(val_data_tgb, neg_sampler, evaluator, metric, split_mode="val", x=x)
        
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        
        # ============================
        # ======== Test ==============
        # ============================
        start_test = timeit.default_timer()
        dataset_tgb.load_test_ns()
        perf_metric_test = perf_metric_val = test_step(test_data_tgb, neg_sampler, evaluator, metric, split_mode="test", x=x)
        
        print(f"\tTest {metric}: {perf_metric_test: .4f}")
        print(f"\tTest: Elapsed time (s): {timeit.default_timer() - start_test: .4f}")
        
        print("-"*30)
        print(f"Epoch: {epoch:02d}, Total elapsed time (s): {timeit.default_timer() - epoch_start_time: .4f}")
        print("="*50)
    
    
    
    