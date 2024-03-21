from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from torch_geometric_temporal.signal import temporal_signal_split

import timeit 
import numpy as np
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse
from torch_geometric.utils import negative_sampling
from tgb.linkproppred.dataset import LinkPropPredDataset
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
    def __init__(self, node_count, node_features, hid_dim):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, hid_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    
class LinkPredictor(torch.nn.Module):
    """
    A Link Predictor
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


        
        
def test_step(full_data, masks, neg_sampler, evaluator, metric, split_mode, x=None):
    """
    DTDG inference step
    """
    model.eval()
    decoder.eval()
    
    perf_list = np.zeros(full_data['sources'][masks].shape[0])
    perf_idx = 0
    edge_weight = None  # TODO: edge_features should be loaded!
    
    for pos_src, pos_dst, pos_t in zip(full_data['sources'][masks],
                                       full_data['destinations'][masks],
                                       full_data['timestamps'][masks]):
        
        neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), 
                                                 np.array([pos_dst]), 
                                                 np.array([pos_t]), 
                                                 split_mode=split_mode)
        
        for neg_idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
            query_src = torch.from_numpy(query_src).long().to(args.device)
            query_dst = torch.from_numpy(query_dst).long().to(args.device)
            edge_index = torch.stack((query_src, query_dst), dim=0)
            
            z = model(x=x, edge_index=edge_index, edge_weight=edge_weight)
            y_pred = decoder(z[edge_index[:, 0]], z[edge_index[:, 1]])
            
            input_dict = {
                "y_pred_pos": y_pred[0].detach().cpu().numpy(),
                "y_pred_neg": y_pred[1:].detach().cpu().numpy(),
                "eval_metric": [metric],
                }
            perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
            perf_idx += 1
            
    perf_metric = float(np.mean(perf_list))
    
    return perf_metric


def train_step(data, x=None):
    r"""
    DTDG train step 
    """
    model.train()
    decoder.train()
    
    edge_weight = None
    snapshot_list = data['edge_index']
    total_loss = 0 
    
    for snapshot_idx in range(data['time_length']):
        
        pos_index = snapshot_list[snapshot_idx]
        pos_index = pos_index.long().to(args.device)

        # generate random samples for training
        neg_index = generate_random_negatives(pos_index, num_nodes=args.num_nodes, num_neg_samples=1)
        
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
    
    # Data Loadings --- TGB
    dataset_tgb = LinkPropPredDataset(name=args.dataset)
    full_data = dataset_tgb.full_data  
    metric = dataset_tgb.eval_metric
    neg_sampler = dataset_tgb.negative_sampler
    val_mask = dataset_tgb.val_mask
    test_mask = dataset_tgb.test_mask
    # set TGB evaluator
    evaluator = Evaluator(name=args.dataset)
    
    # loading with innate loader
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    args.num_nodes = int(1.1 * train_data['num_nodes'])  # allocate larger than training set to accomodate inductive nodes
    
    x = torch.rand(args.num_nodes, args.nfeat).to(args.device)  # TODO: should load initial node features!
    
    # set the embedding model
    model = RecurrentGCN(node_count=args.num_nodes, node_features=args.nfeat, hid_dim=HID_DIM).to(args.device)
    
    # set the decoder
    decoder = LinkPredictor(in_channels=EMB_DIM).to(args.device)
    
    # set optimizer
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
        perf_metric_val = test_step(full_data, val_mask, neg_sampler, evaluator, 
                                    metric, split_mode="val", x=x)
        
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        
        # ============================
        # ======== Test ==============
        # ============================
        start_test = timeit.default_timer()
        dataset_tgb.load_test_ns()
        perf_metric_val = test_step(full_data, test_mask, neg_sampler, evaluator, 
                                    metric, split_mode="test", x=x)
        
        print(f"\tTest {metric}: {perf_metric_val: .4f}")
        print(f"\tTest: Elapsed time (s): {timeit.default_timer() - start_test: .4f}")
        
        print("-"*30)
        print(f"Epoch: {epoch:02d}, Total elapsed time (s): {timeit.default_timer() - epoch_start_time: .4f}")
        print("="*50)
    
    
    
    