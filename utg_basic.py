"""
design a basic encoder and decoder framework for ctdg for example
the encoder is a simple GNN
the decoder is a simple MLP
"""
import os
import sys
import timeit
import torch
import numpy as np
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from torch_geometric.loader import TemporalDataLoader
import torch.optim as optim
import wandb




# internal import
from utils.utils_func import set_random
from models.gnn_arch import GCN
from models.decoders import TimeProjDecoder
from models.tgn.time_enc import TimeEncoder


def run(args, data, seed=1):


    set_random(seed)
    batch_size = 50
    
    time_encoder = TimeEncoder(out_channels=args.time_dim)
    encoder = GCN(in_channels=args.num_feat, hidden_channels=args.hidden_channels, out_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout)
    decoder = TimeProjDecoder(in_channels=args.hidden_channels, time_dim=args.time_dim, hidden_channels=args.hidden_channels, out_channels=1, num_layers=args.num_layers, dropout=args.dropout)

    optimizer = optim.Adam(set(time_encoder.parameters())|set(encoder.parameters())|set(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    #set to training mode
    time_encoder.train()
    encoder.train()
    decoder.train()


    #! set up node features
    node_feat = None
    



    #* set up TGB queries, this is only for val and test
    dataset = LinkPropPredDataset(name=args.dataset)
    full_data = dataset.full_data  
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max())


    #get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    

    #define the processed graph snapshots
    train_snapshots = data['train_data']['edge_index']
    train_ts = data['train_data']['ts_map']
    ts_idx = 0

    for epoch in range(1, args.max_epoch + 1):
        epoch_start_time = timeit.default_timer()
        train_src = full_data['sources'][train_mask]
        train_dst = full_data['destinations'][train_mask]
        train_time = full_data['timestamps'][train_mask]


        start = True
        embeddings = None

        #* think about the training loop, the accumulates edges and make predictions, backprop when a snapshot is processed. 
        optimizer.zero_grad()

        #! should iterate over batches of edges
        train_loader = TemporalDataLoader(data[train_mask], batch_size=batch_size)

        loss = 0


        for batch in train_loader:
            pos_src, pos_dst, pos_t = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            )

            if (start):
                #! start with the embedding from first snapshot, as it is required 
                pos_index = train_snapshots[0]
                pos_index = pos_index.long().to(args.device)
                embeddings = encoder(pos_index, x=None) 
                start = False

            #* get the negative samples
            # Sample negative destination nodes.
            neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (pos_src.size(0),),
                dtype=torch.long,
                device=args.device,
            )

            time_embed = time_encoder(pos_t)
            pos_edges = torch.stack([pos_src, pos_dst], dim=0)
            neg_edges = torch.stack([pos_src, neg_dst], dim=0)

            pos_out = decoder(embeddings[pos_edges[0]], embeddings[pos_edges[1]], time_embed)
            neg_out = decoder(embeddings[neg_edges[0]], embeddings[neg_edges[1]], time_embed)

            loss += criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            if (pos_t[0] > train_ts[ts_idx]):
                loss.backward()
                optimizer.step()
                pos_index = train_snapshots[ts_idx]
                pos_index = pos_index.long().to(args.device)
                embeddings = encoder(pos_index, x=None) 
                ts_idx += 1
                optimizer.zero_grad()
            

if __name__ == '__main__':
    from utils.configs import args
    from utils.data_util import loader, prepare_dir


    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    args.time_dim = 32
    args.hidden_channels = 128
    args.num_layers = 2
    args.num_feat = args.nfeat #128

    for seed in range(args.seed, args.seed + args.num_runs):
        print ("--------------------------------")
        print ("excuting run with seed ", seed)
        run(args, data, seed=args.seed)
        print ("--------------------------------")

    

