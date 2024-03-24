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
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from torch_geometric.loader import TemporalDataLoader
import torch.optim as optim
import wandb




# internal import
from utils.utils_func import set_random
from models.gnn_arch import GCN
from models.decoders import TimeProjDecoder, SimpleLinkPredictor
from models.tgn.time_enc import TimeEncoder


def test(args, data):
    print ("hi")






def run(args, data, seed=1):
    #data stores the discretized snapshots


    set_random(seed)
    batch_size = 200

    #ctdg dataset
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
    full_data = dataset.get_TemporalData()
    full_data = full_data.to(args.device)
    #get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    
    #define the processed graph snapshots
    train_snapshots = data['train_data']['edge_index']
    train_ts = data['train_data']['ts_map']
    ts_idx = 0

    train_data = full_data[train_mask]


    #! set up node features
    node_feat = dataset.node_feat
    if (node_feat is not None):
        node_feat = node_feat.to(args.device)
        num_feat = node_feat.size(1)
    else:
        num_feat = 256
        # node_feat = torch.ones((full_data.num_nodes,num_feat)).to(args.device)
        node_feat = torch.randn((full_data.num_nodes,num_feat)).to(args.device)




    #* set up TGB queries, this is only for val and test
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max())


    time_encoder = TimeEncoder(out_channels=args.time_dim).to(args.device)
    encoder = GCN(in_channels=num_feat, hidden_channels=args.hidden_channels, out_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout).to(args.device)
    # decoder = TimeProjDecoder(in_channels=args.hidden_channels, time_dim=args.time_dim, hidden_channels=args.hidden_channels, out_channels=1, num_layers=args.num_layers, dropout=args.dropout).to(args.device)
    decoder = SimpleLinkPredictor(in_channels=args.hidden_channels).to(args.device)
    # optimizer = optim.Adam(set(time_encoder.parameters())|set(encoder.parameters())|set(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam(set(encoder.parameters())|set(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()
    
    #set to training mode
    time_encoder.train()
    encoder.train()
    decoder.train()



    for epoch in range(1, args.max_epoch + 1):
        start_epoch_train = timeit.default_timer()
        embeddings = None

        #! should iterate over batches of edges
        train_loader = TemporalDataLoader(train_data, batch_size=batch_size)

        total_loss = 0

        #! start with the embedding from first snapshot, as it is required 
        pos_index = train_snapshots[0]
        pos_index = pos_index.long().to(args.device)
        embeddings = encoder(pos_index, x=node_feat) 
        max_idx = len(train_ts)-1


        for batch in train_loader:
            optimizer.zero_grad()
            pos_src, pos_dst, pos_t, pos_msg = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            )
            #* get the negative samples
            # Sample negative destination nodes.
            neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (pos_src.size(0),),
                dtype=torch.long,
                device=args.device,
            )

            # time_embed = time_encoder(pos_t.float().to(args.device))
            pos_edges = torch.stack([pos_src, pos_dst], dim=0)
            neg_edges = torch.stack([pos_src, neg_dst], dim=0)

            # pos_out = decoder(embeddings[pos_edges[0]], embeddings[pos_edges[1]], time_embed)
            # neg_out = decoder(embeddings[neg_edges[0]], embeddings[neg_edges[1]], time_embed)
            pos_out = decoder(embeddings[pos_edges[0]], embeddings[pos_edges[1]])
            neg_out = decoder(embeddings[neg_edges[0]], embeddings[neg_edges[1]])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            total_loss += float(loss) * batch.num_events

            #! due to time encoding is used in each batch, to train it correctly, backprop each batch
            loss.backward()
            optimizer.step()

            if ((ts_idx < max_idx) and (pos_t[0] > train_ts[ts_idx+1])):
                ts_idx += 1
            
            pos_index = train_snapshots[ts_idx]
            pos_index = pos_index.long().to(args.device)
            embeddings = encoder(pos_index, x=node_feat) 

        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}")
        print ("training loss is ", total_loss / train_data.num_events)




if __name__ == '__main__':
    from utils.configs import args
    from utils.data_util import loader, prepare_dir


    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    args.time_dim = 32
    args.hidden_channels = 128
    args.num_layers = 2

    for seed in range(args.seed, args.seed + args.num_runs):
        print ("--------------------------------")
        print ("excuting run with seed ", seed)
        run(args, data, seed=args.seed)
        print ("--------------------------------")

    

