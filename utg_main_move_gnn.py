"""
design a basic encoder and decoder framework for ctdg for example
the encoder is a simple GNN
the decoder is a simple MLP

#! MLP wants to take in the catenation of node embed from different snapshots
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


def test_tgb(embeddings, 
             test_loader, 
             test_snapshots, 
             ts_list,
             node_feat,
             encoder, 
             decoder,
             neg_sampler,
             evaluator,
             metric, 
             split_mode='val', 
             context_size=2):
    encoder.eval()
    decoder.eval()

    perf_list = []
    ts_idx = min(list(ts_list.keys()))
    max_ts_idx = max(list(ts_list.keys()))


    for batch in test_loader:
        pos_src, pos_dst, pos_t, pos_msg = (
        batch.src,
        batch.dst,
        batch.t,
        batch.msg,
        )

        #! update the model now if the prediction batch has moved to next snapshot
        while (pos_t[0] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            with torch.no_grad():
                pos_index = test_snapshots[ts_idx]
                pos_index = pos_index.long().to(args.device)
                for k in range(context_size-1):
                    embeddings[k] = embeddings[k+1].detach() #* shift the index by 1
                embeddings[-1] = encoder(x=node_feat, edge_index=pos_index).detach()
            ts_idx += 1


        neg_batch_list = neg_sampler.query_batch(np.array(pos_src.cpu()), np.array(pos_dst.cpu()), np.array(pos_t.cpu()), split_mode=split_mode)
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = torch.full((1 + len(neg_batch),), pos_src[idx], device=args.device)
            query_dst = torch.tensor(
                        np.concatenate(
                            ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                            axis=0,
                        ),
                        device=args.device,
                    )
            with torch.no_grad():
                y_pred  = decoder(embeddings[query_src], embeddings[query_dst])
            y_pred = y_pred.squeeze(dim=-1).detach()

            input_dict = {
            "y_pred_pos": np.array([y_pred[0].cpu()]),
            "y_pred_neg": np.array(y_pred[1:].cpu()),
            "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

    #* update to the final snapshot
    with torch.no_grad():
        pos_index = test_snapshots[ts_idx]
        pos_index = pos_index.long().to(args.device)
        embeddings[-1] = encoder(x=node_feat, edge_index=pos_index).detach()
    test_metrics = float(np.mean(np.array(perf_list)))

    return test_metrics, embeddings






def run(args, data, seed=1):
    
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="utg",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "utg_gnn_mlp",
            "dataset": args.dataset,
            "time granularity": args.time_scale,
            }
        )


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
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]

    #* set up TGB queries, this is only for val and test
    metric = dataset.eval_metric
    neg_sampler = dataset.negative_sampler
    evaluator = Evaluator(name=args.dataset)
    min_dst_idx, max_dst_idx = int(full_data.dst.min()), int(full_data.dst.max())

    #! set up node features
    node_feat = dataset.node_feat
    if (node_feat is not None):
        node_feat = node_feat.to(args.device)
        num_feat = node_feat.size(1)
    else:
        num_feat = 256
        # node_feat = torch.ones((full_data.num_nodes,num_feat)).to(args.device)
        node_feat = torch.randn((full_data.num_nodes,num_feat)).to(args.device)


    context_size = 2
    encoder = GCN(in_channels=num_feat, hidden_channels=args.hidden_channels, out_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout).to(args.device)
   
    decoder = SimpleLinkPredictor(in_channels=int(args.hidden_channels*context_size)).to(args.device)

    optimizer = optim.Adam(set(encoder.parameters())|set(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    
    #set to training mode
    encoder.train()
    decoder.train()

    #! should iterate over batches of edges
    train_loader = TemporalDataLoader(train_data, batch_size=batch_size)

    best_epoch = 0
    best_val = 0
    best_test = 0

    for epoch in range(1, args.max_epoch + 1):
        start_epoch_train = timeit.default_timer()
        embeddings = [None] * context_size #place holder

        #define the processed graph snapshots
        train_snapshots = data['train_data']['edge_index']
        ts_list = data['train_data']['ts_map']
        ts_idx = min(list(ts_list.keys()))
        max_ts_idx = max(list(ts_list.keys()))

        #! start with the embedding from first snapshot, as it is required 
        pos_index = train_snapshots[0]
        pos_index = pos_index.long().to(args.device)
        for k in range(context_size-1):
            embeddings[k] = encoder(x=node_feat, edge_index=pos_index).detach() #populate the context embeddings with detached ones
        embeddings[-1] = encoder(x=node_feat, edge_index=pos_index)


        total_loss = 0
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

            pos_edges = torch.stack([pos_src, pos_dst], dim=0)
            neg_edges = torch.stack([pos_src, neg_dst], dim=0)

            #? organize the past context
            pos_src_embed_list = []
            pos_dst_embed_list = []
            neg_src_embed_list = []
            neg_dst_embed_list = []
            for k in range(context_size):
                pos_src_embed_list.append(embeddings[k][pos_edges[0]])
                pos_dst_embed_list.append(embeddings[k][pos_edges[1]])
                neg_src_embed_list.append(embeddings[k][neg_edges[0]])
                neg_dst_embed_list.append(embeddings[k][neg_edges[1]])
            
            pos_src_embed = torch.cat(pos_src_embed_list, dim=1)
            pos_dst_embed = torch.cat(pos_dst_embed_list, dim=1)
            neg_src_embed = torch.cat(neg_src_embed_list, dim=1)
            neg_dst_embed = torch.cat(neg_dst_embed_list, dim=1)


            pos_out = decoder(pos_src_embed, pos_dst_embed)
            neg_out = decoder(neg_src_embed, neg_dst_embed)

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            total_loss += float(loss) * batch.num_events

            #! due to time encoding is used in each batch, to train it correctly, backprop each batch
            loss.backward()
            optimizer.step()

            #? it is possible to cover multiple snapshots in a batch
            while (pos_t[0] > ts_list[ts_idx] and ts_idx < max_ts_idx):
                #* each time a new snapshot is seen, shift the matrix by one
                """
                update the embeddings in the following way
                [0,0,0]
                [0,0,1]
                [0,1,2]
                [1,2,3]
                ...
                """
                pos_index = train_snapshots[ts_idx]
                pos_index = pos_index.long().to(args.device)
                for k in range(context_size-1):
                    embeddings[k] = embeddings[k+1].detach() #* shift the index by 1
                embeddings[-1] = encoder(x=node_feat, edge_index=pos_index) 
                ts_idx += 1
            
            pos_index = train_snapshots[ts_idx]
            pos_index = pos_index.long().to(args.device)
            embeddings[-1] = encoder(x=node_feat, edge_index=pos_index) 

        train_time = timeit.default_timer() - start_epoch_train
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {train_time: .4f}")
        print ("training loss is ", total_loss / train_data.num_events)


        val_snapshots = data['val_data']['edge_index']
        ts_list = data['val_data']['ts_map']
        val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
        evaluator = Evaluator(name=args.dataset)
        neg_sampler = dataset.negative_sampler
        dataset.load_val_ns()

        start_epoch_val = timeit.default_timer()
        val_metrics, embeddings = test_tgb(embeddings, val_loader, val_snapshots, ts_list,
             node_feat,encoder, decoder,neg_sampler,evaluator,metric, split_mode='val', context_size=context_size)
        val_time = timeit.default_timer() - start_epoch_val

        print ("validation metrics is ", val_metrics)
        print ("val elapsed time is ", val_time)

        if (args.wandb):
            wandb.log({"train_loss":(total_loss / train_data.num_events),
                        "val_" + metric: val_metrics,
                        "train time": train_time,
                        "val time": val_time,
                        })
        
        if (val_metrics > best_val):
            dataset.load_test_ns()
            test_snapshots = data['test_data']['edge_index']
            ts_list = data['test_data']['ts_map']
            test_loader = TemporalDataLoader(test_data, batch_size=batch_size)
            neg_sampler = dataset.negative_sampler
            dataset.load_test_ns()

            test_start_time = timeit.default_timer()
            test_metrics, embeddings = test_tgb(embeddings, test_loader, test_snapshots, ts_list,
             node_feat,encoder, decoder,neg_sampler,evaluator,metric, split_mode='test', context_size=context_size)
            test_time = timeit.default_timer() - test_start_time
            best_val = val_metrics
            best_test = test_metrics

            print ("test metric is ", test_metrics)
            print ("test elapsed time is ", test_time)
            print ("--------------------------------")

            if ((epoch - best_epoch) >= args.patience and epoch > 1):
                best_epoch = epoch
                break
            best_epoch = epoch
    
    print ("run finishes")
    print ("best epoch is, ", best_epoch)
    print ("best val performance is, ", best_val)
    print ("best test performance is, ", best_test)







        









        




if __name__ == '__main__':
    from utils.configs import args
    from utils.data_util import loader


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

    

