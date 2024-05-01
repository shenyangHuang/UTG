"""
design a basic encoder and decoder framework for ctdg for example
python utg_main_gnn.py --dataset=tgbl-wiki -t hourly --lr 0.001 --max_epoch 200 --seed 1 --num_runs 1 --patience 20
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
             split_mode='val'):
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
        
        #* update the model at the end of the batch to see if a snapshot has been passed
        while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
            with torch.no_grad():
                pos_index = test_snapshots[ts_idx]
                pos_index = pos_index.long().to(args.device)
                embeddings = encoder(x=node_feat, edge_index=pos_index) 
                embeddings = embeddings.detach()
            ts_idx += 1

    #* update to the final snapshot
    with torch.no_grad():
        pos_index = test_snapshots[ts_idx]
        pos_index = pos_index.long().to(args.device)
        embeddings = encoder(x=node_feat, edge_index=pos_index) 
        embeddings = embeddings.detach()

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
            "architecture": "utg_gcn_mlp",
            "dataset": args.dataset,
            "time granularity": args.time_scale,
            }
        )


    set_random(seed)
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1

    num_feat = 256
    node_feat = torch.randn((num_nodes,num_feat)).to(args.device)

    encoder = GCN(in_channels=num_feat, hidden_channels=args.hidden_channels, out_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout).to(args.device)
    decoder = SimpleLinkPredictor(in_channels=args.hidden_channels).to(args.device)
    optimizer = optim.Adam(set(encoder.parameters())|set(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    best_epoch = 0
    best_val = 0
    best_test = 0    

    for epoch in range(1, args.max_epoch + 1):
        start_epoch_train = timeit.default_timer()
        embeddings = None
        #set to training mode
        encoder.train()
        decoder.train()
        optimizer.zero_grad()
        total_loss = 0


        #define the processed graph snapshots
        train_snapshots = data['train_data']['edge_index']

        for snapshot_idx in range(train_data['time_length']):
            optimizer.zero_grad()
            if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                cur_index = train_snapshots[snapshot_idx]
                cur_index = cur_index.long().to(args.device)
                embeddings = encoder(x=node_feat, edge_index=cur_index) 
            else:
                prev_index = train_snapshots[snapshot_idx-1]
                prev_index = prev_index.long().to(args.device)
                embeddings = encoder(x=node_feat, edge_index=prev_index)
            
            pos_index = train_snapshots[snapshot_idx]
            pos_index = pos_index.long().to(args.device)

            neg_dst = torch.randint(
                0,
                num_nodes,
                (pos_index.shape[1],),
                dtype=torch.long,
                device=args.device,
            )

            pos_out = decoder(embeddings[pos_index[0]], embeddings[pos_index[1]])
            neg_out = decoder(embeddings[pos_index[0]], embeddings[neg_dst])

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            optimizer.step()

            total_loss += (float(loss) / pos_index.shape[1])
        
        embeddings = embeddings.detach()


        train_time = timeit.default_timer() - start_epoch_train
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {train_time: .4f}")
        print ("training loss is ", total_loss)

        val_start_time = timeit.default_timer()
        encoder.eval()
        decoder.eval()
        evaluator = Evaluator(name="tgbl-wiki") #reuse MRR evaluator from TGB
        metric = "mrr"
        neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

        #* load the val negative samples
        neg_sampler.load_eval_set(fname=args.dataset + "_val_ns.pkl", split_mode="val")
        val_snapshots = val_data['edge_index'] #converted to undirected, also removes self loops as required by HTGN
        val_edges = val_data['original_edges'] #original edges unmodified
        ts_min = min(val_snapshots.keys())

        perf_list = {}
        perf_idx = 0

        for snapshot_idx in val_snapshots.keys():
            pos_index = torch.from_numpy(val_edges[snapshot_idx]) # (2,-1)
            pos_index = pos_index.long().to(args.device)

            for i in range(pos_index.shape[1]):
                pos_src = pos_index[0][i].item()
                pos_dst = pos_index[1][i].item()
                pos_t = snapshot_idx
                neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode='val')

                for idx, neg_batch in enumerate(neg_batch_list):
                    query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                    query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                    query_src = torch.from_numpy(query_src).long().to(args.device)
                    query_dst = torch.from_numpy(query_dst).long().to(args.device)
                    edge_index = torch.stack((query_src, query_dst), dim=0)
                    with torch.no_grad():
                        y_pred = decoder(embeddings[edge_index[0]], embeddings[edge_index[1]])
                    y_pred = y_pred.reshape(-1)
                    y_pred = y_pred.detach().cpu().numpy()

                    input_dict = {
                            "y_pred_pos": np.array([y_pred[0]]),
                            "y_pred_neg": y_pred[1:],
                            "eval_metric": [metric],
                        }
                    perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                    perf_idx += 1
            #* update the snapshot embedding
            prev_index = val_snapshots[snapshot_idx]
            prev_index = prev_index.long().to(args.device)
            embeddings = encoder(x=node_feat, edge_index=prev_index)
            embeddings = embeddings.detach()
        
        result = list(perf_list.values())
        perf_list = np.array(result)
        val_metrics = float(np.mean(perf_list))
        val_time = timeit.default_timer() - val_start_time

        print(f"Val {metric}: {val_metrics}")
        print ("Val time: ", val_time)
        if (args.wandb):
            wandb.log({"train_loss":(total_loss),
                    "val_" + metric: val_metrics,
                    "train time": train_time,
                    "val time": val_time,
                    })
            
        #! report test results when validation improves
        if (val_metrics > best_val):
            best_val = val_metrics
            neg_sampler.load_eval_set(fname=args.dataset + "_test_ns.pkl", split_mode="test",)
            test_start_time = timeit.default_timer()
            #* load the test negative samples
            neg_sampler.load_eval_set(fname=args.dataset + "_test_ns.pkl", split_mode="test")

            test_snapshots = test_data['edge_index'] #converted to undirected, also removes self loops as required by HTGN
            test_edges = test_data['original_edges'] #original edges unmodified
            ts_min = min(test_snapshots.keys())

            embeddings = embeddings.detach()
            perf_list = {}
            perf_idx = 0

            for snapshot_idx in test_snapshots.keys():
                pos_index = torch.from_numpy(test_edges[snapshot_idx])
                pos_index = pos_index.long().to(args.device)

                for i in range(pos_index.shape[1]):
                    pos_src = pos_index[0][i].item()
                    pos_dst = pos_index[1][i].item()
                    pos_t = snapshot_idx
                    neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode='test')

                    for idx, neg_batch in enumerate(neg_batch_list):
                        query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                        query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                        query_src = torch.from_numpy(query_src).long().to(args.device)
                        query_dst = torch.from_numpy(query_dst).long().to(args.device)
                        edge_index = torch.stack((query_src, query_dst), dim=0)
                        with torch.no_grad():
                            y_pred = decoder(embeddings[edge_index[0]], embeddings[edge_index[1]])
                        y_pred = y_pred.reshape(-1)
                        y_pred = y_pred.detach().cpu().numpy()

                        input_dict = {
                                "y_pred_pos": np.array([y_pred[0]]),
                                "y_pred_neg": y_pred[1:],
                                "eval_metric": [metric],
                            }
                        perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                        perf_idx += 1

                #* update the snapshot embedding
                prev_index = test_snapshots[snapshot_idx]
                prev_index = prev_index.long().to(args.device)
                embeddings = encoder(x=node_feat, edge_index=prev_index)
                embeddings = embeddings.detach()
            
            result = list(perf_list.values())
            perf_list = np.array(result)
            test_metrics = float(np.mean(perf_list))
            test_time = timeit.default_timer() - test_start_time
            print(f"Test {metric}: {test_metrics}")
            print ("Test time: ", test_time)

        
            best_test = test_metrics
            #* implementing patience
            if ((epoch - best_epoch) >= args.patience and epoch > 1):
                best_epoch = epoch
                print ("run finishes")
                print ("best epoch is, ", best_epoch)
                print ("best val performance is, ", best_val)
                print ("best test performance is, ", best_test)
                print ("------------------------------------------")
                break
            best_epoch = epoch
    print ("run finishes")
    print ("best epoch is, ", best_epoch)
    print ("best val performance is, ", best_val)
    print ("best test performance is, ", best_test)
    print ("------------------------------------------")



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
        run(args, data, seed=seed)
        print ("--------------------------------")

    

