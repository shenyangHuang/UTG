import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM 
from torch_geometric.utils.negative_sampling import negative_sampling
# from models.tgn.decoder import LinkPredictor
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
import wandb
import timeit



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, K=1):
        #https://pytorch-geometric-temporal.readthedocs.io/en/latest/modules/root.html#recurrent-graph-convolutional-layers
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(in_channels=node_feat_dim, 
                                out_channels=hidden_dim, 
                                K=K,) #K is the Chebyshev filter size
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight, h, c):
        r"""
        forward function for the model, 
        this is used for each snapshot
        h: node hidden state matrix from previous time
        c: cell state matrix from previous time
        """
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0


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




if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)

    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="utg",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "architecture": "gclstm",
            "dataset": args.dataset,
            "time granularity": args.time_scale,
            }
        )
    #! add support for node features in the future
    #node_feat_dim = 16 #all 0s for now
    node_feat_dim = 256 #for node features
    edge_feat_dim = 1 #for edge weights
    hidden_dim = 256

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    num_epochs = 200
    lr = args.lr

    #* initialization of the model to prep for training
    model = RecurrentGCN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, K=1).to(args.device)
    # node_feat = torch.zeros((num_nodes, node_feat_dim)).to(args.device)
    node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)

    # link_pred = LinkPredictor(in_channels=hidden_dim).to(args.device)
    link_pred = LinkPredictor(hidden_dim, hidden_dim, 1,
                              2, 0.2).to(args.device)


    optimizer = torch.optim.Adam(
        set(model.parameters()) | set(link_pred.parameters()), lr=lr)
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.MSELoss()

    best_val = 0
    best_test = 0

    for epoch in range(num_epochs):
        train_start_time = timeit.default_timer()
        optimizer.zero_grad()
        total_loss = 0
        model.train()
        link_pred.train()
        snapshot_list = train_data['edge_index']
        h_0, c_0, h = None, None, None
        total_loss = 0
        for snapshot_idx in range(train_data['time_length']):

            optimizer.zero_grad()
            # neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1), force_undirected = True)
            if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                cur_index = snapshot_list[snapshot_idx]
                cur_index = cur_index.long().to(args.device)
                # TODO, also need to support edge attributes correctly in TGX
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(cur_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, cur_index, edge_attr, h_0, c_0)
            else: #subsequent snapshot, feed the previous snapshot
                prev_index = snapshot_list[snapshot_idx-1]
                prev_index = prev_index.long().to(args.device)
                if ('edge_attr' not in train_data):
                    edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)

            pos_index = snapshot_list[snapshot_idx]
            pos_index = pos_index.long().to(args.device)

            neg_dst = torch.randint(
                    0,
                    num_nodes,
                    (pos_index.shape[1],),
                    dtype=torch.long,
                    device=args.device,
                )

            pos_out = link_pred(h[pos_index[0]], h[pos_index[1]])
            # pos_loss = -torch.log(pos_out + 1e-15).mean()


            neg_out = link_pred(h[pos_index[0]], h[neg_dst])
            # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
            #loss = pos_loss + neg_loss

            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            loss.backward()
            optimizer.step()

            total_loss += float(loss)


            h_0 = h_0.detach()
            c_0 = c_0.detach()

        print (f'Epoch {epoch}/{num_epochs}, Loss: {total_loss/num_nodes}')

        train_time = timeit.default_timer() - train_start_time
        #! Evaluation starts here
        #! need to optimize code to have train, test function, maybe in a class

        val_start_time = timeit.default_timer()
        model.eval()
        link_pred.eval()
        evaluator = Evaluator(name="tgbl-wiki") #reuse MRR evaluator from TGB
        metric = "mrr"
        neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

        #* load the val negative samples
        neg_sampler.load_eval_set(fname=args.dataset + "_val_ns.pkl", split_mode="val")

        val_snapshots = val_data['edge_index'] #converted to undirected, also removes self loops as required by HTGN
        val_edges = val_data['original_edges'] #original edges unmodified
        ts_min = min(val_snapshots.keys())

        h_0 = h_0.detach()
        c_0 = c_0.detach()
        h = h.detach()

        perf_list = {}
        perf_idx = 0

        for snapshot_idx in val_snapshots.keys():
            pos_index = torch.from_numpy(val_edges[snapshot_idx])
            pos_index = pos_index.long().to(args.device)
            #* update the node embeddings with edges from previous snapshot
            if (snapshot_idx > ts_min):
                #* update the snapshot embedding
                prev_index = val_snapshots[snapshot_idx-1]
                prev_index = prev_index.long().to(args.device)
                if ('edge_attr' not in val_data):
                    edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                else:
                    raise NotImplementedError("Edge attributes are not yet supported")
                h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
            
            for i in range(pos_index.shape[0]):
                pos_src = pos_index[i][0].item()
                pos_dst = pos_index[i][1].item()
                pos_t = snapshot_idx
                neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode='val')
                
                for idx, neg_batch in enumerate(neg_batch_list):
                    query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                    query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                    query_src = torch.from_numpy(query_src).long().to(args.device)
                    query_dst = torch.from_numpy(query_dst).long().to(args.device)
                    edge_index = torch.stack((query_src, query_dst), dim=0)
                    y_pred = link_pred(h[edge_index[0]], h[edge_index[1]])
                    y_pred = y_pred.reshape(-1)
                    y_pred = y_pred.detach().cpu().numpy()

                    input_dict = {
                            "y_pred_pos": np.array([y_pred[0]]),
                            "y_pred_neg": y_pred[1:],
                            "eval_metric": [metric],
                        }
                    perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                    perf_idx += 1

        result = list(perf_list.values())
        perf_list = np.array(result)
        val_metrics = float(np.mean(perf_list))
        val_time = timeit.default_timer() - val_start_time

        print(f"Epoch {epoch} : Val {metric}: {val_metrics}")
        if (args.wandb):
            wandb.log({"train_loss":(total_loss/num_nodes),
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

            h_0 = h_0.detach()
            c_0 = c_0.detach()
            h = h.detach()

            perf_list = {}
            perf_idx = 0

            for snapshot_idx in test_snapshots.keys():
                pos_index = torch.from_numpy(test_edges[snapshot_idx])
                pos_index = pos_index.long().to(args.device)
                #* update the node embeddings with edges from previous snapshot
                if (snapshot_idx > ts_min):
                    #* update the snapshot embedding
                    prev_index = test_snapshots[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    if ('edge_attr' not in test_data):
                        edge_attr = torch.ones(prev_index.size(1), edge_feat_dim).to(args.device)
                    else:
                        raise NotImplementedError("Edge attributes are not yet supported")
                    h, h_0, c_0 = model(node_feat, prev_index, edge_attr, h_0, c_0)
                
                for i in range(pos_index.shape[0]):
                    pos_src = pos_index[i][0].item()
                    pos_dst = pos_index[i][1].item()
                    pos_t = snapshot_idx
                    neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode='test')
                    
                    for idx, neg_batch in enumerate(neg_batch_list):
                        query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                        query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                        query_src = torch.from_numpy(query_src).long().to(args.device)
                        query_dst = torch.from_numpy(query_dst).long().to(args.device)
                        edge_index = torch.stack((query_src, query_dst), dim=0)
                        y_pred = link_pred(h[edge_index[0]], h[edge_index[1]])
                        y_pred = y_pred.reshape(-1)
                        y_pred = y_pred.detach().cpu().numpy()

                        input_dict = {
                                "y_pred_pos": np.array([y_pred[0]]),
                                "y_pred_neg": y_pred[1:],
                                "eval_metric": [metric],
                            }
                        perf_list[perf_idx] = evaluator.eval(input_dict)[metric]
                        perf_idx += 1
            result = list(perf_list.values())
            perf_list = np.array(result)
            test_metrics = float(np.mean(perf_list))
            test_time = timeit.default_timer() - test_start_time
            print(f"Epoch {epoch} : Test {metric}: {test_metrics}")

            best_test = test_metrics
            #* implementing patience
            if ((epoch - best_epoch) >= args.patience and epoch > 1):
                best_epoch = epoch
                print ("run finishes")
                print ("best epoch is, ", best_epoch)
                print ("best val performance is, ", best_val)
                print ("best test performance is, ", best_test)
            best_epoch = epoch








