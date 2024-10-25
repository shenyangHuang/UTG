"""
Command to run:
    python dtdg_roland_dgnn.py --model=ROLAND_DGNN --dataset=uci -t weekly --lr 1e-3 --max_epoch 50 --num_runs 1 --patience 10

    python -u dtdg_roland_dgnn.py -d enron -t monthly --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 100

"""

import torch
import numpy as np
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
import timeit

from models.ROLAND_DGNN import ROLANDGNN
from models.decoders import LinkPredictor


def test_tgb(test_data, node_feat,
             encoder, decoder, 
             current_embeddings, num_previous_edges,
             neg_sampler, evaluator,
             metric, split_mode='val'):
    
    snapshots = test_data['edge_index'] 
    edges = test_data['original_edges']
    
    encoder.eval()
    decoder.eval()

    perf_list = []

    for snapshot_idx in snapshots.keys():
        pos_index = torch.from_numpy(edges[snapshot_idx])
        pos_index = pos_index.long().to(args.device)

        for i in range(pos_index.shape[1]):
            pos_src = pos_index[0][i].item()
            pos_dst = pos_index[1][i].item()

            pos_t = snapshot_idx
            neg_batch_list = neg_sampler.query_batch(np.array([pos_src]), np.array([pos_dst]), np.array([pos_t]), split_mode=split_mode)

            for idx, neg_batch in enumerate(neg_batch_list):
                query_src = np.array([int(pos_src) for _ in range(len(neg_batch) + 1)])
                query_dst = np.concatenate([np.array([int(pos_dst)]), neg_batch])
                query_src = torch.from_numpy(query_src).long().to(args.device)
                query_dst = torch.from_numpy(query_dst).long().to(args.device)
                edge_index = torch.stack((query_src, query_dst), dim=0)

                with torch.no_grad():
                    y_pred = decoder(current_embeddings[1][query_src], current_embeddings[1][query_dst])
                y_pred = y_pred.squeeze(dim=-1).detach()

            input_dict = {
                        "y_pred_pos": np.array([y_pred[0].cpu()]),
                        "y_pred_neg": np.array(y_pred[1:].cpu()),
                        "eval_metric": [metric],
                        }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # these are the previous snapshot parameters, update before a new prediction
        # computer new number of edges
        num_current_edges = len(pos_src)
        num_previous_edges = num_previous_edges + num_current_edges
        # update last-embeddings for the next round
        last_embeddings = current_embeddings
        
        #* update the model now if the prediction batch has moved to next snapshot
        cur_index = snapshots[snapshot_idx]
        cur_index = cur_index.long().to(args.device)
        current_embeddings = encoder(node_feat, cur_index, \
                last_embeddings, num_current_edges, num_previous_edges)

    test_metric = float(np.mean(np.array(perf_list)))

    return test_metric, last_embeddings, num_previous_edges


if __name__ == '__main__':
    from utils.configs import args
    from utils.utils_func import set_random
    from utils.data_util import loader

    # set the random seed
    set_random(args.seed)

    # set up some parameters
    batch_size = args.batch_size
    num_epochs = args.max_epoch
    lr = args.lr

    # ROLAND-DGNN related parameters
    roland_embed_size = 256
    node_feat_dim = roland_embed_size
    dec_hid_dim = roland_embed_size
    output_dim = 1
    link_pred_num_layers = 2
    edge_feat_dim = 1
    update_strategy = args.roland_update  # default: "gru"

    # Data Loading...
    data = loader(dataset=args.dataset, time_scale=args.time_scale)

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    num_nodes = data['train_data']['num_nodes'] + 1
    print(f"INFO: Number of nodes: {num_nodes}")

    # set up node features: TODO: there is no node features for any dataset
    node_feat = torch.randn((num_nodes, node_feat_dim)).to(args.device)


    for seed in range(args.seed, args.seed + args.num_runs):
        set_random(seed)
        print(f"INFO: Start a fresh run with random seed: {seed}")
        
        # initialization of the model to prep for training
        model_dim = {
            "input_dim": node_feat_dim,
            "hidden_conv_1": dec_hid_dim,  
            "hidden_conv_2": dec_hid_dim,  
        }

        # define model
        encoder = ROLANDGNN(model_dim, num_nodes, args.dropout, update=update_strategy).to(args.device)
        decoder = LinkPredictor(dec_hid_dim, dec_hid_dim, output_dim, link_pred_num_layers, args.dropout).to(args.device)
        optimizer = torch.optim.Adam(
            set(encoder.parameters()) | set(decoder.parameters()), lr=lr)
        
        # criterion
        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.MSELoss()

        best_val = 0
        best_test = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            print ("-"*100)
            train_start_time = timeit.default_timer()
            total_loss = 0

            encoder.train()
            decoder.train()

            snapshot_list = train_data['edge_index']
            total_loss = 0
            for snapshot_idx in range(train_data['time_length']):

                # neg_edges = negative_sampling(pos_index, num_nodes=num_nodes, num_neg_samples=(pos_index.size(1)*1), force_undirected = True)
                if (snapshot_idx == 0): #first snapshot, feed the current snapshot
                    cur_index = snapshot_list[snapshot_idx]
                    cur_index = cur_index.long().to(args.device)

                    num_previous_edges = 0
                    last_embeddings = [torch.Tensor([[0 for i in range(model_dim["hidden_conv_1"])] for j in range(num_nodes)]).to(args.device), \
                                       torch.Tensor([[0 for i in range(model_dim["hidden_conv_2"])] for j in range(num_nodes)]).to(args.device)]
                    # current_embeddings = encoder(node_feat, cur_index)
                    edge_index = cur_index

                else: #subsequent snapshot, feed the previous snapshot
                    prev_index = snapshot_list[snapshot_idx-1]
                    prev_index = prev_index.long().to(args.device)
                    edge_index = prev_index
                    # current_embeddings = encoder(node_feat, prev_index, \
                    #     last_embeddings, None, num_previous_edges)

                # pass through the encoder
                num_current_edges = edge_index.shape[1]
                current_embeddings = encoder(node_feat, edge_index, \
                        last_embeddings, num_current_edges, num_previous_edges)
                
                # computer new number of edges
                num_previous_edges = num_previous_edges + num_current_edges
                # update last-embeddings for the next round
                last_embeddings = current_embeddings

                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)

                neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_index.shape[1],),
                        dtype=torch.long,
                        device=args.device,
                    )

                pos_pred = decoder(last_embeddings[1][pos_index[0]], last_embeddings[1][pos_index[1]])
                neg_pred = decoder(last_embeddings[1][pos_index[0]], last_embeddings[1][neg_dst])

                loss = criterion(pos_pred, torch.ones_like(pos_pred))
                loss += criterion(neg_pred, torch.zeros_like(neg_pred))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() / pos_index.shape[1]

            train_time = timeit.default_timer() - train_start_time
            train_time = timeit.default_timer() - train_start_time
            print (f'INFO: Epoch {epoch}/{num_epochs}, Loss: {total_loss}')
            print ("INFO: Train time: ", train_time)

            # VALIDATION
            evaluator = Evaluator(name=args.dataset) #reuse MRR evaluator from TGB
            metric = "mrr"
            neg_sampler = NegativeEdgeSampler(dataset_name=args.dataset, strategy="hist_rnd")

            #* load the val negative samples
            neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_val_ns.pkl", split_mode="val")

            val_start_time = timeit.default_timer()

            test_tgb(test_data, node_feat,
             encoder, decoder, 
             current_embeddings, num_previous_edges,
             neg_sampler, evaluator,
             metric, split_mode='val')


            val_metric, val_last_embeddings, val_num_previous_edges = test_tgb(val_data, node_feat, 
                                                                        encoder, decoder, 
                                                                        current_embeddings, 
                                                                        num_previous_edges, 
                                                                        neg_sampler, evaluator,
                                                                        metric, split_mode='val')

            val_time = timeit.default_timer() - val_start_time
            print(f"INFO: Validation {metric}: {val_metric}")
            print (f"INFO: Validation time: {val_time}")
                
            # TEST
            # report test results when validation improves
            if (val_metric > best_val):
                best_val = val_metric
                neg_sampler.load_eval_set(fname=f"data/{args.dataset}/{args.dataset}_test_ns.pkl", split_mode="test",)

                test_start_time = timeit.default_timer()

                test_metric, test_last_embeddings, test_num_previous_edges = test_tgb(test_data, node_feat, 
                                                                                    encoder, decoder, 
                                                                                    val_last_embeddings, 
                                                                                    val_num_previous_edges, 
                                                                                    neg_sampler, evaluator,
                                                                                    metric, split_mode='test')



                test_time = timeit.default_timer() - test_start_time
                best_val = val_metric
                best_test = test_metric

                print (f"INFO: Test metric: {test_metric}")
                print (f"INFO: Test time: {test_time}")
                print ("-"*100)

                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    break
                best_epoch = epoch

        print (f"INFO: Run {seed} finished.")
        print (f"INFO: Best epoch: {best_epoch}")
        print (f"INFO: Best val performance: {best_val}")
        print (f"INFO: Best test performance: {best_test}")
        print ("-"*100)









