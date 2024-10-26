import os
import sys
import time
import timeit
import torch
import numpy as np
from torch_geometric.utils.negative_sampling import negative_sampling
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from torch_geometric.loader import TemporalDataLoader
import wandb
import math

from torcheval.metrics import BinaryAUROC
from torcheval.metrics import BinaryAUPRC

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



class Runner(object):
    def __init__(self):
        
        self.train_data = data['train_data']
        #* for tgb dataset, we will convert the val and test set on the fly here instead of loading. 
        self.val_data = data['val_data']
        self.test_data = data['test_data']


        args.num_nodes = data['train_data']['num_nodes'] + int(0.1 * data['train_data']['num_nodes']) # make it larger to fit the inductive nodes

        if args.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="utg",
                
                # track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": args.model,
                "dataset": args.dataset,
                "time granularity": args.time_scale,
                }
            )


        # set initial features; the assumption is to have trainable features
        self.x = None  # TODO: @Andy --> this might need to change if we want to load initial features
        
        # set the model
        self.model = load_model(args).to(args.device)

        
        # set the loss
        self.loss = ReconLoss(args) if args.model not in ['DynVAE', 'VGRNN', 'HVGRNN'] else VGAEloss(args)
        
    def optimizer(self, using_riemannianAdam=True):
        if using_riemannianAdam:
            import geoopt
            optimizer = geoopt.optim.radam.RiemannianAdam(self.model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
        else:
            import torch.optim as optim
            optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer
    

    #! assumes there is no time gap, throws error if so 
    def test_tgb(self, 
                 test_loader, 
                 test_snapshots,
                 num_nodes,  
                 embeddings):
        r"""
        TGB evaluation for discrete model

        Parameters:
            full_data: TGB dataset
            test_mask: mask for the test set
            test_snapshots: the test set in the form of snapshots
            neg_sampler: TGB negative sampler
            evaluator: TGB evaluator
            embeddings: the embeddings of the model
            metric: the metric to be used for evaluation
            split_mode: the split mode for the negative sampler
        """
        auc = BinaryAUROC()
        auprc = BinaryAUPRC()

        embeddings = embeddings.detach()

        ts_list = test_snapshots['ts_map']
        ts_idx = min(list(ts_list.keys()))
        max_ts_idx = max(list(ts_list.keys()))

        for pos_batch in test_loader:
            pos_src, pos_dst, pos_t, pos_msg = (
                pos_batch.src,
                pos_batch.dst,
                pos_batch.t,
                pos_batch.msg,
            )

            neg_dst = torch.randint(
                        0,
                        num_nodes,
                        (pos_src.shape[0],),
                        dtype=torch.long,
                        device=pos_src.device,
                    )
            query_src = torch.cat((pos_src, pos_src), dim=0)
            query_dst = torch.cat((pos_dst, neg_dst), dim=0)
            y_label = torch.cat((torch.ones((pos_src.shape[0],), 
                                            dtype=torch.float,), 
                                torch.zeros((pos_src.shape[0],),
                                        dtype=torch.float,)), dim=0)
            edge_index = torch.stack((query_src, query_dst), dim=0).to(embeddings.device)
            with torch.no_grad():
                    y_pred = torch.from_numpy(self.loss.predict_link(embeddings, edge_index))
           
            auc.update(y_pred, y_label)
            auprc.update(y_pred, y_label)
            
            #* update the model with past snapshots that have passed, check the last timestamp of this batch
            while (pos_t[-1] > ts_list[ts_idx] and ts_idx < max_ts_idx):
                with torch.no_grad():
                    pos_index = test_snapshots['edge_index'][ts_idx]
                    pos_index = pos_index.long().to(args.device)
                    z = self.model(pos_index, self.x)
                    embeddings = self.model.update_hiddens_all_with(z)
                ts_idx += 1
        
        #* update to the final snapshot
        with torch.no_grad():
            pos_index = test_snapshots['edge_index'][max_ts_idx]  #last snapshot
            pos_index = pos_index.long().to(args.device)
            z = self.model(pos_index, self.x)
            embeddings = self.model.update_hiddens_all_with(z)

        test_auc = auc.compute()
        test_auprc = auprc.compute()

        return test_auc, test_auprc, embeddings

    def run(self, seed=1):
        
        set_random(seed)
        optimizer = self.optimizer()  # @TODO: RiemannianAdam or Adam?!
        self.model.reset_parameters()
        self.model.train()

        best_val = 0 
        best_test = 0
        best_epoch = 0

        BATCH_SIZE = 200 #from TGB 


        dataset = PyGLinkPropPredDataset(name=args.dataset, root="datasets")
        full_data = dataset.get_TemporalData()
        metric = dataset.eval_metric
        neg_sampler = dataset.negative_sampler
        # get masks
        val_mask = dataset.val_mask
        test_mask = dataset.test_mask

        evaluator = Evaluator(name=args.dataset)

        for epoch in range(1, args.max_epoch + 1):
            epoch_start_time = timeit.default_timer()
            
            epoch_losses = []
            self.model.init_hiddens()
            # ==========================
            # Train
            # important to reset the node embeddings z at the start of the epoch
            train_start_time = timeit.default_timer()
            self.model.train()
            z = None
            snapshot_list = self.train_data['edge_index']
            for snapshot_idx in range(self.train_data['time_length']):
                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)
                optimizer.zero_grad()

                #* generate random samples for training
                neg_index = generate_random_negatives(pos_index, num_nodes=args.num_nodes, num_neg_samples=1)
                if (snapshot_idx == 0):
                    z = self.model(pos_index, self.x)
                               
                if args.use_htc == 0:
                    epoch_loss = self.loss(z, pos_edge_index=pos_index,  neg_edge_index=neg_index)
                else:
                    epoch_loss = self.loss(z, pos_edge_index=pos_index, neg_edge_index=neg_index) + self.model.htc(z)
                
                epoch_loss.backward()
                optimizer.step()
                epoch_losses.append(epoch_loss.item())

                #* update the embedding after the prediction
                pos_index = snapshot_list[snapshot_idx]
                pos_index = pos_index.long().to(args.device)
                z = self.model(pos_index, self.x)
                z = self.model.update_hiddens_all_with(z) 
            
            average_epoch_loss = np.mean(epoch_losses)
            #? terminate if the loss is nan
            if math.isnan(average_epoch_loss):
                print('nan loss')
                break



            train_end_time = timeit.default_timer()

            #! validation and test code start up, use different logic than training
            
            # ==========================
            # Validation    
            # 1. load the queries from TGB
            # 2. remap the ts of the qeury
            self.model.eval()

            # id_map = self.train_data['id_map'] #remapped node id

            # dataset.load_val_ns()
            val_data = full_data[val_mask]
            val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
            #* steps for snapshots edges in test set
            #1. discretize the test set into snapshots
            #2. map from integer to unix ts
            val_start_time = timeit.default_timer()
            val_auc, val_auprc, z = self.test_tgb(val_loader, self.val_data, args.num_nodes, z)

            val_end_time = timeit.default_timer()

             # logging stats
            epoch_end_time = timeit.default_timer()
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            

            logger.info("Epoch:{}, Time (train + val): {:.4f}, GPU Mem.: {:.1f}MiB".format(epoch,
                                                                                epoch_end_time - epoch_start_time,
                                                                                gpu_mem_alloc))
            logger.info("\tTrain: Loss: {:.4f}, Elapsed time: {:.4f}".format(average_epoch_loss, train_end_time - train_start_time))
            logger.info("\tValidation: AUC: {:.4f}, AUPRC: {:.4f}, Elapsed time: {:.4f}".format(val_auc, val_auprc, val_end_time - val_start_time))

 
            if (args.wandb):
                wandb.log({"train_loss": average_epoch_loss,
                        "val_auc": val_auc,
                        "val_auprc": val_auprc,
                        "train time": train_end_time - train_start_time,
                        "val time": val_end_time - val_start_time,
                        })
                
             #! report test results when validation improves
            if (val_auc > best_val):
                best_val = val_auc
                # dataset.load_test_ns()

                test_data = full_data[test_mask]
                test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

                test_start_time = timeit.default_timer()
                test_auc, test_auprc, z = self.test_tgb(test_loader, self.test_data, args.num_nodes, z)
                test_end_time = timeit.default_timer()

                logger.info("\tTest: AUC: {:.4f}, AUPRC: {:.4f}, Elapsed time: {:.4f}".format(test_auc, test_auprc, test_end_time - test_start_time))
                
                #* implementing patience
                if ((epoch - best_epoch) >= args.patience and epoch > 1):
                    best_epoch = epoch
                    print ("run finishes")
                    print ("best epoch is, ", best_epoch)
                    print ("best val AUC:", val_auc)
                    print ("best val AUPRC:", val_auprc)
                    print ("best test AUC:", test_auc)
                    print ("best test AUPRC:", test_auprc)
                    return
                best_epoch = epoch

        print ("run finishes")
        print ("best epoch is, ", best_epoch)
        print ("best val performance is, ", best_val)
        print ("best test performance is, ", best_test)


           
            

if __name__ == '__main__':
    from utils.configs import args
    from utils.log_utils import logger, init_logger
    from utils.utils_func import set_random
    from models.load_model import load_model
    from models.loss import ReconLoss, VGAEloss
    from utils.data_util import loader, prepare_dir

    set_random(args.seed)
    data = loader(dataset=args.dataset, time_scale=args.time_scale)
    init_logger(prepare_dir(args.output_folder) + args.dataset + '_timeScale_' + str(args.time_scale) + '_seed_' + str(args.seed) + '.log')
    
    for seed in range(args.seed, args.seed + args.num_runs):
        runner = Runner()
        print ("--------------------------------")
        print ("excuting run with seed ", seed)
        runner.run(seed=seed)
        print ("--------------------------------")