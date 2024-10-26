#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=1-00:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=4           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=16G                   # memory (per node)
#SBATCH --job-name=UTG_htgn_reddit_auc_ap
#SBATCH --output=outlog/%x-%j.log

# export HOME="/home/mila/h/huangshe"
# module load python/3.9
# source $HOME/tgbenv/bin/activate
# pwd



#* for GCN

# python utg_main_gnn.py --dataset=tgbl-wiki -t hourly --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 50 --batch_size 200

# python utg_main_gnn.py --dataset=tgbl-review -t monthly --lr 2e-4 --max_epoch 200 --seed 1 --num_runs 5 --patience 20 --batch_size 2000

# python ctdg_main_htgn_auc_ap.py --dataset=tgbl-review -t monthly --lr 2e-4 --max_epoch 200 --seed 1 --num_runs 5 --patience 20 --batch_size 2000

python ctdg_main_htgn_auc_ap.py --model=HTGN --dataset=tgbl-reddit -t hourly --lr 1e-3 --max_epoch 200 --num_runs 5 --patience 50 --seed 1

python ctdg_main_htgn_auc_ap.py --model=HTGN --dataset=tgbl-reddit -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50 --seed 1



#* for TGN

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d enron -t monthly --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d uci -t weekly --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d mooc -t daily --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d social_evo -t daily --lr 2e-4 --max_epoch 500 --seed 1 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d contacts -t hourly --lr 2e-4 --max_epoch 200 --seed 1 --num_runs 5 --patience 50




#* for EGCNO

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset=enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset=uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset social_evo -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset contacts -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python dtdg_egcno.py --dataset=canparl -t biyearly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

#* for HTGN TGB

# CUDA_VISIBLE_DEVICES=0 python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-wiki -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-review -t monthly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-coin -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50



#* for gclstm

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset=canparl -t biyearly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset=enron -t monthly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset=uci -t weekly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset mooc -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset social_evo -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm.py --dataset contacts -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_tgn.py -d canparl -t biyearly --lr 0.001 --max_epoch 500 --seed 1 --num_runs 5 --patience 100





#* for utg 

# CUDA_VISIBLE_DEVICES=0 python -u utg_main_gnn_time.py --dataset=tgbl-wiki -t hourly --lr 0.001 --max_epoch 500 --num_runs 1 --patience 100 --wandb

# CUDA_VISIBLE_DEVICES=0 python -u utg_main_gnn.py --dataset=tgbl-wiki -t hourly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100 --wandb

#* for HTGN

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_main_htgn.py --model=HTGN --dataset=enron -t monthly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_main_htgn.py --model=HTGN --dataset=uci -t weekly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_main_htgn.py --model=HTGN --dataset mooc -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_main_htgn.py --model=HTGN --dataset social_evo -t daily --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_main_htgn.py --model=HTGN --dataset contacts -t hourly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u ctdg_main_htgn.py --model=HTGN --dataset=tgbl-wiki -t hourly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u ctdg_main_htgn.py --model=HTGN --dataset=tgbl-review -t monthly --lr 0.001 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u ctdg_main_htgn.py --model=HTGN --dataset=tgbl-coin -t hourly --lr 0.001 --max_epoch 200 --num_runs 5 --patience 50




# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_training.py -d tgbl-coin -t weekly --seed 3
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-wiki -t hourly --seed 3 --dtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-wiki -t hourly --seed 1 --nodtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-review -t minutely --seed 1 --nodtrain
# CUDA_VISIBLE_DEVICES=0 python tgn_dtdg_eval.py -d tgbl-review -t hourly --seed 1 --dtrain