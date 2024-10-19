#!/bin/bash
#SBATCH --partition=long #unkillable #main #long
#SBATCH --output=gclstm_mooc_full.txt 
#SBATCH --error=gclstm_mooc_full_error.txt 
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                  # Ask for 1 titan xp gpu:rtx8000:1 
#SBATCH --mem=32G #64G                             # Ask for 32 GB of RAM
#SBATCH --time=48:00:00    #48:00:00                   # The job will run for 1 day

export HOME="/home/mila/h/huangshe"
module load python/3.9
source $HOME/tgbenv/bin/activate
pwd

#* for gclstm original training

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_original.py --dataset uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_original.py --dataset enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_original.py --dataset contacts -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_original.py --dataset social_evo -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_original.py --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

#* for HTGN original training

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_htgn_original.py --model=HTGN --dataset=enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_htgn_original.py --model=HTGN --dataset=uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_htgn_original.py --model=HTGN --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_htgn_original.py --model=HTGN --dataset social_evo -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_htgn_original.py --model=HTGN --dataset contacts -t hourly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100



#* for EGCNO original training

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset=enron -t monthly --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset=uci -t weekly --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset mooc -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset social_evo -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset contacts -t hourly --lr 1e-3 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_egcno_original.py --dataset=canparl -t biyearly --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100



#* for gclstm cur t snapshot gradient with update

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_samet.py --dataset uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_samet.py --dataset enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_samet.py --dataset contacts -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_samet.py --dataset social_evo -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_samet.py --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100


#* for gclstm full gradient with update t - 1

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_full.py --dataset uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_full.py --dataset enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_full.py --dataset contacts -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_full.py --dataset social_evo -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_full.py --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100




#* for gclstm full gradient no update

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_noupdate.py --dataset uci -t weekly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_noupdate.py --dataset enron -t monthly --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_noupdate.py --dataset contacts -t hourly --lr 2e-4 --max_epoch 200 --num_runs 5 --patience 50

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_noupdate.py --dataset social_evo -t daily --lr 1e-3 --max_epoch 500 --num_runs 5 --patience 100

# CUDA_VISIBLE_DEVICES=0 python -u dtdg_gclstm_noupdate.py --dataset mooc -t daily --lr 2e-4 --max_epoch 500 --num_runs 5 --patience 100





