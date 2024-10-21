#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=5-00:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=4           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=64G                   # memory (per node)
#SBATCH --job-name=UTG_review_EGCNO_May8_seed=5
#SBATCH --output=outlog/%x-%j.log



# -----------------------------------------------------------
# --------------- Dataset & Time-granularity ----------------
# -----------------------------------------------------------

# # ===================== Data
# # tgbl-wiki
# data="tgbl-wiki"
# time_scale="hourly"

# tgbl-review
data="tgbl-review"
time_scale="monthly"

# # tgbl-subreddit
# data="tgbl-subreddit"
# time_scale="hourly"

# # tgbl-lastfm
# data="tgbl-lastfm"
# time_scale="weekly"

# # =====================

# # ===================== Model
# model="HTGN"
# model="GCLSTM"
model="EGCNO"
# model="GCN"
# # =====================

max_epoch=200
patience=50
num_runs=1

#lr=1e-3
lr=2e-4

min_seed=5
max_seed=6


for ((seed=$min_seed; seed<$max_seed; seed++))
do
    echo "========================="
    echo "Model: $model"
    echo "DATA: $data"
    echo "seed: $seed"
    echo "num_runs: $num_runs"
    echo "time_scale: $time_scale"
    echo "learning rate: $lr"
    echo "max_epochs: $max_epoch"
    echo "patience: $patience"
    echo "========================="

    if [ "$model" = "HTGN" ]; then
        echo "======================================"
        echo "================= HTGN ================="
        echo "======================================"
        # ------------- HTGN -------------
        python -u ctdg_main_htgn.py --model="$model" --dataset="$data" -t "$time_scale" \
        --lr "$lr" --max_epoch "$max_epoch" --num_runs "$num_runs" --patience "$patience" --seed "$seed"
    
    elif [ "$model" = "GCLSTM" ]; then
        echo "======================================"
        echo "================= GCLSTM ================="
        echo "======================================"
        # ------------- GCLSTM -------------
        python -u ctdg_gclstm.py --model="$model" --dataset="$data" -t "$time_scale" \
        --lr "$lr" --max_epoch "$max_epoch" --num_runs "$num_runs" --patience "$patience" --seed "$seed"

    elif [ "$model" = "EGCNO" ]; then
        echo "======================================"
        echo "================= EGCNO ================="
        echo "======================================"
        # ------------- EGCNO -------------
        python -u ctdg_egcno.py --model="$model" --dataset="$data" -t "$time_scale" \
        --lr "$lr" --max_epoch "$max_epoch" --num_runs "$num_runs" --patience "$patience" --seed "$seed"

    elif [ "$model" = "GCN" ]; then
        echo "======================================"
        echo "================= GCN ================="
        echo "======================================"
        # ------------- GCN -------------
        python -u ctdg_utg_gcn.py --model="$model" --dataset="$data" -t "$time_scale" \
        --lr "$lr" --max_epoch "$max_epoch" --num_runs "$num_runs" --patience "$patience" --seed "$seed"

    
    else
      echo "======================================"
      echo "================= Invalid Model!!! ================="
      echo "======================================"

    fi
done


echo "=========================================="
echo "=========================================="



# python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-wiki -t hourly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1
# python ctdg_gclstm.py --model=HTGN --dataset=tgbl-wiki -t hourly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1
# python ctdg_egcno.py --model=HTGN --dataset=tgbl-wiki -t hourly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1

# python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-review -t monthly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1

# python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-subreddit -t hourly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1

# python ctdg_main_htgn.py --model=HTGN --dataset=tgbl-lastfm -t weekly --lr 1e-3 --max_epoch 1 --num_runs 1 --patience 1