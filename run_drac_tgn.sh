#!/bin/bash

#SBATCH --account=def-rrabba
##SBATCH --account=def-bengioy
#SBATCH --time=2-00:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=4           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=32G                   # memory (per node)
#SBATCH --job-name=UTG_lastfm_TGN_seed=5
#SBATCH --output=outlog/%x-%j.log


SEED=5
python -u tgn_tgb.py --seed "$SEED"