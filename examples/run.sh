#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=20:00:00           # time (DD-HH:MM)
#SBATCH --cpus-per-task=2           # CPU cores/threads
#SBATCH --gres=gpu:1                # number of GPU(s) per node
#SBATCH --mem=32G                   # memory (per node)
#SBATCH --job-name=UT_0
#SBATCH --output=outlog/%x-%j.log


echo "==================================================="
echo "EvolveGCN-H --- tgbl-wiki --- hourly --- LR=0.0001"
echo "==================================================="
echo ""

python evolvegcnh_CT.py --dataset tgbl-wiki -t hourly --lr 0.0001

echo ""
echo ""
echo "==================================================="
echo "==================================================="
