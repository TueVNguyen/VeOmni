#!/bin/bash

set -x
conda env list
eval "$(conda shell.bash hook)"
conda activate sft_veomni
export TOKENIZERS_PARALLELISM=false
export HF_CACHE_DIR=/home/slurm/tuenv2/tuenv/hf_cache
export WANDB_API_KEY=5e2ecb188b3d7cfac3e3067854bef0a9241ca1e9
export WANDB_ENTITY=meoconxinhxan
NNODES=${COUNT_NODE:=1}
NPROC_PER_NODE=${NPROC_PER_NODE:=8}
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}


H=$(hostname)
# RANK=$(echo -e $HOSTNAMES | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]")

echo hostname = $(hostname)
echo COUNT_NODE = $COUNT_NODE

torchrun --nnodes=$COUNT_NODE --nproc_per_node=$NPROC_PER_NODE \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    --node_rank=$SLURM_PROCID  $@ 2>&1 | tee log.txt
