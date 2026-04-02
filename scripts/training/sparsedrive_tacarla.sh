#!/bin/bash
# SparseDriveV2 training on TaCarla dataset.
#
# Prerequisites:
#   1. Cache dataset:   python scripts/tacarla/run_tacarla_caching.py ...
#   2. Cluster anchors: python scripts/tacarla/run_tacarla_caching.py ... --cluster
#   3. Download resnet50 backbone to ckpt/resnet50.bin
#
# Usage:
#   TACARLA_CACHE=/path/to/tacarla_cache bash scripts/training/sparsedrive_tacarla.sh

export HYDRA_FULL_ERROR=1

config=default_training
agent=sparsedrive_agent_tacarla

TACARLA_CACHE="${TACARLA_CACHE:?Set TACARLA_CACHE to the cache directory}"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name $config \
    --config-path config/training \
    agent=$agent \
    experiment_name=sparsedrive_tacarla \
    train_test_split=tacarla_train \
    'train_logs=null' \
    'val_logs=null' \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    cache_path=$TACARLA_CACHE \
    dataloader.params.batch_size=6 \
    dataloader.params.num_workers=8 \
    dataloader.params.prefetch_factor=4 \
    trainer.params.max_epochs=100 \
    agent.lr=4e-4
