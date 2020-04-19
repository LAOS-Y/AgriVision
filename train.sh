#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO

PYTHON=${PYTHON:-"python"}

export CUDA_VISIBLE_DEVICES=$1
PORT=$2
CONFIG=$3
NCCL_SOCKET=$4

arr=(`echo $CUDA_VISIBLE_DEVICES | tr ',' ' '`)
NUM_GPU=${#arr[*]}

export NCCL_SOCKET_IFNAME=$NCCL_SOCKET

echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
echo "NUM_GPU:" $NUM_GPU
echo "NCCL_SOCKET_IFNAME:" $NCCL_SOCKET_IFNAME

OMP_NUM_THREADS=1 $PYTHON\
    -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=$PORT \
    train.py --cfg $CONFIG