#!/bin/bash
# Stop distributed Qwen3 Megatron jobs on all nodes in hostfile.

HOSTFILE=./hostfile
NUM_NODES=$(grep -v '^#\|^$' $HOSTFILE | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
for host in ${hostlist[@]}; do
    ssh -f -n $host "pkill -f 'pretrain_gpt.py' || true"
    ssh -f -n $host "pkill -f 'torchrun' || true"
    echo "$host: kill signals sent."
done
