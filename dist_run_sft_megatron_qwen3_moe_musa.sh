#!/bin/bash

CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M:%S")
echo $CURRENT_TIME
mkdir -p ./output/$CURRENT_TIME

TP_SIZE=2
PP_SIZE=1
EP_SIZE=1
WORLD_SIZE=8
MICRO_BATCH_SIZE=1
NUM_MICROBATCHES=2
(( EDP_SIZE = $WORLD_SIZE / ($TP_SIZE * $PP_SIZE * $EP_SIZE) ))
echo "DP_SIZE: $EDP_SIZE"
(( GLOBAL_BATCH_SIZE = $MICRO_BATCH_SIZE * $NUM_MICROBATCHES * $EDP_SIZE * $EP_SIZE ))
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"

set -u
  WORK_HOME="$PWD"
  PATCH_HOME="$PWD"/../..
  EXPNAME="qwen3_30b_a3b_sft_tp${TP_SIZE}_pp${PP_SIZE}_ep${EP_SIZE}_edp${EDP_SIZE}_mbs${MICRO_BATCH_SIZE}_numbs${NUM_MICROBATCHES}_gbs${GLOBAL_BATCH_SIZE}_gpus${WORLD_SIZE}"
  DATA_PATH=$WORK_HOME/megatron-dataset_text_document
  HOSTFILE=./hostfile
  LOG_FILE=./output/$CURRENT_TIME/$EXPNAME.log
  TOKENIZED_MODEL=$WORK_HOME/tokenizer
  SCRIPT_FILE=./30B_A3B/run_sft_qwen3_musa.sh
  RDZV_ID=$CURRENT_TIME
set +u

cmd="bash -c 'cd $WORK_HOME; \
     MOCK_DATA=${MOCK_DATA:-0} \
     bash $SCRIPT_FILE $WORK_HOME $PATCH_HOME $EXPNAME $HOSTFILE \"$DATA_PATH\" \
     $TP_SIZE $PP_SIZE $EP_SIZE \
     $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE $TOKENIZED_MODEL $RDZV_ID"

COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)

for host in ${hostlist[@]}; do
  cmd_ssh=$cmd" &'"
  echo $cmd_ssh
  ssh -f -n $host $cmd_ssh
  ((COUNT++))
done
