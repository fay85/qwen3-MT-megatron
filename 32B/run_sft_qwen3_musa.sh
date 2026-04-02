#!/bin/bash
# Qwen3-32B SFT on multi-node MUSA GPUs (Megatron-Core + TransformerEngine FP8).
#
# HF reference: Qwen/Qwen3-32B
#   hidden_size=5120, num_layers=64, num_heads=64, num_kv_heads=8,
#   head_dim=128 (requires --kv-channels 128), intermediate_size=25600,
#   rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=false,
#   vocab_size=151936
#
# Args: WORK_HOME PATCH_HOME EXPNAME HOSTFILE DATA_DIR TP PP MBS GBS TOKENIZER RDZV_ID

set -u
  WORK_HOME=$1
  PATCH_HOME=$2
  EXPNAME=$3
  HOSTFILE=$4
  DATA_DIR=${5:-}
  TP_SIZE=$6
  PP_SIZE=$7
  MICRO_BATCH_SIZE=$8
  GLOBAL_BATCH_SIZE=${9}
  TOKENIZED_MODEL=${10}
  RDZV_ID=${11}
set +u

export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
export MCCL_PROTOS=2
export MCCL_CHECK_POINTERS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=${PATCH_HOME}/../Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:$PYTHONPATH

# --- Validate key paths early ---
echo "=== [32B] Path validation ==="
MISSING=0
for d in "$WORK_HOME" "$PATCH_HOME" "$MEGATRON_PATH"; do
    if [ ! -d "$d" ]; then echo "ERROR: directory not found: $d"; MISSING=1; fi
done
if [ ! -f "$WORK_HOME/pretrain_gpt.py" ]; then
    echo "ERROR: pretrain_gpt.py not found in $WORK_HOME"; MISSING=1
fi
if [ ! -f "$HOSTFILE" ]; then
    echo "ERROR: hostfile not found: $HOSTFILE"; MISSING=1
fi
if [ "$MISSING" = "1" ]; then echo "Aborting."; exit 1; fi
echo "All paths OK."
echo "  WORK_HOME=$WORK_HOME"
echo "  PATCH_HOME=$PATCH_HOME"
echo "  MEGATRON_PATH=$MEGATRON_PATH"
echo "  HOSTFILE=$HOSTFILE"
echo "========================"

if [ ! -d "${MEGATRON_PATH}/build" ]; then
    cd "${MEGATRON_PATH}"
    python setup.py build_ext --inplace
    cd -
fi

CHECKPOINT_PATH=$WORK_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
DATA_PATH=${DATA_DIR:-$WORK_HOME/megatron-dataset_text_document}

LOG_PATH=$WORK_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$WORK_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH

export NODE_ADDR=$(ip a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n1 | cut -d '/' -f1)
export GPUS_PER_NODE=8
export NUM_NODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=14392

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --log_dir $WORK_HOME/output_log/$RDZV_ID/$EXPNAME
    --redirects ${LOG_REDIRECTS_LEVEL:-3}
)

# Qwen3-32B: 64 layers, hidden=5120, 64 heads, 8 KV heads, head_dim=128.
# hidden_size/num_heads = 80, but head_dim=128, so --kv-channels 128 is required.
MODEL_ARGS=(
    --num-layers 64
    --hidden-size 5120
    --ffn-hidden-size 25600
    --num-attention-heads 64
    --kv-channels 128
    --group-query-attention
    --num-query-groups 8
    --seq-length 4096
    --max-position-embeddings 40960
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --disable-bias-linear
    --position-embedding-type rope
    --no-position-embedding
    --swiglu
    --normalization RMSNorm
    --untie-embeddings-and-output-weights
)

TRAIN_SAMPLES=${TRAIN_SAMPLES:-244140}

TRAINING_ARGS=(
    --seed 42
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --init-method-std 0.008
    --use-mcore-models
    --no-bias-dropout-fusion
    --no-bias-swiglu-fusion
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 0
    --distributed-backend nccl
    --eod-mask-loss
)

REGULARIZATION_ARGS=(
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

WARMUP_STEPS=${WARMUP_STEPS:-200}
WARMUP_SAMPLES=$((WARMUP_STEPS * GLOBAL_BATCH_SIZE))

LEARNING_RATE_ARGS=(
    --lr ${LR:-3e-6}
    --lr-decay-style cosine
    --lr-warmup-samples ${WARMUP_SAMPLES}
    --min-lr ${MIN_LR:-3e-7}
    --initial-loss-scale 65536
    --min-loss-scale 1.0
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
)
if [ "$PP_SIZE" -gt 1 ]; then
    MODEL_PARALLEL_ARGS+=(--decoder-last-pipeline-num-layers 32)
fi

MIXED_PRECISION_ARGS=(
    --bf16
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --accumulate-allreduce-grads-in-fp32
)

if [ "${MOCK_DATA:-0}" = "1" ]; then
    DATA_ARGS="--mock-data --split 1 --tokenizer-type NullTokenizer --vocab-size 151936"
else
    DATA_ARGS="
        --data-path $DATA_PATH \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZED_MODEL} \
        --split ${DATA_SPLIT:-98,2,0}
    "
fi

if [ "${MOCK_DATA:-0}" = "1" ]; then
    TRAINING_ARGS+=(--num-workers "${DATA_NUM_WORKERS:-0}")
elif [ -n "${DATA_NUM_WORKERS:-}" ]; then
    TRAINING_ARGS+=(--num-workers "${DATA_NUM_WORKERS}")
fi

LOAD_PATH="${PRETRAINED_CHECKPOINT:-$CHECKPOINT_PATH}"

if [ -n "${PRETRAINED_CHECKPOINT:-}" ]; then
    TRAINING_ARGS+=(--finetune)
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval ${SAVE_INTERVAL:-10000}
    --eval-interval ${EVAL_INTERVAL:-1000}
    --save $CHECKPOINT_PATH
    --load $LOAD_PATH
    --eval-iters ${EVAL_ITERS:-10}
    --tensorboard-dir $TB_PATH
)

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --fp8-format hybrid
    --fp8-param-gather
)

cmd="torchrun ${DISTRIBUTED_ARGS[@]} $WORK_HOME/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]} \
        ${TRANSFORMER_ENGINE_ARGS[@]}
    "
echo $cmd
eval $cmd
