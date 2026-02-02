#!/bin/bash

# Qwen3 VL RL training on geo3k dataset
# Supports both megatron and fsdp training backends
# Usage: 
#   SLIME_SCRIPT_TRAIN_BACKEND=fsdp ./run_geo3k_vlm.sh
#   SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-30B-A3B-Thinking ./run_geo3k_vlm.sh
export PATH=$CONDA_PREFIX/bin:$PATH
# echo "DEBUG: PATH for Ray is: $PATH"
# PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
export LD_LIBRARY_PATH=/home/hanfeng.wxt/micromamba/envs/slime_pjh_v3_backkup/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY=1526cd13c8d1f8c8529ea57f23d553b20b03451c # set your wandb api key
# export CUDA_VISIBLE_DEVICES=5,6
# Configuration
TRAIN_BACKEND=${SLIME_SCRIPT_TRAIN_BACKEND:-"megatron"}
# MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-8B-Thinking"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-30B-A3B-Thinking"}
MODEL_ROOT=${SLIME_SCRIPT_MODEL_ROOT:-"/mnt/workspace/users/xintong/pjh/models"}
DATASET_NAME=${SLIME_SCRIPT_DATASET_NAME:-"chenhegu/geo3k_imgurl"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")
DATASET_ROOT=${SLIME_SCRIPT_DATASET_ROOT:-"/mnt/workspace/users/xintong/pjh/datasets"}
OUTPUT_DIR=${SLIME_SCRIPT_OUTPUT_DIR:-"/mnt/workspace/users/xintong/pjh/All_results/slime"}
OUT_NAME=${SLIME_SCRIPT_OUTPUT_NAME:-"geo3k_test_v3_backkup_"${MODEL_NAME}}


# Validate MODEL_NAME
VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen2.5-VL-32B-Instruct
  Qwen2.5-VL-72B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
  Qwen3-VL-235B-A22B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
  Qwen3-VL-30B-A3B-Thinking
  Qwen3-VL-30B-A3B-Thinking-FP8
  Qwen3-VL-235B-A22B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

# External Ray flag
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# Cleanup
pkill -9 sglang
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force
   pkill -9 ray
fi
pkill -9 slime
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray
fi
pkill -9 slime
pkill -9 redis

set -ex

export PYTHONBUFFERED=16

# Log to file (set SLIME_SCRIPT_LOG=0 to disable)
if [ "${SLIME_SCRIPT_LOG:-1}" = "1" ]; then
   LOG_DIR=${OUTPUT_DIR}/${OUT_NAME}/logs
   mkdir -p "${LOG_DIR}"
   LOG_FILE=${SLIME_SCRIPT_LOG_FILE:-"${LOG_DIR}/geo3k_vlm_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"}
   exec > >(tee -a "${LOG_FILE}") 2>&1
fi

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


# Common args
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_ROOT}/${MODEL_NAME}
   # qwen3 vl model has rotary base 5000000, set it when applicable, moe is 10000000
   --rotary-base 5000000
   --save ${OUTPUT_DIR}/${OUT_NAME}
   # 模型保存间隔（步数）
   --save-interval 10
   --save-debug-rollout-data ${OUTPUT_DIR}/${OUT_NAME}/debug_rollout/rollout_{rollout_id}.pt
)

ROLLOUT_ARGS=(
   --prompt-data ${DATASET_ROOT}/${DATASET_LOCAL_NAME}/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 35
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 512
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data test_geo3k ${DATASET_ROOT}/${DATASET_LOCAL_NAME}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # --sglang-cuda-graph-max-bs 16
   --sglang-ep-size 1
   # --sglang-cuda-graph-bs 1 2 4 8 16 24 32
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
   # --sglang-moe-a2a-backend deepep
   # --sglang-deepep-mode auto
)

# Wandb args (only if WANDB_API_KEY is set)
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime-geo3k-vlm-ali
      --wandb-group ${MODEL_NAME_LOWER}-${TRAIN_BACKEND}
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi

MISC_ARGS=(
   --colocate
)

# cat /mnt/workspace/users/xintong/pjh/models/Qwen3-VL-30B-A3B-Thinking/config.json | grep "num_key_value_heads" 结果是4导致这里的tp不能超过kv头数（4）
# --load ${MODEL_ROOT}/${MODEL_NAME}
BACKEND_ARGS=(
   --train-backend megatron
   --load ${MODEL_ROOT}/${MODEL_NAME}
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)
   
# get MODEL_ARGS from scripts/models for megatron backend
# SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SLIME_DIR="/mnt/workspace/users/xintong/pjh/slime"
# MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/-FP8//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
# VL models require rotary-base 5000000
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"
# MODEL_ARGS_ROTARY_BASE=5000000 source "/export/home/pan/slime/scripts/run-qwen3-30B-A3B.sh"
   

# Start Ray if not using external Ray
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/mnt/workspace/users/xintong/pjh/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
   #  \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\",
   #  \"PATH\": \"${PATH}\",
   #  \"TRITON_PTXAS_PATH\": \"${PTXAS_PATH}\"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
