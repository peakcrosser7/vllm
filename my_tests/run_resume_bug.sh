#!/bin/bash

PORT=8235
TP=2
MAX_MODEL_LEN=262144

DO_NSYS=0


MODEL_DIR=/mnt/disk0/huanghaoyan.hhy/Qwen3-Next-80B-A3B-Instruct/
echo "MODEL_DIR: $MODEL_DIR"

env_vars=(
    # "CUDA_LAUNCH_BLOCKING=0"
)

for var in "${env_vars[@]}"; do
    var_name="${var%%=*}"
    var_value="${var#*=}"
    echo -e "\t$var_name=$var_value"
done

CMD=( env )
for var in "${env_vars[@]}"; do
    CMD+=( "$var" )
done
CMD+=(
    $NSYS vllm serve
    $MODEL_DIR
    # --trust-remote-code
    --port "$PORT"
    # use 97GB H20
    --gpu-memory-utilization 0.82    # 2kchunk mtp3
    -tp $TP
    --enforce-eager
    
    # --no-enable-prefix-caching
    --enable-prefix-caching
    # --no-enable-chunked-prefill
    --enable-chunked-prefill
    # --mamba-cache-mode all
    --max-num-batched-tokens 2048
    --distributed-executor-backend mp
    --block-size 64
    --max-num-seqs 128
    --speculative-config "{\"method\": \"qwen3_next_mtp\", \"num_speculative_tokens\": 3}"
    --hf_overrides "{\"max_position_embeddings\": $MAX_MODEL_LEN}"
)

echo -e "\nExecuting command:"
printf " %s" "${CMD[@]}"
echo -e "\n"

"${CMD[@]}"
