#!/bin/bash

PORT=8235
TP=2
MAX_MODEL_LEN=262144

DO_NSYS=0


MODEL_DIR=/mnt/disk0/huanghaoyan.hhy/Qwen3-Next-80B-A3B-Instruct/
echo "MODEL_DIR: $MODEL_DIR"

env_vars=(
    "CUDA_VISIBLE_DEVICES=4,5,6,7"
    "CUDA_LAUNCH_BLOCKING=1"
    "TORCH_USE_CUDA_DSA=1"
    # "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    # "TRITON_INTERPRET=1"
    # "CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1"
    # "CUDA_COREDUMP_SHOW_PROGRESS=1"
    # "CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'"
    # "CUDA_COREDUMP_FILE=./cuda_coredump_%h.%p.%t"
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
    # --gpu-memory-utilization 0.928
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
