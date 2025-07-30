# 两阶段推理测试


set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
DATA_NAME=$3


SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
export DATA_NAME=$DATA_NAME

for tokens in 50 100 250 500 750 1000 1500 2000 3000 4000

do
    echo "max_tokens_per_call: $tokens \n"
    export BUDGET=$tokens
    echo "export BUDGET=$tokens \n"
    TOKENIZERS_PARALLELISM=true \
    # python3 -u ./remaining_eval_multi_process.py \
    python3 -u remaining_eval_multi_process_RL_QA.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_names ${DATA_NAME} \
        --output_dir ./$modelname/$tokens \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --num_test_sample 12000\
        --use_safetensors \
        --save_outputs \
        --use_vllm \
        --overwrite \
        --max_tokens_per_call $tokens 
done