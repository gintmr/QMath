
# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="deepseek3"
# PROMPT_TYPE="qwen25-math-cot"
# Qwen2.5-Math-1.5B-Instruct
DATA_NAME="math500"

# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export remaining=default

bash delete_file.sh ./start_positions.pt
bash delete_file.sh ./early_positions.pt


MODEL_NAME_OR_PATH='/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_FULL_1_5B_withoutremaining_v2/TCMv4_FULL_1_5B_withoutremaining_v2_6epoch/models'
PARENT_DIR=$(dirname "$MODEL_NAME_OR_PATH")  # 获取父目录
MODEL_NAME=$(basename "$PARENT_DIR")        # 获取父目录的最后一部分
echo MODEL_NAME: $MODEL_NAME
export PE_MODE=default
export position=ori
export tip=withoutremaining
export stage=2
export mode=TIP-$tip-STAGE-$stage
export model=MODEL-$MODEL_NAME
export modelname=MODEL-$MODEL_NAME-TIP-$tip-STAGE-$stage-DATA-$DATA_NAME
bash ./sh/remaining_length_metric.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME