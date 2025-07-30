import os
import re
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
# from .Chain_Of_Thought_Metric.tools import *

# 父文件夹路径
parent_folder = "/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation"

# pattern = re.compile(r"MODEL-.*-TIP-.*-STAGE-\d+")
pattern = re.compile(r"MODEL-.*-TIP-.*-STAGE-add-DATA-.*")

entry_list = []
setting_names = []
# tokenizer = AutoTokenizer.from_pretrained("/data05/wuxinrui/LLaMA-Factory/DS-distill-QWen-1_5B_TCM/models", trust_remote_code=True)

for folder in os.listdir(parent_folder):
    # if pattern.match(folder):
    if folder == "MODEL-TCMv4_ablation_v1_step_77_reward_0.824-TIP-withoutremaining-STAGE-2-DATA-math500":
        entry_list.append(os.path.join(parent_folder, folder))
        setting_names.append(folder)

def is_balanced(s: str) -> bool:
    """验证大括号是否成对且正确嵌套"""
    stack = 0
    for char in s:
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
            if stack < 0:
                return False
    return stack == 0

        
def find_sequence(array, sequence):
    sequence_length = len(sequence)
    array_length = len(array)
    start_positions = []

    for i in range(array_length - sequence_length + 1):
        if array[i:i+sequence_length] == sequence:
            start_positions.append(i)

    return start_positions


def get_final_answer_status(anwser):
    find_string = "**Final Answer**\\boxed"
    # find_string = "**Final Answer**"
    sequence = [334, 19357, 21806, 334, 59, 79075]

    Thought_END = 0
    START = 0
    END = 0
    if not find_string in anwser:
        return Thought_END, START, END
    
    else:
        Thought_END = 1
        
        # #G ---------------- 匹配正反大括号
        # start_idx = anwser.find('**Final Answer**\\boxed{')
        # stack = 1
        # end_idx = start_idx + len('**Final Answer**\\boxed{')
        # while end_idx < len(anwser) and stack > 0:
        #     if anwser[end_idx] == "{":
        #         stack += 1
        #     elif anwser[end_idx] == "}":
        #         stack -= 1
        #     end_idx += 1
        # if stack == 0 and is_balanced(anwser[start_idx:end_idx]):
        #     Thought_END = 1
        #     # tokenized_answer = tokenizer(anwser)['input_ids']
        #     # #G 找出结束时的token位置
        #     # start_positions = find_sequence(tokenized_answer, sequence)
        # else:
        #     Thought_END = 0
        #G -----------------
            
        return Thought_END, START, END
           
def calculate_ratios(thinking_ended, answer_correct):
    # 初始化计数器
    ET_count = 0
    total_count = len(thinking_ended)
    EF_count = 0
    OT_count = 0

    # 遍历列表
    for i in range(total_count):
        if thinking_ended[i] == 1 and answer_correct[i] == 1:
            ET_count += 1
        if thinking_ended[i] == 1 and answer_correct[i] == 0:
            EF_count += 1
        if thinking_ended[i] == 0 and answer_correct[i] == 1:
            OT_count += 1

    # 计算比值
    ET_ratio = ET_count / total_count if total_count > 0 else 0
    EF_ratio = EF_count / total_count if total_count > 0 else 0
    OT_ratio = OT_count / total_count if total_count > 0 else 0

    acc = sum(thinking_ended) / total_count if total_count > 0 else 0
    
    return ET_ratio, EF_ratio, OT_ratio, acc

if __name__ == "__main__":

    for entry, setting_name in zip(entry_list, setting_names):
        CoT_metric_one_setting = []
        for sub_entry in os.listdir(entry):
            if not os.path.isdir(os.path.join(entry, sub_entry)):
                continue
        #G 对于每一个测试配置对应文件夹
            for root, dirs, files in os.walk(os.path.join(entry, sub_entry)):
            #g 遍历不同长度结果
                for file in files:
                    if "metrics" not in file:
                        cot_answer_path = file
                        cot_answer_path = os.path.join(root, cot_answer_path)
                        #G 找到所有存储回答的jsonl文件
            with open(cot_answer_path, "r") as f:
                CoT_metric_single_budget = {}
                budget_length = int(sub_entry)
                CoT_metric_single_budget['budget_length'] = budget_length
                True_list = []
                End_list = []
                for line in f:
                    data = json.loads(line)
                    if data['score'][0] == False:
                        True_list.append(0)
                    else:
                        True_list.append(1)
                    
                    Thought_END, START, END = get_final_answer_status(data['code'][0])
                    End_list.append(Thought_END)
                    #G 获取每个回答的最终状态
                    
                ET_ratio, EF_ratio, OT_ratio, acc = calculate_ratios(End_list, True_list)
                
                CoT_metric_single_budget['ET_ratio'] = ET_ratio
                CoT_metric_single_budget['EF_ratio'] = EF_ratio
                CoT_metric_single_budget['OT_ratio'] = OT_ratio
                CoT_metric_single_budget['acc'] = acc
                
                CoT_metric_one_setting.append(CoT_metric_single_budget)
        CoT_metric_one_setting.sort(key=lambda x: x['budget_length'])
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 定义指标名称和对应的索引
        metrics = ['ET_ratio', 'EF_ratio', 'OT_ratio', 'acc']
        titles = ['ET Ratio', 'EF Ratio', 'OT Ratio', 'Accuracy']
        colors = ['blue', 'green', 'red', 'purple']
        main_title = setting_name
        for i, metric in enumerate(metrics):
            ax = axs[i // 2, i % 2]
            budget_lengths = [data['budget_length'] for data in CoT_metric_one_setting]
            metric_values = [data[metric] for data in CoT_metric_one_setting]
            ax.plot(budget_lengths, metric_values, marker='o', color=colors[i], linewidth=2)
            ax.set_title(titles[i])
            ax.set_xlabel('Budget Length')
            ax.set_ylabel(metric)
            ax.grid(True)
        fig.suptitle(main_title, fontsize=16, y=1)

        plt.tight_layout()
        pic_name = os.path.join(entry, "metric.png")
        plt.savefig(os.path.join(entry, pic_name), dpi=300)
        print(f"Saved figure as {pic_name}")
        plt.close()
