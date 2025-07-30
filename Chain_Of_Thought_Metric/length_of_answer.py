import os
import re
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt

# 父文件夹路径
parent_folder = "/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation"
pattern = re.compile(r"MODEL-.*-TIP-.*-STAGE-add-DATA-.*")

entry_list = []
setting_names = []
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)  # 使用合适的tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio/1_5B_TCMv4_8ratio_models/models", trust_remote_code=True)

def calculate_token_length(text):
    """计算文本的token长度"""
    tokens = tokenizer(text)['input_ids']
    return len(tokens)

if __name__ == "__main__":
    for folder in os.listdir(parent_folder):
        # if pattern.match(folder):
        if folder == "MODEL-TCMv4_8ratio_v1_step_77_reward_0.832-TIP-8ratio-STAGE-2-DATA-math500":
            entry_list.append(os.path.join(parent_folder, folder))
            setting_names.append(folder)

    for entry, setting_name in zip(entry_list, setting_names):
        token_length_metrics = []
        
        for sub_entry in os.listdir(entry):
            if not os.path.isdir(os.path.join(entry, sub_entry)):
                continue
                
            for root, dirs, files in os.walk(os.path.join(entry, sub_entry)):
                for file in files:
                    if "metrics" in file:
                        continue
                        
                    cot_answer_path = os.path.join(root, file)
                    
                    with open(cot_answer_path, "r") as f:
                        token_length_data = {}
                        budget_length = int(sub_entry)
                        token_length_data['budget_length'] = budget_length
                        total_tokens = 0
                        answer_count = 0
                        
                        for line in f:
                            data = json.loads(line)
                            answer_text = data['code'][0]
                            token_length = calculate_token_length(answer_text)
                            total_tokens += token_length
                            answer_count += 1
                        
                        avg_token_length = total_tokens / answer_count if answer_count > 0 else 0
                        token_length_data['avg_token_length'] = avg_token_length
                        token_length_data['total_tokens'] = total_tokens
                        token_length_metrics.append(token_length_data)
        
        # 按budget_length排序
        token_length_metrics.sort(key=lambda x: x['budget_length'])
        
        # 绘制图表
        fig, ax = plt.subplots(figsize=(10, 6))
        budget_lengths = [data['budget_length'] for data in token_length_metrics]
        avg_lengths = [data['avg_token_length'] for data in token_length_metrics]
        
        ax.plot(budget_lengths, avg_lengths, marker='o', color='blue', linewidth=2)
        ax.set_title(f"Average Token Length by Budget Length - {setting_name}")
        ax.set_xlabel('Budget Length')
        ax.set_ylabel('Average Token Length')
        ax.grid(True)
        
        plt.tight_layout()
        pic_name = os.path.join(entry, "token_length_metrics.png")
        plt.savefig(pic_name, dpi=300)
        print(f"Saved figure as {pic_name}")
        plt.close()