from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="/data05/wuxinrui/LLaMA-Factory/1_5B_TCMv2_long_short_loss/1_5B_TCMv2_long_short_loss_em_lm/models", help="模型名称或路径")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
special_token = "\n<remaining>100</remaining>\n"
normal_token = "\n<remaining>50</remaining>\n"  # 对比普通词

# 获取 Embedding
special_id = tokenizer.encode(special_token, add_special_tokens=False)[0]
normal_id = tokenizer.encode(normal_token, add_special_tokens=False)[0]
special_emb = model.get_input_embeddings().weight[special_id]
normal_emb = model.get_input_embeddings().weight[normal_id]

# 计算相似度
cos_sim = torch.cosine_similarity(special_emb, normal_emb, dim=0)
print(f"两词之间相似度: {cos_sim.item():.3f}")
print(f"normal_emb = {normal_emb}")