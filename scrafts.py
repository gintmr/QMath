import json

with open("/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation/MODEL-FULL7B_SFT-TIP-TCMv2-STAGE-1-DATA-math500/100/outputs_full/deepseek3/math500/test_deepseek3_5000_seed0_t0.0_s0_e-1_b100_original.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        print(data["score"] == [True])
        print(data["code"][0])