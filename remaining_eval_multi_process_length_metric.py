import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_tools import apply_RL_prompt, solve_final_answer

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import logging
## 启动logging功能
if not os.path.exists(f'{os.environ["modelname"]}'):
    os.mkdir(f'{os.environ["modelname"]}')
if not os.path.exists(f'{os.environ["model"]}'):
    os.mkdir(f'{os.environ["model"]}')

DATA_NAME = os.environ["DATA_NAME"]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', filename=f'{os.environ["model"]}/{os.environ["mode"]}-{DATA_NAME}-Length_Metric.log', filemode='a')
print(f"logging in {os.environ['model']}/{os.environ['mode']}-{DATA_NAME}-Length_Metric.log")

logging.info(f"modelname's infor:  {os.environ['modelname']}")
logging.info(f"mode's infor:  {os.environ['mode']}")
logging.info(f"model's infor:  {os.environ['model']}")

with open('./special_tokens.json') as f:
        special_tokens = json.load(f)

bins_tokens = [
    special_tokens[f"{i}"] for i in range(400)
]


import os
import multiprocessing
from multiprocessing import Process, Manager, Queue
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import fcntl
import torch

def process_chunk(chunk, args, available_gpus, stop_words, output_queue, index, two_stage=False):
    """
    处理一个数据块的函数
    """
    GPU_LOCK_FILES = {gpu_id: f"/tmp/gpu_{gpu_id}.lock" for gpu_id in available_gpus}
    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[index % len(available_gpus)])
    gpu_id = available_gpus[index % len(available_gpus)]
    lock_file = f"/tmp/gpu_{gpu_id}.lock"  # 每个 GPU 对应一个锁文件
    
    

    # 获取文件锁（排他锁）
    with open(lock_file, 'w') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # 获取排他锁，其他进程会阻塞直到锁释放

        try:
            # 设置当前进程使用的 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # 初始化 LLM（确保在锁内执行，避免多个进程同时初始化）
            llm = LLM(
                model=args.model_name_or_path,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                max_seq_len_to_capture=65536,
            )

            chunk_outputs = []
            for i in range(0, len(chunk), 125):
                chunk_batch = chunk[i:i + 125]
                if args.use_vllm:
                    if os.environ['stage'] == "add":
                        budget = args.max_tokens_per_call + (args.max_tokens_per_call // 50) + 50
                    else:
                        budget = args.max_tokens_per_call + (args.max_tokens_per_call // 50)
                    os.environ["position"] = 'start'
                    chunk_batch_outputs = llm.generate(
                        chunk_batch,
                        SamplingParams(
                            temperature=args.temperature,
                            top_p=0.9,
                            max_tokens=int(budget*1.3),
                            n=1,
                            stop=stop_words,
                            stop_token_ids=(
                                [151645, 151643]
                                if "qwen2" in args.model_name_or_path.lower()
                                else None
                            ),
                            skip_special_tokens=False,
                        ),
                    )
                    if os.path.exists('./start_positions.pt'):
                        os.remove('./start_positions.pt')
                    if os.path.exists('./early_positions.pt'):
                        os.remove('./early_positions.pt')

                    chunk_batch_outputs = sorted(chunk_batch_outputs, key=lambda x: int(x.request_id))
                    chunk_batch_outputs = [output.outputs[0].text for output in chunk_batch_outputs]
                    batch_chunk = [single_chunk + chunk_output for single_chunk, chunk_output in zip(chunk_batch, chunk_batch_outputs)]
                    chunk_outputs.extend(batch_chunk)

            # 处理完成后，将结果存入队列
            output_queue.put((index, chunk_outputs))

        finally:
            # 显式销毁 llm 对象
            if 'llm' in locals():
                del llm  # 确保 LLM 引擎被销毁
            # torch.cuda.empty_cache()  # 清理 GPU 内存
            # 释放锁（确保即使发生异常也能释放）
            fcntl.flock(f, fcntl.LOCK_UN)


def clean_code(code):
    for bin_token in bins_tokens:
        if bin_token in code:
            code = code.replace(bin_token, "")
    return code
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=-1, help="ratio of cot to use for generation")
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen/QwQ-32B-Preview", type=str)
    parser.add_argument("--output_dir", default="Qwen/QwQ-32B-Preview/math_eval", type=str)
    parser.add_argument("--prompt_type", default="qwen25-math-cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.",)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--adapt_few_shot", action="store_true", help="Few shot for multiple-choice questions, zero shot for others.",)
    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling (vllm)
    # if args.ratio > 0:
    #     args.max_tokens_per_call = 50
    return args

def set_output_path(args, data_name):
    # args.output_dir defines experiment path,such as outputs/12_25
    model_name_list = args.model_name_or_path.split('/')[-1]
    model_name = model_name_list
    for part in model_name_list:
        if 'models' in part:
            model_name = part
    
    # print(f"args.output_dir: {args.output_dir}")
    # print(f"model_name: {model_name}")
    # print(f"args.prompt_type: {args.prompt_type}")
    output_dir = os.path.join(args.output_dir, model_name, args.prompt_type)
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_b{int(args.max_tokens_per_call)}_original.jsonl"
    print(out_file)
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
    return out_file_prefix, output_dir, out_file


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset， -1 for full data
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    
    # get out_file_prefix, output_dir and out_file
    out_file_prefix, output_dir, out_file = set_output_path(args, data_name)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        # llm = LLM(
        #     model=args.model_name_or_path,
        #     tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        #     pipeline_parallel_size=args.pipeline_parallel_size,
        #     trust_remote_code=True,
        #     gpu_memory_utilization=0.85,
        #     enforce_eager=True,
        #     max_seq_len_to_capture=5000000,
        #     # enable_flash_attn=True
        # )
        llm = None
        tokenizer = None
        # if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True, max_length=16000,
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.extend(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results

    answer_len_list, length_reward_list = [], []
    for result in results:
        answer_len, length_reward = result[0], result[1]
        answer_len_list.extend([answer_len])
        length_reward_list.extend([length_reward])
    
    print(f"len(answer_len_list): {len(answer_len_list)}")
    
    avg_answer_len = sum(answer_len_list) / len(answer_len_list)
    avg_length_reward = sum(length_reward_list) / len(length_reward_list)
        
        
    results ={
            "avg_answer_len": avg_answer_len,
            "avg_length_reward": avg_length_reward,
        }
    print(f"!!!!!!!!!!!!\n"*10)
    print(f"results: {results}")
    # print all results

    logging.info(f"!!!!!!!!!!!!\n"*10)
    logging.info(f"results: {results}")
    return results

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print(examples[0])
    print("\n" + "-" * 50)
    print("data:", data_name, ", remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    print(f"available_gpus = {available_gpus}")

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # load done samples
    if args.ratio > 0 :
        done_samples_path = out_file.replace("_r" + str(args.ratio), "")
        done_samples = list(load_jsonl(done_samples_path))
    else:
        done_samples = []
    done_samples = {sample["idx"]: sample for sample in done_samples}
    
    samples = []
    print("\nProcessing", len(examples), "examples", "=" * 50)
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)
        # # add ratio part of complete cot
        if args.ratio > 0 :
            done_cot = done_samples[idx]["code"][0]
            cut_cot = done_cot[:int(len(done_cot)*args.ratio)]
            # # 将prompt中的<|im_start|>assistant\n换成新内容
            # full_prompt = full_prompt.replace("<|im_start|>assistant\n", "<|im_start|>assistant\n" + cut_cot + "\n\nFinal answer within \\boxed{{}}:\n")
            # 直接在prompt的后面添加新内容
            full_prompt = full_prompt + cut_cot + "\n\nFinal answer within \\boxed{{}}:\n"

        

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    input_prompts = apply_RL_prompt(input_prompts, args, budget = args.max_tokens_per_call)
    
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal", "qwen25-math-cot"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    # start inference
    # measure time use
    start_time = time.time()
    print(f"start_time: {start_time}")
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        prompts = [item[1] for item in current_prompts]
        # prompts = apply_RL_prompt(prompts, args, budget = args.max_tokens_per_call)

        num_prompts = len(prompts)
        chunk_size = 256
        #(num_prompts + 4) // 5  # 确保包含所有的 prompts
        
        outputs = []
        if os.environ['tip'] == "remaining" or os.environ['tip'] == "ATD_R":
            for i in range(0, num_prompts, chunk_size):
                # print(prompts[i])
                chunk = prompts[i:i + chunk_size]  # 获取当前的 chunk
                
                if args.use_vllm:
                    
                    budget  = args.max_tokens_per_call
                    i = 0
                    while 50*(2**i) < budget:
                        i += 1
                    i -= 1
                    for k in range(i, -2, -1):
                        stop_budget = budget - 50*(2**k) if k >= 0 else 50
                        # print(f"stop_budget: {stop_budget}")
                        # chunk = [data + f"\n<remaining>[{budget} token]</remaining>\n" for data in chunk]
                        if budget == args.max_tokens_per_call:
                            chunk = chunk
                        else:
                            # "<｜end▁of▁sentence｜>"
                            
                            chunk = [data + f"\n<remaining>{budget}</remaining>\n" if "<｜end▁of▁sentence｜>" not in data else data for data in chunk]
                        print(f"chunk0: {chunk[0]}")

                        if stop_budget > 0:
                            chunk_outputs = llm.generate(
                                chunk,
                                SamplingParams(
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_tokens=stop_budget,
                                    n=1,
                                    stop=stop_words,
                                    stop_token_ids=(
                                        [151645, 151643]
                                        if "qwen2" in args.model_name_or_path.lower()
                                        else None
                                    ),
                                    skip_special_tokens=False, #G 设置特殊token的可见性
                                ),
                            )
                            if os.path.exists('./start_positions.pt'):
                                os.remove('./start_positions.pt')
                                print('start_positions.pt removed')
                            if os.path.exists('./early_positions.pt'):
                                os.remove('./early_positions.pt')
                                print('early_positions.pt removed')
                            chunk_outputs = sorted(chunk_outputs, key=lambda x: int(x.request_id))
                            chunk_outputs = [output.outputs[0].text for output in chunk_outputs]
                            ### 输出已经被整理过了，不需要再进行排序
                            chunk = [single_chunk + chunk_output for single_chunk, chunk_output in zip(chunk, chunk_outputs)]
                            budget = 50*(2**k) if k >= 0 else 0
                            chunk, end_chunk, open_chunk = solve_final_answer(chunk)
                            print(f"len of end_chunk: {len(end_chunk)}")
                            print(f"len of open_chunk: {len(open_chunk)}")
                            # outputs.extend(end_chunk)
                            # chunk = open_chunk
                            # print(f"now budget: {budget}")
                            # print(f"k = {k}")     
                    chunk_outputs = chunk    
                    outputs.extend(chunk_outputs) 
                    # chunk_outputs = sorted(chunk, key=lambda x: int(x.request_id))
                    # initial_outputs = [output.outputs[0].text for output in chunk_outputs]
                    
                    # Add the think/final answer tags and create new prompts
                    # modified_outputs = []
                    # for output in chunk_outputs:
                    #     modified_output = output.rstrip() + "</think>\n\n**Final Answer**\n\\boxed"
                    #     modified_outputs.append(modified_output)
                    
                    # # Second generation with modified outputs
                    # second_prompts = [p + mo for p, mo in zip(chunk, modified_outputs)]
                    # second_outputs = llm.generate(
                    #     second_prompts,
                    #     SamplingParams(
                    #         temperature=args.temperature,
                    #         top_p=args.top_p,
                    #         max_tokens=20,
                    #         n=1,
                    #         stop=stop_words,
                    #         stop_token_ids=(
                    #             [151645, 151643]
                    #             if "qwen2" in args.model_name_or_path.lower()
                    #             else None
                    #         ),
                    #     ),
                    # )
                    
                    
                    # if os.path.exists('./start_positions.pt'):
                    #     os.remove('./start_positions.pt')
                    #     print('start_positions.pt removed')
                    # if os.path.exists('./early_positions.pt'):
                    #     os.remove('./early_positions.pt')
                    #     print('early_positions.pt removed')
                    
                    # second_outputs = sorted(second_outputs, key=lambda x: int(x.request_id))
                    # second_outputs = [output.outputs[0].text for output in second_outputs]
                    
                    # # Combine initial and second outputs
                    # combined_outputs = [init + "</think>\n\n**Final Answer**\n\\boxed" + second for init, second in zip(second_prompts, second_outputs)]
                    # outputs.extend(combined_outputs)                
                else:
                    # Similar modification for non-vllm case
                    chunk_outputs = generate_completions(
                        model=llm,
                        tokenizer=tokenizer,
                        prompts=chunk,
                        max_new_tokens=args.max_tokens_per_call,
                        batch_size=16,
                        stop_id_sequences=stop_words,
                    )
                    
                    # Add the think/final answer tags and create new prompts
                    modified_outputs = []
                    for output in chunk_outputs:
                        modified_output = output.rstrip() + "\n</think>\n\n**Final Answer**\n\\boxed"
                        modified_outputs.append(modified_output)
                    
                    # Second generation with modified outputs
                    second_prompts = [p + mo for p, mo in zip(chunk, modified_outputs)]
                    second_outputs = generate_completions(
                        model=llm,
                        tokenizer=tokenizer,
                        prompts=second_prompts,
                        max_new_tokens=args.max_tokens_per_call,
                        batch_size=16,
                        stop_id_sequences=stop_words,
                    )
                    
                    # Combine initial and second outputs
                    combined_outputs = [init + second for init, second in zip(chunk_outputs, second_outputs)]
                    outputs.extend(combined_outputs)
                    
                    
        elif os.environ["tip"] == "TCM":
            for i in range(0, num_prompts, chunk_size):
                chunk = prompts[i:i + chunk_size]  # 获取当前的 chunk
                if args.use_vllm:
                    budget  = args.max_tokens_per_call
                    i = budget // 50 + 1
                    for k in reversed(range(i)):
                        stop_budget = budget - 50 * k

                        if budget == args.max_tokens_per_call:
                            chunk = chunk
                        else:
                            chunk = [data + f"\n<remaining>{budget}</remaining>\n" if "<｜end▁of▁sentence｜>" not in data else data for data in chunk]
                        print(f"chunk0: {chunk[0]}")
                        if stop_budget > 0:
                            chunk_outputs = llm.generate(
                                chunk,
                                SamplingParams(
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_tokens=stop_budget,
                                    n=1,
                                    stop=stop_words,
                                    stop_token_ids=(
                                        [151645, 151643]
                                        if "qwen2" in args.model_name_or_path.lower()
                                        else None
                                    ),
                                    skip_special_tokens=False, #G 设置特殊token的可见性
                                ),
                            )
                            if os.path.exists('./start_positions.pt'):
                                os.remove('./start_positions.pt')
                                print('start_positions.pt removed')
                            if os.path.exists('./early_positions.pt'):
                                os.remove('./early_positions.pt')
                                print('early_positions.pt removed')
                                
                            chunk_outputs = sorted(chunk_outputs, key=lambda x: int(x.request_id))
                            chunk_outputs = [output.outputs[0].text for output in chunk_outputs]
                            ### 输出已经被整理过了，不需要再进行排序
                            chunk = [single_chunk + chunk_output for single_chunk, chunk_output in zip(chunk, chunk_outputs)]
                            budget = 50 * k if k >= 0 else 0
                            chunk, end_chunk, open_chunk = solve_final_answer(chunk)
                            print(f"len of end_chunk: {len(end_chunk)}")
                            print(f"len of open_chunk: {len(open_chunk)}")
                            print(F"len of chunk: {len(chunk)}s")
                            # outputs.extend(end_chunk)
                            # chunk = open_chunk
                    chunk_outputs = chunk    
                    outputs.extend(chunk_outputs) 
                else:
                    raise(ValueError("Not implemented for non-vllm mode while tip == TCM"))
                
                
        elif os.environ["tip"] == "SST":
            for i in range(0, num_prompts, chunk_size):
                chunk = prompts[i:i + chunk_size]  # 获取当前的 chunk
                if args.use_vllm:
                    budget  = args.max_tokens_per_call
                    i = budget // 50 + 1
                    for k in reversed(range(i)):
                        stop_budget = budget - 50 * k

                        if budget == args.max_tokens_per_call:
                            chunk = chunk
                        else:
                            chunk = [data + f"\n<countdown>\n" if "<｜end▁of▁sentence｜>" not in data else data for data in chunk]
                        print(f"chunk0: {chunk[0]}")
                        if stop_budget > 0:
                            chunk_outputs = llm.generate(
                                chunk,
                                SamplingParams(
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    max_tokens=stop_budget,
                                    n=1,
                                    stop=stop_words,
                                    stop_token_ids=(
                                        [151645, 151643]
                                        if "qwen2" in args.model_name_or_path.lower()
                                        else None
                                    ),
                                ),
                            )
                            if os.path.exists('./start_positions.pt'):
                                os.remove('./start_positions.pt')
                                print('start_positions.pt removed')
                            if os.path.exists('./early_positions.pt'):
                                os.remove('./early_positions.pt')
                                print('early_positions.pt removed')
                                
                            chunk_outputs = sorted(chunk_outputs, key=lambda x: int(x.request_id))
                            chunk_outputs = [output.outputs[0].text for output in chunk_outputs]
                            ### 输出已经被整理过了，不需要再进行排序
                            chunk = [single_chunk + chunk_output for single_chunk, chunk_output in zip(chunk, chunk_outputs)]
                            budget = 50 * k if k >= 0 else 0
                            chunk, end_chunk, open_chunk = solve_final_answer(chunk)
                            print(f"len of end_chunk: {len(end_chunk)}")
                            print(f"len of open_chunk: {len(open_chunk)}")
                            print(F"len of chunk: {len(chunk)}s")
                            # outputs.extend(end_chunk)
                            # chunk = open_chunk
                    chunk_outputs = chunk    
                    outputs.extend(chunk_outputs) 
                else:
                    raise(ValueError("Not implemented for non-vllm mode while tip == TTS"))
        
        # elif os.environ["tip"] == "TCMv2":
        else:

            available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")  # 示例GPU列表，根据实际情况修改
            num_gpus = len(available_gpus)
            manager = Manager()
            output_queue = Queue()
            processes = []
            multi_outputs = []
            chunk_size = len(prompts) // num_gpus
            chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
            num_rounds = (len(chunks) + num_gpus - 1) // num_gpus #g 轮次数等于最大GPU上的任务数

            for round_idx in range(num_rounds):
                start_idx = round_idx * num_gpus
                end_idx = min((round_idx + 1) * num_gpus, len(chunks))
                
                for i in range(start_idx, end_idx):
                    chunk = chunks[i]
                    p = Process(target=process_chunk, args=(chunk, args, available_gpus, stop_words, output_queue, i))
                    processes.append(p)
                    p.start()
            # for i, chunk in enumerate(chunks):
            #     print(f"Processing chunk {i} with size {len(chunk)}")
            #     p = Process(target=process_chunk, args=(chunk, args, available_gpus, stop_words, output_queue, i))
            #     processes.append(p)
            #     p.start()
            for _ in range(len(chunks)):
                result = output_queue.get()
                if isinstance(result, tuple) and len(result) == 2:
                    multi_outputs.append(result)
                else:
                    print(f"Error: Received non-tuple result: {result}")
                # multi_outputs.extend(output_queue.get())
            for p in processes:
                p.join()
            multi_outputs.sort(key=lambda x: x[0])
            outputs = []
            for _, chunk_output in multi_outputs:
                outputs.extend(chunk_output)

        print('stage one finished!!!\n' * 20)
        # print("Special tokens in tokenizer:", tokenizer.special_tokens_map)
        # test_token = "\n<remaining>50</remaining>\n"
        # print(f"Encoding '{test_token}':", tokenizer.encode(test_token, add_special_tokens=False))
        print(outputs[:3])
        
        
        #################!
        print(f"outputs:{len(outputs)}")
        print(f"current_prompts:{len(current_prompts)}")
        assert len(outputs) == len(current_prompts)
        
        
        def get_length_reward(answer, budget):
            len_of_token = len(tokenizer.encode(answer, add_special_tokens=False))
            if (len_of_token) >= budget:
                reward = max((1 - 16 * ((len_of_token - budget) / budget) * ((len_of_token - budget) / budget)), 0)
            else:
                reward = 1

            return [len_of_token, reward]
            
        budget = args.max_tokens_per_call

        length_reward = []
        for output in outputs:
            answer = output.split('<｜Assistant｜>')[-1]
            length_reward.append(get_length_reward(answer, budget))
        
        return length_reward
            
    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    set_seed(args.seed)
    setup(args)