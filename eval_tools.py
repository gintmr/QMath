import os
def apply_RL_prompt(chunk, args, budget):
    if args.prompt_type == "deepseek3" and os.environ['tip'] == "Ahead":
        #### Ahead方法为：在问题的末尾加上remaining。
        return RL_deepseek3_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "remaining":
        return RL_deepseek3_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt":
        return prompt_format(chunk, budget)
    elif args.prompt_type == "qwen25-math-cot" and os.environ['tip'] == "prompt":
        return prompt_format_qw(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt-based":
        return deepseek3_prompt_based(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt-based1":
        return deepseek3_prompt_based1(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt-based2":
        return deepseek3_prompt_based2(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "prompt-based3":
        return deepseek3_prompt_based3(chunk, budget)
    elif os.environ['tip'] == "default":
        return chunk
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "ATD_A" or os.environ['tip'] == "ATD_R":
        return ATD_A_deepseek3_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "TCM": 
        return TCM_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "TCMv2":
        return TCMv2_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "withoutremaining":
        return withoutremaining_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "8ratio":
        return _8ratio_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "TCM+":
        return TCMv2_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "SST":
        return SST_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "Tokenskip":
        return Tokenskip_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "thinkprune":
        return thinkprune_prompt(chunk, budget)
    else:
        return chunk





def Tokenskip_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    alpha = os.environ['alpha']
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        add_prompt = f'<｜end▁of▁sentence｜>{alpha}<｜end▁of▁sentence｜>'
        chunk[i] = head + add_prompt + find_strings + tail
    return chunk
    

def SST_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    up_to_50 = budget // 50 + 1
    N = up_to_50
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        add_prompt = f"\n(Complete thinking within {N} <countdown> or fewer.)"
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'

        add_response = "\n<think>\n\n<countdown>\n"
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + tail + add_response
        # print(f"chunk[i] = {chunk[i]}")
    return chunk


def TCMv2_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        add_prompt = f"\n(Complete thinking within \n<remaining>{budget}</remaining>\n tokens or fewer.)"
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'

        add_response = f""
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
        # print(f"chunk[i] = {chunk[i]}")
    return chunk



def withoutremaining_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer.)"
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'

        add_response = f""
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
        # print(f"chunk[i] = {chunk[i]}")
    return chunk

def _8ratio_prompt(chunk, budget):
    os.environ['budget'] = str(budget)
    print(f"budget = {budget}")
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer, 7 special tokens ( \n<remaining>7/8</remaining>\n , \n<remaining>6/8</remaining>\n , \n<remaining>5/8</remaining>\n , \n<remaining>4/8</remaining>\n , \n<remaining>3/8</remaining>\n , \n<remaining>2/8</remaining>\n , \n<remaining>1/8</remaining>\n ) will split the thinking process into 8 parts.)"
        
        add_response = f""

        chunk[i] = head + add_prompt + find_strings + add_response + tail
        
    return chunk

def thinkprune_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        add_prompt = f"The output of the assistant should be within {budget} tokens"
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'

        add_response = f""
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
        # print(f"chunk[i] = {chunk[i]}")
    return chunk

def TCM_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        add_prompt = f"\n(Complete thinking within \n<remaining>{budget}</remaining>\n tokens or fewer.)"
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'
        add_response = f""
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
        # print(f"chunk[i] = {chunk[i]}")
    return chunk


def prompt_format(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"\n(Complete thinking within {budget} tokens or fewer.)\n"
        chunk[i] = head + find_strings + tail
    return chunk


def prompt_format_qw(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<|im_start|>assistant"

    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"\n(Complete thinking within {budget} tokens or fewer.)\n"
        chunk[i] = head + find_strings + tail
    return chunk





def ATD_A_deepseek3_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        add_prompt = f'\n(Respond in {budget} tokens or fewer. Complete the process between <think> and </think> within the token budget. Display the countdown exponentially as <remaining>xxx</remaining>, where xxx = 50 * 2^n, n >= 0. Think more concisely as countdown decreases.)\n'
        add_response = f"\n(I will complete the process within {budget} tokens and show the countdown as <remaining>xxx</remaining>, following the exponential rule.I will think more concisely as countdown decreases.)\n"
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
    return chunk




def RL_deepseek3_prompt(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"(Please respond in {budget} tokens or fewer)\n"
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + find_strings + tail
    return chunk
        
    
def deepseek3_prompt_based(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"You should finish thinking with in {budget} tokens.\n"
        chunk[i] = head + find_strings + tail
    return chunk
    

def deepseek3_prompt_based1(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"(Please respond in {budget} tokens or fewer)\n"
        chunk[i] = head + find_strings + tail
    return chunk



def deepseek3_prompt_based2(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"(HARD STOP at {budget} tokens)\n"
        chunk[i] = head + find_strings + tail
    return chunk


def deepseek3_prompt_based3(chunk, budget):
    '''
     <｜User｜>Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. 
     Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
     <｜Assistant｜>
    '''
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        head += f"(Response length limit: Strictly {budget} tokens max. Budget cannot be exceeded)\n"
        chunk[i] = head + find_strings + tail
    return chunk


    
    
    
# def solve_final_answer(chunk):
#     k = 0
#     for i in range(len(chunk)):
#         if "**Final Answer**\\boxed" in chunk[i][:-10] and "<｜end▁of▁sentence｜>" not in chunk[i]:
#             chunk[i] += "<｜end▁of▁sentence｜>"
#             k += 1
#     print(f"###added {k} final answer!")
#     return chunk

# import re

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

def solve_final_answer(chunk: list) -> list:
    
    """处理包含嵌套大括号的答案匹配"""
    
    end_chunk = []
    open_chunk = []
    
    k = 0
    pattern = "**Final Answer**\\boxed{"  # 初级匹配
    
    for i in range(len(chunk)):
        line = chunk[i]
        if not pattern in line:
            open_chunk.append(chunk[i])
            continue
        # 深度扫描完整结构
        start_idx = line.find('**Final Answer**\\boxed{')
        if start_idx == -1:
            open_chunk.append(chunk[i])
            continue
        # 手动解析嵌套结构
        stack = 1
        end_idx = start_idx + len('**Final Answer**\\boxed{')
        while end_idx < len(line) and stack > 0:
            if line[end_idx] == "{":
                stack += 1
            elif line[end_idx] == "}":
                stack -= 1
            end_idx += 1
        
        # 验证闭合状态
        if stack == 0 and is_balanced(line[start_idx:end_idx]):
            
            chunk[i] += "<｜end▁of▁sentence｜>"  # 保持原有操作
            k += 1
            end_chunk.append(chunk[i])
        else:
            open_chunk.append(chunk[i])

    print(f"### Find {k} anwsers have final answer!")
    return chunk, end_chunk, open_chunk