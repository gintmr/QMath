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
    sequence = [334, 19357, 21806, 334, 59, 79075]

    Thought_END = 0
    START = 0
    END = 0
    if not find_string in anwser:
        return Thought_END, START, END
    
    else:
        start_idx = anwser.find('**Final Answer**\\boxed{')
        stack = 1
        end_idx = start_idx + len('**Final Answer**\\boxed{')
        while end_idx < len(anwser) and stack > 0:
            if anwser[end_idx] == "{":
                stack += 1
            elif anwser[end_idx] == "}":
                stack -= 1
            end_idx += 1
        if stack == 0 and is_balanced(anwser[start_idx:end_idx]):
            Thought_END = 1
            # tokenized_answer = tokenizer(anwser)['input_ids']
            # #G 找出结束时的token位置
            # start_positions = find_sequence(tokenized_answer, sequence)

        else:
            Thought_END = 0
            
        return Thought_END, START, END
           
def calculate_ratios(thinking_ended, answer_correct):
    # 初始化计数器
    TP_count = 0
    total_count = len(thinking_ended)
    thinking_ended_count = 0
    answer_correct_count = 0

    # 遍历列表
    for i in range(total_count):
        if thinking_ended[i] == 1 and answer_correct[i] == 1:
            ET_count += 1
        if thinking_ended[i] == 1 and answer_correct[i] == 0:
            EF_count += 1
        if answer_correct[i] == 0 and answer_correct[i] == 1:
            OT_count += 1

    # 计算比值
    ET_ratio = ET_count / total_count if total_count > 0 else 0
    EF_ratio = TP_count / total_count if total_count > 0 else 0
    OT_ratio = OT_count / total_count if total_count > 0 else 0

    acc = sum(answer_correct) / total_count if total_count > 0 else 0
    
    return ET_ratio, EF_ratio, OT_ratio, acc