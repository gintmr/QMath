import subprocess
import time
import logging 


logger = logging.getLogger()
log_file_path = "gpu_hunter.log"
logger.addHandler(logging.FileHandler(log_file_path))
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(message)s')
print("start hunting")
def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits", "-i", "0,1,2,3"],
            universal_newlines=True
        )
        return output.strip().split('\n')
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def check_low_usage(threshold=10):
    gpu_data = get_gpu_memory_usage()
    if not gpu_data:
        return False

    for gpu in gpu_data:
        used, total = map(int, gpu.split(', '))
        usage_percent = (used / total) * 100
        if usage_percent >= threshold:
            return False
    return True

def main():
    check_interval = 60*3  # 检查间隔（秒）
    command_to_run = "bash /mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation/remaining_eval/TCMv2_RL_copy.sh"  # 替换为需要执行的命令

    while True:
        if check_low_usage(threshold=10):
            print("All GPUs have memory usage below 10%. Executing command...")
            
            subprocess.run('conda deactivate', shell=True)
            subprocess.run('conda activate QMath-wxr', shell=True)
            subprocess.run(command_to_run, shell=True)

            print("Command executed. Exiting GPU monitoring.")
            break  # 退出循环，停止监听
        else:
            print("GPUs are in use. Waiting...")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    main()