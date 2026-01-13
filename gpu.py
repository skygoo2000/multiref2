import subprocess
from multiprocessing import Process
import torch
import time


def get_gpu_count():
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    COMMAND = "nvidia-smi --query-gpu=name --format=csv"
    gpu_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    return len(gpu_info)


def get_gpu_memory(gpu_index):
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    COMMAND = f"nvidia-smi --query-gpu=memory.used --format=csv --id={gpu_index}"
    memory_used_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    return memory_used_values[0]


@torch.no_grad()
def guard_gpu(gpu_index, tensor1, tensor2, threshold=20000):
    a = tensor1.to(f"cuda:{gpu_index}")
    b = tensor2.to(f"cuda:{gpu_index}")
    while True:
        try:
            used_memory = get_gpu_memory(gpu_index)
            print("GPU {}: Used memory: {}MB".format(gpu_index, used_memory))
            if used_memory < threshold:  # less than 1GB
                print("bmm...")
                for _ in range(8192):
                    # for _ in range(128):
                    torch.bmm(
                       a,b
                    )
            time.sleep(0.1)  # to prevent instant spike and drop in usage
        except Exception as e:
            print(e)


if __name__ == "__main__":
    l = 128
    # Set the size of the tensors
    size = (l, l, l)
    # Randomly initialize two tensors
    tensor1 = torch.randn(size)
    tensor2 = torch.randn(size)
    gpu_number = get_gpu_count()
    print(f"==> Number of GPUs: {gpu_number}")
    processes = []
    for i in range(gpu_number):
        p = Process(target=guard_gpu, args=(i, tensor1, tensor2))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()