import torch
import subprocess


def print_gpu_memory():
    if torch.cuda.is_available():
        # NVIDIA GPU (CUDA)
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            memory_used = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: Used {memory_used:.2f}GB | Reserved {memory_reserved:.2f}GB | Total {total_memory:.2f}GB")

    elif torch.backends.mps.is_available():
        # Apple Silicon GPU (MPS)
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        total_memory = int(result.stdout.split(':')[1].strip()) / 1e9
        memory_used = torch.mps.current_allocated_memory() / 1e9
        print(f"GPU 0: Used {memory_used:.2f}GB | Total {total_memory:.2f}GB")
    else:
        print("No GPU available - running on CPU")