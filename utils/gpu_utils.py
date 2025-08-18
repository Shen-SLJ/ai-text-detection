import torch

def is_gpu_available() -> bool:
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available for model training & prediction."
        )
        return False

    gpu_capability = torch.cuda.get_device_capability()
    gpu_capability_str = f"sm_{gpu_capability[0]}{gpu_capability[1]}"
    is_gpu_capable = gpu_capability_str in torch.cuda.get_arch_list()

    if not is_gpu_capable:
        print(
            f"WARNING: CUDA available, but GPU capability ({gpu_capability_str}) unsupported."
        )
        return False

    print("INFO: CUDA available.")
    return True
