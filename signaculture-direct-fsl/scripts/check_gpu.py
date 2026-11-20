import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
free,total=torch.cuda.mem_get_info()
print(f"{torch.cuda.get_device_name()} · {free/2**30:.2f}/{total/2**30:.2f} GiB free")
