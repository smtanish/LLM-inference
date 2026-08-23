import torch
is_avail = torch.cuda.is_available()
print(f"CUDA Available: {is_avail}")
if is_avail:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total Memory: {mem_gb:.2f} GB")
else:
    print(f"Version: {torch.version.cuda}")
    if torch.version.cuda is None:
        print("Not installed")
