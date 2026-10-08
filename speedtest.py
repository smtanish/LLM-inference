import torch
import time

x=torch.randn(512*1024*1024,device="cuda",dtype=torch.float16)

for _ in range(10):
    y=x.clone()

torch.cuda.synchronize()

t0=time.perf_counter()

for _ in range(20):
    y=x.clone()

torch.cuda.synchronize()

t1=time.perf_counter()

elapsed=t1-t0
time_per_iter=elapsed/20
bandwidth=(2*1024**3)/time_per_iter/1e9

print("total time:",elapsed,"sec")
print("time per clone:",time_per_iter*1000,"ms")
print("bandwidth:",bandwidth,"GB/s")