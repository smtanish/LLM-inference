import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import matplotlib.pyplot as plt
# load model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()
#prepare prompt
prompt = "explain machine learning in one sentence."
messages = [
    {
        "role": "user",
        "content": prompt
    }
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)
input_ids = inputs["input_ids"]
max_new_tokens = 30
#uncached decoder
def benchmark_uncached():
    current_ids = input_ids.clone()
    step_times = []
    # Warmup
    for _ in range(3):
        with torch.inference_mode():
            model(
                input_ids=current_ids,
                use_cache=False,
                logits_to_keep=1
            )
    torch.cuda.synchronize()
    for step in range(max_new_tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=current_ids,
                use_cache=False,
                logits_to_keep=1
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )
        current_ids = torch.cat(
            [current_ids, next_token],
            dim=1
        )
    return step_times
#kv cache decoder
def benchmark_cached():
    # Prefill
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1
        )
    cache = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )
    step_times = []
    # Warmup cached decode
    warmup_cache = cache
    for _ in range(3):
        current_position = warmup_cache.get_seq_length()
        position_ids = torch.tensor(
            [[current_position]],
            device=model.device
        )
        attention_mask = torch.ones(
            (1, current_position + 1),
            dtype=torch.long,
            device=model.device
        )
        with torch.inference_mode():
            model(
                input_ids=next_token,
                past_key_values=warmup_cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1
            )
    # Re-run prefill so timing starts with a fresh cache
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1
        )
    cache = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True
    )
    torch.cuda.synchronize()
    for step in range(max_new_tokens - 1):
        current_position = cache.get_seq_length()
        position_ids = torch.tensor(
            [[current_position]],
            device=model.device
        )
        attention_mask = torch.ones(
            (1, current_position + 1),
            dtype=torch.long,
            device=model.device
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True
        )
    return step_times
#run benchmarks
uncached_times = benchmark_uncached()
cached_times = benchmark_cached()
#print results
print("\nUncached:")
print("Average:", sum(uncached_times) / len(uncached_times), "ms/token")
print("Total:", sum(uncached_times), "ms")
print("\nKV cached:")
print("Average:", sum(cached_times) / len(cached_times), "ms/token")
print("Total:", sum(cached_times), "ms")
speedup = (
    sum(uncached_times) / len(uncached_times)
) / (
    sum(cached_times) / len(cached_times)
)
print("\nAverage speedup:", speedup, "x")
#plot
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(uncached_times) + 1),
    uncached_times,
    label="Uncached"
)
plt.plot(
    range(1, len(cached_times) + 1),
    cached_times,
    label="KV Cache"
)
plt.xlabel("Generation step")
plt.ylabel("Forward-pass latency (ms)")
plt.title("Uncached vs KV-Cache Decoding")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()