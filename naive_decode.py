import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
import time
import matplotlib.pyplot as plt 
model_name="Qwen/Qwen2.5-0.5B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()
print("model loaded")
def naive_decode(prompt,max_new_tokens):
    messages=[
        {
            "role":"user",
            "content":prompt,
        }
    ]
    text=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs=tokenizer(
        text,
        return_tensors="pt"
    )
    input_ids=inputs["input_ids"].to(model.device)
    for _ in range(3):
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=1
            )
    generated_tokens=[]
    eos_token_ids=model.generation_config.eos_token_id
    if isinstance(eos_token_ids,int):
        eos_token_ids=[eos_token_ids]
    latencies=[]
    sequence_lengths=[]
    for step in range(max_new_tokens):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.inference_mode():
            outputs=model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=1
            )
        torch.cuda.synchronize()
        t1=time.perf_counter()
        latencies.append((t1-t0)*1000)
        sequence_lengths.append(input_ids.shape[1])
        logits=outputs.logits[:,-1,:]
        next_token=torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )
        generated_tokens.append(next_token.item())
        if next_token.item() in eos_token_ids:
            break
        input_ids=torch.cat(
            [input_ids,next_token],
            dim=1
        )
    response=tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    print(response)
    return generated_tokens,latencies,sequence_lengths
prompt="explain machine learning in one sentence."
manual_tokens,manual_latencies,manual_sequence_lengths=naive_decode(prompt,50)
print("Latency measurements:",len(manual_latencies))
print("Sequence lengths:",len(manual_sequence_lengths))
plt.plot(manual_latencies)
plt.xlabel("decoding step")
plt.ylabel("latency(ms)")
plt.title("naive decoding: latency vs step")
plt.show()
print("step 0 latency: ",manual_latencies[0],"ms")
print("step 29 latency: ",manual_latencies[29],"ms")
print("step 29/step 0 ratio:",manual_latencies[29]/manual_latencies[0])
total_time=sum(manual_latencies)/1000
tokens_per_sec=len(manual_latencies)/total_time
print("total uncached time: ",total_time,"sec")
print("total uncached tokens/second: ",tokens_per_sec)
messages=[
    {
        "role":"user",
        "content":prompt,
    }
]
text=tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs=tokenizer(
    text,
    return_tensors="pt"
)
inputs=inputs.to(model.device)
with torch.inference_mode():
    hf_output=model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=False,
        repetition_penalty=1.0
    )
input_length=inputs["input_ids"].shape[1]
hf_tokens=hf_output[0][input_length:].tolist()
print("Manual tokens:",manual_tokens)
print("HF tokens:    ",hf_tokens)
print("Tokens match:",manual_tokens==hf_tokens)
torch.cuda.synchronize()
t0=time.perf_counter()
with torch.inference_mode():
    cached_output=model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=True,
        repetition_penalty=1.0
    )
torch.cuda.synchronize()
t1=time.perf_counter()
cached_time=(t1-t0)
cached_tokens=cached_output.shape[1]-input_length
cached_tokens_per_sec=cached_tokens/cached_time
print("total cached time(sec)",cached_time)
print("cached tokens/second",cached_tokens_per_sec)
base_inputs=tokenizer(
    "explain machine learning in one sentence.",
    return_tensors="pt"
).to(model.device)

batch_sizes=[1,2,4,8,16,32,64,128,256]

for batch_size in batch_sizes:
    input_ids=base_inputs["input_ids"].repeat(batch_size,1)

    for _ in range(5):
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=1
            )

    torch.cuda.synchronize()
    t0=time.perf_counter()

    for _ in range(20):
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=1
            )

    torch.cuda.synchronize()
    t1=time.perf_counter()

    total=(t1-t0)/20
    print(
        "batch size:",batch_size,
        "total time:",total*1000,"ms",
        "time per sequence:",total/batch_size*1000,"ms"
    )