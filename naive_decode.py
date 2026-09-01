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
print(model.generation_config)
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
    return generated_tokens
prompt="explain machine learning in one sentence."
manual_tokens=naive_decode(prompt,50)
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

