import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name="Qwen/Qwen2.5-0.5B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()
print("model loaded")
memory_baseline=torch.cuda.memory_allocated() / 1e9
print(f"Memory allocated:{memory_baseline:.3f} gb")
messages=[
    {
        "role":"user",
        "content":"explain machine learning in one sentence."
    }
]
text=tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
inputs=tokenizer(
    text,
    return_tensors="pt"
)
inputs=inputs.to(model.device)
print("input IDs shape:",inputs["input_ids"].shape)
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    outputs=model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False
    )
input_length=inputs["input_ids"].shape[1]
generated_tokens=outputs[0][input_length:]
response=tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)
print(response)
peak_memory=torch.cuda.max_memory_allocated() /1e9
extra_memory=peak_memory-memory_baseline
print(f"peak gpu memory:{peak_memory:.3f}gb")
print(f"extra memory :{extra_memory:.3f}gb")
print(model.config)
L=model.config.num_hidden_layers
H=model.config.num_attention_heads
hidden_size=model.config.hidden_size
H_kv=model.config.num_key_value_heads
vocab_size=model.config.vocab_size
head_dim=hidden_size//H
print(f"num_hidden_layers: {L}")
print(f"num_attention_heads: {H}")
print(f"num_key_value_heads: {H_kv}")
print(f"hidden_size: {hidden_size}")
print(f"vocab_size: {vocab_size}")
print(f"derived head_dim: {head_dim}")
if hasattr(model.config, "head_dim"):
    print(f"config head_dim:{model.config.head_dim}")
    print(f"head_dim agrees:{head_dim == model.config.head_dim}")
else:
    print("Config does not expose head_dim; derived value is:",head_dim)
bps=2*L*H_kv*head_dim*2
print(f"KV cache: {bps} bytes/token")
print(f"KV cache: {bps/1024:.2f} KiB/token")
with torch.inference_mode():
    out=model(
        **inputs,
        use_cache=True
    )
print("type of past_key_values:",type(out.past_key_values))
print("DynamicCache attributes:")
print(dir(out.past_key_values))
print("number of cache layers:",len(out.past_key_values.layers))
layer0=out.past_key_values.layers[0]
print("layer 0 type",type(layer0))
print("layer 0 attributes:")
print(dir(layer0))
print("cache sequence length:",out.past_key_values.get_seq_length())
key0=layer0.keys 
value0=layer0.values 
print("layer 0 key shape",key0.shape)
print("layer 0 value shape",value0.shape)
print(
    "kv heads match config check:",
    key0.shape[1]==model.config.num_key_value_heads
)
total_kv_bytes=0
for layer in out.past_key_values.layers:
    total_kv_bytes+=layer.keys.numel()*layer.keys.element_size()
    total_kv_bytes+=layer.values.numel()*layer.values.element_size()
seq_len=out.past_key_values.get_seq_length()
measured_bpt=total_kv_bytes/seq_len
print("total kv bytes:",total_kv_bytes)
print("measured bpt:",measured_bpt)
print(f"measured KV KiB/token: {measured_bpt / 1024:.2f}")
print("logits shape:",out.logits.shape)
logits_bytes=out.logits.numel()*out.logits.element_size()
print("logits bytes:",logits_bytes)
print(f"logits KiB: {logits_bytes/1e6:.3f} MB")
gqa=H/H_kv
print("GQA ratio:",gqa,"x")
mha_bps=2*L*H*head_dim*2
print(f"kv cache without gqa:{mha_bps}bytes/token")
print(f"kv cache without gqa:{mha_bps/1024:.2f} KiB/token")
gpu_memory=8e9
gqa_tokens=gpu_memory/bps
mha_tokens=gpu_memory/mha_bps
print(f"GQA capacity:{gqa_tokens:.0f} tokens")
print(f"MHA capacity:{mha_tokens:.0f} tokens")
print(f"capacity multiplier {gqa_tokens/mha_tokens:.1f}x")