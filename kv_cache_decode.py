import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
model_name="Qwen/Qwen2.5-0.5B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()
def kv_prefill(prompt):
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
    ).to(model.device)
    with torch.inference_mode():
        outputs=model(
            input_ids=inputs["input_ids"],
            use_cache=True,
            logits_to_keep=1
        )
    cache=outputs.past_key_values
    logits=outputs.logits[:,-1,:]

    next_token=torch.argmax(
        logits,
        dim=-1,
        keepdim=True
    )
    print("prompt length:",inputs["input_ids"].shape[1])
    print("cache length:",cache.get_seq_length())
    print("first token:",next_token.item())
    print("cache check:",cache.get_seq_length()==inputs["input_ids"].shape[1])
    return inputs["input_ids"],next_token,cache

prompt="explain machine learning in one sentence."
prefill_input_ids,next_token,cache=kv_prefill(prompt)
generated_tokens=[next_token.item()]
eos_token_ids=model.generation_config.eos_token_id
if isinstance(eos_token_ids,int):
    eos_token_ids=[eos_token_ids]
for step in range(1,50):
    current_position=cache.get_seq_length()
    position_ids=torch.tensor(
        [[current_position]],
        device=model.device
    )
    attention_mask=torch.ones(
        (1,current_position+1),
        dtype=torch.long,
        device=model.device
    )
    with torch.inference_mode():
        outputs=model(
            input_ids=next_token,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1
        )
    next_token=torch.argmax(
        outputs.logits[:,-1,:],
        dim=-1,
        keepdim=True
    )
    generated_tokens.append(next_token.item())
    if next_token.item() in eos_token_ids:
        break
print("generated tokens:",generated_tokens)
print("number of generated tokens:",len(generated_tokens))
print("final cache length:",cache.get_seq_length())
response=tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)
print("response:",response)