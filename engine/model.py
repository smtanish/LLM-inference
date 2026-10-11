import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
def load_model(model_name=MODEL_NAME):
    """Load the model and tokenizer once and return both."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer
