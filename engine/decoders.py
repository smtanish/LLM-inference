import time
import torch
def _synchronize_if_cuda(tensor):
    """Wait for queued GPU work when timing a forward call."""
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
def _timed_forward(model, reference_tensor, **kwargs):
    """Measure one model forward call in milliseconds."""
    _synchronize_if_cuda(reference_tensor)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**kwargs)
    _synchronize_if_cuda(reference_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return outputs, elapsed_ms
def _get_eos_ids(model):
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        return []
    if isinstance(eos_ids, int):
        return [eos_ids]
    return list(eos_ids)
def _get_pad_id(model, eos_ids):
    pad_id = model.generation_config.pad_token_id
    if pad_id is not None:
        return pad_id
    # Needed only if finished sequences must be padded during batching.
    return eos_ids[0] if eos_ids else 0
def _eos_mask(next_token, eos_ids):
    """Return a Boolean mask indicating which batch rows predicted EOS."""
    is_eos = torch.zeros(
        next_token.shape[0],
        dtype=torch.bool,
        device=next_token.device,
    )
    for eos_id in eos_ids:
        is_eos |= next_token[:, 0].eq(eos_id)
    return is_eos
def naive_decode(model, input_ids, max_new_tokens, ignore_eos=False):
    """
    Greedy decoding without a KV cache.

    Returns:
        tokens: [batch_size, generated_length]
        timings: {"forward_ms": [...]}
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, input_length].")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    batch_size = input_ids.shape[0]
    if max_new_tokens == 0:
        return input_ids.new_empty((batch_size, 0)), {"forward_ms": []}
    current_ids = input_ids
    generated_steps = []
    forward_ms = []
    eos_ids = _get_eos_ids(model)
    pad_id = _get_pad_id(model, eos_ids)
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )
    for _ in range(max_new_tokens):
        outputs, elapsed_ms = _timed_forward(
            model,
            current_ids,
            input_ids=current_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        forward_ms.append(elapsed_ms)
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        if not ignore_eos and eos_ids:
            is_eos = _eos_mask(next_token, eos_ids)
            # Rows that already finished receive padding tokens.
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_id),
                next_token,
            )
            generated_steps.append(next_token)
            finished = finished | is_eos

            if bool(finished.all().item()):
                break
        else:
            generated_steps.append(next_token)
        current_ids = torch.cat(
            (current_ids, next_token),
            dim=1,
        )
    tokens = torch.cat(generated_steps, dim=1)
    return tokens, {"forward_ms": forward_ms}
def kv_decode(model, input_ids, max_new_tokens, ignore_eos=False):
    """
    Greedy decoding with a reusable KV cache.

    Returns:
        tokens: [batch_size, generated_length]
        timings: {"forward_ms": [...]}

    The first timing measures prefill, which produces the first token.
    Subsequent timings measure the forward calls that produce later tokens.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, input_length].")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    batch_size = input_ids.shape[0]
    if max_new_tokens == 0:
        return input_ids.new_empty((batch_size, 0)), {"forward_ms": []}
    # Prefill: process the complete prompt and initialize the KV cache.
    outputs, elapsed_ms = _timed_forward(
        model,
        input_ids,
        input_ids=input_ids,
        use_cache=True,
        logits_to_keep=1,
    )
    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("The model did not return a KV cache.")
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )
    forward_ms = [elapsed_ms]
    generated_steps = []
    eos_ids = _get_eos_ids(model)
    pad_id = _get_pad_id(model, eos_ids)
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=input_ids.device,
    )
    for step in range(max_new_tokens):
        if not ignore_eos and eos_ids:
            is_eos = _eos_mask(next_token, eos_ids)
            token_to_append = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_id),
                next_token,
            )
            generated_steps.append(token_to_append)
            finished = finished | is_eos
            if bool(finished.all().item()):
                break
        else:
            token_to_append = next_token
            generated_steps.append(token_to_append)
        # No extra forward pass is needed after the final output token.
        if step + 1 == max_new_tokens:
            break
        current_position = cache.get_seq_length()
        position_ids = torch.full(
            (batch_size, 1),
            current_position,
            dtype=torch.long,
            device=input_ids.device,
        )
        attention_mask = torch.ones(
            (batch_size, current_position + 1),
            dtype=torch.long,
            device=input_ids.device,
        )
        # Feed only the latest token; reuse the existing KV cache.
        outputs, elapsed_ms = _timed_forward(
            model,
            token_to_append,
            input_ids=token_to_append,
            past_key_values=cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        forward_ms.append(elapsed_ms)
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
    tokens = torch.cat(generated_steps, dim=1)
    return tokens, {"forward_ms": forward_ms}
