import torch
from torch.utils.checkpoint import checkpoint
from torch.amp import GradScaler, autocast

def checkpointing(model, inputs):
    return checkpoint(model, inputs)

def mixed_precision(model, inputs, use_mixed_precision, scaler=None):
    with autocast(enabled=use_mixed_precision):
        outputs = model(inputs)
    return outputs

def use_scaler(use_mixed_precision):
    return GradScaler() if use_mixed_precision else None

def gradient_clipping(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def gradient_accumulation(loss, optimizer, scaler, step, accumulation_steps, use_mixed_precision):
    if use_mixed_precision and scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        if use_mixed_precision and scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

def split_model(model, device_type, core_count):

    devices = []
    if device_type == "gpu":
        devices = [torch.device(f"cuda:{i}") for i in range(core_count)]
    elif device_type == "tpu":
        import torch_xla.core.xla_model as xm
        devices = [xm.xla_device() for i in range(core_count)]
    else:
        devices = [torch.device("cpu")] * core_count

    layers = list(model.children())
    num_layers = len(layers)
    layers_per_device = num_layers // core_count

    for i, device in enumerate(devices):
        start = i * layers_per_device
        end = (i + 1) * layers_per_device if i < core_count - 1 else num_layers
        for layer in layers[start:end]:
            layer.to(device)
    
    return model, devices