from .optimization import checkpointing, mixed_precision, use_scaler, gradient_clipping, gradient_accumulation, split_model
from .device_utils import detect_device

device_type, core_num = detect_device()

def train_parallel_model(model, dataloader, optimizer, loss_fn, device="cpu", num_epochs=10, config=None):

    """
    Trains a model with user-selected optimizations: gradient checkpointing, ZeRO, 
    mixed precision, gradient clipping, gradient accumulation, and automatic loss scaling.

    Parameters:
    - model: PyTorch model to be trained.
    - dataloader: DataLoader for training data.
    - optimizer: Optimizer for model parameters.
    - loss_fn: Loss function.
    - device (str): Device for training ("cpu" or "cuda").
    - num_epochs (int): Number of epochs to train.
    - config (dict): Configuration dictionary specifying which optimizations to apply.
      Possible keys:
        - 'gradient_accumulation': (int) Number of steps for gradient accumulation.
        - 'mixed_precision': (bool) Use mixed precision with automatic loss scaling.
        - 'gradient_clipping': (float) Max norm for gradient clipping.
        - 'checkpointing': (bool) Use gradient checkpointing for memory efficiency.
    """

    accumulation_steps = config.get('gradient_accumulation', 1)
    use_mixed_precision = config.get('mixed_precision', False)
    clip_grad_norm = config.get('gradient_clipping', None)
    use_checkpointing = config.get('checkpointing', False)

    model, devices = split_model(model, device_type, core_num)
    scaler = use_scaler(use_mixed_precision)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(devices[0])
            targets = targets.to(devices[-1])

            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            if use_checkpointing:
                outputs = checkpointing(model, inputs)
            else:
                outputs = mixed_precision(model, inputs, use_mixed_precision, scaler)

            loss = loss_fn(outputs, targets) / accumulation_steps

            gradient_accumulation(
                loss, optimizer, scaler, i, accumulation_steps, use_mixed_precision
            )

            if clip_grad_norm is not None and (i + 1) % accumulation_steps == 0:
                if use_mixed_precision and scaler:
                    scaler.unscale_(optimizer)
                gradient_clipping(model, clip_grad_norm)

            running_loss += loss.item() * accumulation_steps

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    print("Training complete.")