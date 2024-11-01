import psutil
import torch
from device_utils import detect_device, device_info
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed

device_type, core_num = detect_device()

def check_memory(data, batch_size, num_core=None):
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a PyTorch tensor.")
    
    if num_core is None:
        num_core = core_num

    data_size = data.element_size() * data.nelement()
    estimated_memory = data_size * batch_size / num_core

    if device_type == "gpu":
        available_memory = 0
        device = "GPU"

        for gpu_id in range(num_core):
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            reserved_memory = torch.cuda.memory_reserved(gpu_id)
            available_memory += (total_memory - reserved_memory)
    
    elif device_type == "tpu":
        available_memory = 16 * 1024 ** 3
        device = "TPU"
    
    else:
        available_memory = psutil.virtual_memory().available
        device = "CPU"

    if estimated_memory > available_memory:
        raise MemoryError(
            f"Estimated memory usage ({estimated_memory / 1024 ** 2:.2f} MB) "
            f"exceeds available {device} memory ({available_memory / 1024 ** 2:.2f} MB)."
        )
    else:
        print(f"Memory check passed: Sufficient {device} memory available for the batch.")
        return True


def parallel_dataload(dataset, batch_size, device_type="cpu", num_workers=1, rank=0, world_size=1):

    info = device_info()
    if device_type == "gpu" and not info["GPU"]["Available"]:
        raise RuntimeError("Requested device type 'gpu', but no GPU is available.")
    if device_type == "tpu" and not info["TPU"]["Available"]:
        raise RuntimeError("Requested device type 'tpu', but no TPU is available.")

    try:
        if device_type == "tpu":
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            dataloader = pl.ParallelLoader(train_loader, [xm.xla_device()]).per_device_loader(xm.xla_device())
            print("Loaded using TPU.")

        elif device_type == "gpu":
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
                torch.cuda.set_device(rank)
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
                print(f"Using Distributed Data Parallel with {num_gpus} GPUs.")
            else:
                torch.cuda.set_device(0)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                print("Loaded using single GPU.")

        elif device_type == "cpu":
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
            print("Loaded using CPU.")

        else:
            raise ValueError("Invalid device type. Choose from 'cpu', 'gpu', or 'tpu'.")

        return dataloader

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Out of memory error encountered.")
            raise MemoryError("Out of memory: Consider reducing batch size.") from e
        else:
            raise e