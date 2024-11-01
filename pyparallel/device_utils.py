import torch
import psutil

def detect_device():
    try:
        import torch_xla.core.xla_model as xm
        tpu_cores = xm.xrt_world_size()
        if tpu_cores > 0:
            return "tpu", tpu_cores
    except ImportError:
        pass

    if torch.cuda.is_available():
        gpu_core = torch.cuda.device_count()
        return "gpu", gpu_core

    print("TPU or GPU not available. Using CPU.")
    cpu_core = psutil.cpu_count(logical=True)
    return "cpu", cpu_core

def device_info():
    device_info = {}

    device_type, core_count = detect_device()

    if device_type == "gpu":
        num_gpus = core_count
        gpu_devices = []
        for i in range(num_gpus):
            device_name = torch.cuda.get_device_name(i)
            vram_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            vram_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            gpu_devices.append({
                "VRAM Allocated (GB)": round(vram_allocated, 2),
                "VRAM Reserved (GB)": round(vram_reserved, 2)
            })
        
        device_info["GPU"] = {
            "Available": True,
            "Core Count": num_gpus,
            "Devices": gpu_devices
        }
    else:
        device_info["GPU"] = {
            "Available": False,
            "Core Count": 0,
            "Devices": "No GPUs available"
        }

    if device_type == "tpu":
        tpu_devices = [f"TPU Core {i}" for i in range(core_count)]
        device_info["TPU"] = {
            "Available": True,
            "Core Count": core_count,
            "Devices": tpu_devices,
            "VRAM Info": "VRAM usage not directly accessible for TPUs"
        }
    else:
        device_info["TPU"] = {
            "Available": False,
            "Core Count": 0,
            "Devices": "No TPUs available"
        }

    if device_type == "cpu":
        cpu_devices = {
            "Total Memory (GB)": round(psutil.virtual_memory().total / 1024 ** 3, 2),
            "Available Memory (GB)": round(psutil.virtual_memory().available / 1024 ** 3, 2)
        }
        device_info["CPU"] = {
            "Available": True,
            "Core Count": core_count,
            "Devices": cpu_devices
        }
    else:
        device_info["CPU"] = {
            "Available": False,
            "Core Count": 0,
            "Devices": "No CPU information needed as GPU or TPU is available"
        }

    for device_type, details in device_info.items():
        print(f"{device_type} Details:")
        for key, value in details.items():
            if key == "Devices" and isinstance(value, list):
                for idx, device in enumerate(value):
                    print(f"  Device {idx + 1}:")
                    for k, v in device.items():
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print()

    return device_info