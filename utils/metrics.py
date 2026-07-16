import os
import time
import psutil
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Tentar importar pynvml para métricas de GPU
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def get_gpu_utilization() -> Optional[Dict[str, float]]:
    """Retorna utilização da GPU usando pynvml (nvidia-ml-py)."""
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return None

    try:
        pynvml.nvmlInit()
        device_index = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        result = {
            "gpu_utilization_percent": utilization.gpu,
            "memory_utilization_percent": utilization.memory,
            "gpu_memory_total_mb": memory_info.total / 1024 / 1024,
            "gpu_memory_used_mb": memory_info.used / 1024 / 1024,
            "gpu_memory_free_mb": memory_info.free / 1024 / 1024,
        }

        pynvml.nvmlShutdown()
        return result
    except Exception:
        return None


def get_cpu_utilization() -> Dict[str, float]:
    """Retorna utilização da CPU."""
    cpu_freq = psutil.cpu_freq()

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_freq_current_mhz": cpu_freq.current if cpu_freq else None,
        "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else None,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }


def get_memory_usage() -> Dict[str, float]:
    """Returns CPU and GPU memory usage in MB."""
    metrics = {}

    # CPU Memory
    process = psutil.Process(os.getpid())
    metrics["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

    # CPU Utilization
    cpu_util = get_cpu_utilization()
    metrics.update(cpu_util)

    # GPU (PyTorch metrics)
    if torch.cuda.is_available():
        metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        metrics["gpu_memory_max_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / 1024 / 1024
        )

        # GPU Utilization via pynvml
        gpu_util = get_gpu_utilization()
        if gpu_util:
            metrics["gpu_utilization_percent"] = gpu_util["gpu_utilization_percent"]
            metrics["memory_utilization_percent"] = gpu_util["memory_utilization_percent"]
        else:
            metrics["gpu_utilization_percent"] = None
            metrics["memory_utilization_percent"] = None
    else:
        metrics["gpu_memory_allocated_mb"] = None
        metrics["gpu_memory_reserved_mb"] = None
        metrics["gpu_memory_max_allocated_mb"] = None
        metrics["gpu_utilization_percent"] = None
        metrics["memory_utilization_percent"] = None

    return metrics

def get_model_size(model: nn.Module) -> float:
    """Returns model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb

def get_parameters_count(model: nn.Module) -> Dict[str, int]:
    """Returns total and trainable parameters count."""
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

def calculate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Calcula FLOPs do modelo considerando todas as camadas relevantes.
    Retorna dicionário com FLOPs totais e breakdown por tipo de camada.
    """
    flops_by_layer = {
        "conv": 0,
        "linear": 0,
        "norm": 0,
        "activation": 0,
        "pool": 0,
        "attention": 0,
        "other": 0,
    }

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__

        # Convolução 2D (considera grupos para depthwise separable)
        if class_name == 'Conv2d':
            out_h, out_w = output.shape[2:]
            # FLOPs = 2 * K_h * K_w * (C_in / groups) * C_out * H_out * W_out
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * (module.in_channels // module.groups)
            )
            flops = 2 * kernel_ops * module.out_channels * out_h * out_w
            # Adicionar bias
            if module.bias is not None:
                flops += module.out_channels * out_h * out_w
            flops_by_layer["conv"] += flops

        # ConvTranspose2d
        elif class_name == 'ConvTranspose2d':
            out_h, out_w = output.shape[2:]
            kernel_ops = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * (module.in_channels // module.groups)
            )
            flops = 2 * kernel_ops * module.out_channels * out_h * out_w
            if module.bias is not None:
                flops += module.out_channels * out_h * out_w
            flops_by_layer["conv"] += flops

        # Linear
        elif class_name == 'Linear':
            # FLOPs = 2 * in_features * out_features (multiply-add)
            flops = 2 * module.in_features * module.out_features
            if module.bias is not None:
                flops += module.out_features
            flops_by_layer["linear"] += flops

        # Normalization layers
        elif class_name in ['BatchNorm2d', 'BatchNorm1d']:
            # mean, var, normalize, scale, shift = ~5 ops per element
            flops_by_layer["norm"] += 5 * output.numel()

        elif class_name == 'LayerNorm':
            # Similar ao BatchNorm
            flops_by_layer["norm"] += 5 * output.numel()

        elif class_name in ['InstanceNorm2d', 'InstanceNorm1d']:
            flops_by_layer["norm"] += 5 * output.numel()

        elif class_name == 'GroupNorm':
            flops_by_layer["norm"] += 5 * output.numel()

        # Activation functions
        elif class_name in ['ReLU', 'ReLU6', 'LeakyReLU', 'PReLU']:
            # 1 comparação por elemento
            flops_by_layer["activation"] += output.numel()

        elif class_name in ['GELU', 'SiLU', 'Swish', 'Mish']:
            # Aproximadamente 8 ops por elemento (sigmoid, multiply, etc)
            flops_by_layer["activation"] += 8 * output.numel()

        elif class_name == 'Sigmoid':
            # exp, add, div = ~3 ops
            flops_by_layer["activation"] += 3 * output.numel()

        elif class_name == 'Tanh':
            flops_by_layer["activation"] += 3 * output.numel()

        elif class_name == 'Softmax':
            # exp para cada elemento + soma + divisão
            flops_by_layer["activation"] += 3 * output.numel()

        # Pooling layers
        elif class_name in ['MaxPool2d', 'MaxPool1d']:
            kernel_size = module.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            # Comparações dentro do kernel
            flops_by_layer["pool"] += output.numel() * (kernel_size[0] * kernel_size[1] - 1)

        elif class_name in ['AvgPool2d', 'AvgPool1d']:
            kernel_size = module.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            # Soma + divisão
            flops_by_layer["pool"] += output.numel() * (kernel_size[0] * kernel_size[1])

        elif class_name in ['AdaptiveAvgPool2d', 'AdaptiveAvgPool1d']:
            # Estimar baseado no input size
            if len(input) > 0 and input[0] is not None:
                input_elements = input[0].numel()
                output_elements = output.numel()
                if output_elements > 0:
                    ratio = input_elements / output_elements
                    flops_by_layer["pool"] += int(output_elements * ratio)

        elif class_name in ['AdaptiveMaxPool2d', 'AdaptiveMaxPool1d']:
            if len(input) > 0 and input[0] is not None:
                input_elements = input[0].numel()
                output_elements = output.numel()
                if output_elements > 0:
                    ratio = input_elements / output_elements
                    flops_by_layer["pool"] += int(output_elements * (ratio - 1))

        # Multi-head attention
        elif class_name == 'MultiheadAttention':
            # Q, K, V projections + attention scores + output projection
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            seq_len = input[0].shape[0] if len(input) > 0 else 1

            # QKV projections: 3 * 2 * seq_len * embed_dim * embed_dim
            qkv_flops = 3 * 2 * seq_len * embed_dim * embed_dim
            # Attention: seq_len * seq_len * embed_dim (para cada head)
            attn_flops = 2 * num_heads * seq_len * seq_len * (embed_dim // num_heads)
            # Output projection
            out_flops = 2 * seq_len * embed_dim * embed_dim

            flops_by_layer["attention"] += qkv_flops + attn_flops + out_flops

    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:  # Leaf module
            hooks.append(module.register_forward_hook(hook_fn))

    # Move model to same device as input will be
    original_device = next(model.parameters()).device
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass

    for hook in hooks:
        hook.remove()

    model.to(original_device)

    total_flops = sum(flops_by_layer.values())

    return {
        "total_flops": total_flops,
        "flops_by_layer_type": flops_by_layer,
        "gflops": total_flops / 1e9,
        "mflops": total_flops / 1e6,
    }

def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, ...],
    iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Mede tempo de inferência usando CUDA Events para GPU (mais preciso).
    Retorna tempos médios, throughput e uso de memória.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    batch_size = input_size[0]

    cuda_available = torch.cuda.is_available()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(dummy_input)

    if cuda_available:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Medir memória antes
    mem_before = get_memory_usage()

    # Preparar timing
    cpu_times = []
    gpu_times = []

    if cuda_available:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(iterations):
            # CPU timing
            cpu_start = time.perf_counter()

            # GPU timing
            if cuda_available:
                start_event.record()

            model(dummy_input)

            if cuda_available:
                end_event.record()
                torch.cuda.synchronize()
                gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)

            cpu_end = time.perf_counter()
            cpu_times.append(cpu_end - cpu_start)

    mem_after = get_memory_usage()

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    std_cpu_time = np.std(cpu_times)

    result = {
        "avg_inference_time_seconds": avg_cpu_time,
        "std_inference_time_seconds": std_cpu_time,
        "throughput_samples_per_sec": batch_size / avg_cpu_time,
        "cpu_memory_mb": mem_after.get("cpu_memory_mb", 0),
        "iterations": iterations,
    }

    if cuda_available and gpu_times:
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        std_gpu_time = np.std(gpu_times)
        result["avg_gpu_time_seconds"] = avg_gpu_time
        result["std_gpu_time_seconds"] = std_gpu_time
        result["gpu_throughput_samples_per_sec"] = batch_size / avg_gpu_time
        result["gpu_memory_allocated_mb"] = mem_after.get("gpu_memory_allocated_mb", 0)
        result["gpu_memory_peak_mb"] = mem_after.get("gpu_memory_max_allocated_mb", 0)

    return result
