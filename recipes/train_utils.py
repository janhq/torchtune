# This code is adapted from https://github.com/pytorch/torchtitan/tree/main.

import gc
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d

def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (
        model_config.get('num_hidden_layers'),
        model_config.get('num_attention_heads'),
        model_config.get('hidden_size') // model_config.get('num_attention_heads'),
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    flop_per_token = 6 * num_params + 12 * l * h * q * t

    return flop_per_token

# hardcoded BF16 type peak flops for NVIDIA A100 and H100 GPU
def get_peak_flops(device_name: str) -> int:
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 1979e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for SXM and other variants
            return 989e12
    else:  # for other GPU types, assume A100
        return 312e12
    
def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    num_params = sum(p.numel() for p in model.parameters())
    if exclude_embedding:
        num_params -= model.tok_embeddings.weight.numel()
    return num_params