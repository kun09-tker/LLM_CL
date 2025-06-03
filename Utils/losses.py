import torch

from Utils.layers import get_lora_params

def orthogonal_loss(A, B):
    return torch.norm(A.T @ B) ** 2

def orthogonal_llm_cl_loss(variant_adapter,
                           invariant_apdater,
                           domain_name="lm_head",
                           share_name="invariant"):

    variant_params = get_lora_params(variant_adapter, domain_name)
    invariant_params = get_lora_params(invariant_apdater, share_name)

    loss = 0
    for name in invariant_params:
        if "lora_A" in name:
            loss += orthogonal_loss(variant_params[name], invariant_params[name])
        elif "lora_B" in name:
            loss += orthogonal_loss(variant_params[name], invariant_params[name])
    return loss