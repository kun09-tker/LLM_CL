import torch

def get_lora_params(model, module_name="lm_head"):
    lora_params = {}
    for name, param in model.named_parameters():
        if module_name in name and "lora_" in name:
            lora_params[name] = param
    return lora_params

def orthogonal_loss(invariant_params, variant_params):
    loss = 0
    for name in invariant_params:
        if "lora_A" in name:
            loss += torch.norm(variant_params[name].T @ invariant_params[name]) ** 2
        elif "lora_B" in name:
            loss += torch.norm(variant_params[name].T @ invariant_params[name]) ** 2
    return loss