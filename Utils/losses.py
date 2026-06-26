import torch
import torch.nn as nn

# from Utils.layers import get_lora_params

def orthogonality_loss_lora(delta_shared, delta_domain):
    delta_shared = delta_shared[:, 0, :]
    delta_domain = delta_domain[:, 0, :]

    delta_shared = nn.functional.normalize(delta_shared, p=2, dim=1)
    delta_domain = nn.functional.normalize(delta_domain, p=2, dim=1)

    dot_product = torch.sum(delta_shared * delta_domain, dim=1)
    return torch.mean(dot_product ** 2)
