#        x
#        |
#        v
#    +--------+
#    | Linear |  -->  F.linear(x, weight)
#    +--------+
#        |
#        +--(fine - tuning LoRA)--> (x @ Aᵀ) @ Bᵀ * scaling
#        |
#        y
#      Output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class LoRALayer():
    def __init__(
        self,
        rank: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRAAdapter(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 0,
        lora_alpha: int = 1, # Scaling factor for LoRA
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, rank=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if rank > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.scaling = self.lora_alpha / self.rank
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A with a uniform distribution ~U = [-sqrt(6/#input) * sqrt(5), sqrt(6/#input) * sqrt(5)]
            # initialize B the same way as the default for nn.Linear and A to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode: # ====> Training mode
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                # If the weights are currently in a merged state
                # (i.e., LoRA has been added to the base weight),
                # we need to unmerge the LoRA component so that
                # the model learns only through the separate LoRA parameters
                if self.rank > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.rank > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.rank > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)