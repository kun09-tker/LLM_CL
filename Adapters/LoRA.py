#       [x]
#        |
#        v
#    +--------+
#    | Linear |  -->  F.linear(x, weight) [freeze]
#    +--------+
#        |
#        +--(fine-tuning LoRA)--> (x @ Aᵀ) @ Bᵀ * scaling
#        |
#       [y]
#        |
#        v
#    +--------+
#    | Linear |  -->  self.output_linear(y)
#    +--------+
#        |
#        v
#      Output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class LoRAAdapter(nn.Linear):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__(in_features, out_features)
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Lớp LoRA
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Đóng băng trọng số gốc
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Lớp Linear để đạt out_features=3
        self.output_linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.output_linear.weight)


    def forward(self, x):
        # Lớp LoRA
        lora_output = self.lora_dropout(x)
        lora_output = torch.matmul(lora_output, self.lora_A)
        lora_output = torch.matmul(lora_output, self.lora_B)
        lora_output = lora_output * self.scaling
        x_lora = x + lora_output

        # Sigmoid activation
        x_lora = self.lora_dropout(x_lora)

        # Kết hợp với linear output
        output = self.output_linear(x_lora)
        return output