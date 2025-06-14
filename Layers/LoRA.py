import torch.nn as nn
from peft import LoraConfig, get_peft_model

class LoRALinear(nn.Module):
    def __init__(self, name, in_features=768, out_features=768):
        super(LoRALinear, self).__init__()
        linear = nn.Linear(in_features, out_features)
        self.name = name
        self.layers = nn.ModuleDict({name: linear})

    def forward(self, x):
        # Xử lý đầu vào qua mô-đun LoRA
        return self.layers[self.name](x)

class LoRAApdater(nn.Module):
    def __init__(self, name, in_features=768, out_features=768, rank=8, alpha=16, dropout=0.1):
        super(LoRAApdater, self).__init__()

        lora_config = LoraConfig(
            target_modules=[name],
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout
        )

        self.adapter = LoRALinear(name, in_features, out_features)
        self.adapter = get_peft_model(self.adapter, lora_config)
        self.adapter_sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.adapter(x)
        x = self.adapter_sigmod(x)
        return x