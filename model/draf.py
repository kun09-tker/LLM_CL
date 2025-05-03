import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_size, rank=8):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, x):
        return self.up(self.down(x))


class LLM_CLModel(nn.Module):
    def __init__(self, base_model_name, hidden_size, rank=8):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.hidden_size = hidden_size
        self.rank = rank

        self.shared_adapter = LoRAAdapter(hidden_size, rank)
        self.domain_adapters = nn.ModuleDict()

    def add_domain_adapter(self, domain_name):
        self.domain_adapters[domain_name] = LoRAAdapter(self.hidden_size, self.rank)

    def forward(self, inputs_embeds, domain_name=None):
        hidden = inputs_embeds
        hidden = hidden + self.shared_adapter(hidden)
        if domain_name is not None and domain_name in self.domain_adapters:
            hidden = hidden + self.domain_adapters[domain_name](hidden)
        outputs = self.llm(inputs_embeds=hidden)
        return outputs

    def orthogonal_loss(self, domain_name):
        shared_down = self.shared_adapter.down.weight
        shared_up = self.shared_adapter.up.weight
        domain_down = self.domain_adapters[domain_name].down.weight
        domain_up = self.domain_adapters[domain_name].up.weight
        loss = (domain_down @ shared_down.T).norm() + (domain_up @ shared_up.T).norm()
        return loss


class DomainPositioner:
    def __init__(self, hidden_size):
        self.prototypes = {}
        self.hidden_size = hidden_size
        self.shared_cov = None

    def update_prototypes(self, domain_name, embeddings):
        mean = embeddings.mean(dim=0)
        self.prototypes[domain_name] = mean

    def compute_shared_covariance(self, all_embeddings):
        centered = [x - self.prototypes[domain] for domain, x in all_embeddings]
        stacked = torch.cat(centered, dim=0)
        self.shared_cov = torch.cov(stacked.T)

    def select_domain(self, embed):
        min_dist = float('inf')
        selected_domain = None
        for domain, proto in self.prototypes.items():
            diff = (embed - proto).unsqueeze(0)
            dist = (diff @ torch.linalg.pinv(self.shared_cov) @ diff.transpose(0, 1)).item()
            if dist < min_dist:
                min_dist = dist
                selected_domain = domain
        return selected_domain


def compute_lml_loss(outputs, labels):
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


# === Example Training Pipeline ===
if __name__ == "__main__":
    base_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    hidden_size = 768
    model = LLM_CLModel(base_model_name, hidden_size)

    domains = ["restaurant", "laptop"]
    for domain in domains:
        model.add_domain_adapter(domain)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Dummy data
    sentences = {"restaurant": ["The food was great!", "Terrible service."],
                 "laptop": ["Battery life is amazing.", "The screen is too dim."]}

    positioner = DomainPositioner(hidden_size)

    for domain in domains:
        model.train()
        inputs = tokenizer(sentences[domain], return_tensors="pt", padding=True, truncation=True)
        inputs_embeds = model.llm.transformer.wte(inputs['input_ids'])

        outputs = model(inputs_embeds, domain_name=domain)
        loss_lml = compute_lml_loss(outputs, inputs['input_ids'])
        loss_orth = model.orthogonal_loss(domain)
        loss = loss_lml + 1e-6 * loss_orth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update prototypes
        with torch.no_grad():
            hiddens = inputs_embeds.mean(dim=1)
            positioner.update_prototypes(domain, hiddens)

    # Compute shared covariance after training
    all_embeddings = [(domain, model.llm.transformer.wte(tokenizer(sentences[domain], return_tensors="pt", padding=True)['input_ids']).mean(dim=1)) for domain in domains]
    positioner.compute_shared_covariance(all_embeddings)

    # === Testing ===
    test_sentence = "The pizza was delicious but expensive."
    model.eval()
    inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True)
    inputs_embeds = model.llm.transformer.wte(inputs['input_ids'])
    embed = inputs_embeds.mean(dim=1)
    domain = positioner.select_domain(embed)

    outputs = model(inputs_embeds, domain_name=domain)
    print(f"Predicted domain: {domain}")
    print(f"Output shape: {outputs.logits.shape}")