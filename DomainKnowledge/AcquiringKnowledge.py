# Domain knowledge decoupling module to learn a domain-invariant adapter (adapter_shared)
# with separate domain-variant adapters (adapter_domains).

# Domain Knowledge Warmup to leverage the replay data to fine-tune the domain-invariant adapter (adapter_shared)
# for each domain-variant adapters (adapter_domains) with frozen* domain-variant adapters.


#    +----------------------Domain Knowledge Warmup----------------------+
#    |  +==========================+       +=========================+   |
#    |  | +--------+   +--------+  |       | +--------+   +--------+ |   |   +----------------+
#    |  |  \  A₁* /     \  Aₛ   /   |      |  \  Aₙ* /     \  Aₛ   /   |   |   \  A ~ N(μ, σ²) /
#    |  |   +----+       +----+    |       |   +----+       +----+   |   |    +-------------+
#    |  |     x      +      x      | ....  |     x      +     x      |   |          x
#    |  |   +----+       +----+    |       |   +----+       +----+   |   |      +---------+
#    |  |  /  B₁* \     /  Bₛ   \   |      |  /  Bₙ* \     /  Bₛ   \   |   |     /   B = 0  \
#    |  | +---+----+   +---+----+  |       | +----+---+   +----+---+ |   |     +----------+
#    |  +======|============|======+       +======|============|=====+   |      This is Adapter
#    +---------|------------|---------------------|------------|---------+
#              V            V                     V            V        ^        * Frozen
#           +------------------+                +----------------+      |
#           |   Orthogonal     |                |   Orthogonal   |      |
#           |   Constraint     |                |   Constraint   |      |
#           +------------------+                +----------------+      |
#            ^             ^                     ^             ^        +-----+
#            |             |   Domain Knowledge  |             |              |
#   +--------|-------------|------Decoupling-----|-------------|---------+    |
#   |  +=====|=============|======+        +=====|=============|=======+ |    |
#   |  | +---+----+   +----+---+  |        | +---+----+   +---+---+   |  |    |
#   |  |  \  A₁  /     \  Aₛ   /   |       |   \  Aₙ  /     \  Aₛ   /   |  |    |
#   |  |   +----+       +----+    |        |   +----+       +----+    |  |    |
#   |  |     x      +      x      |  >...> |     x      +     x       |  |    |
#   |  |   +----+       +----+    |        |   +----+       +----+    |  |    |
#   |  |  /  B₁  \     /  Bₛ   \   |       |   /  Bₙ  \     /  Bₛ   \   |  |    |
#   |  | +---^----+   +---^----+  |        | +----^---+   +----^---+  |  |    |
#   |  +=====|============|=======+       +======|============|=======+  |    |
#   +------- |------------|----------------------|------------|----------+    |
#            |            |                      |            |               |
#       +----+-----+      |                +-----+----+       |               |
#       | Domain 1 |      |                | Domain N |       |               |
#       +----------+      |                +----------+       |               |
#                         |                                   |               |
#       +-----------------+--+                                |               |
#       |     Replay 1       |                                |               |
#       +--------------------+                                |               |
#                                                             |               |
#       +-----------------------------------------------------+----+          |
#       |                     Replay N                             |----------+
#       +----------------------------------------------------------+


import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

class DomainKnowledgeDecoupler:
    def __init__(self, shared_adapter, domain_adapters, lambda_orth=1e-6):
        self.shared_adapter = shared_adapter  # Adapter
        self.domain_adapters = domain_adapters  # dict[DomainName, Adapter]
        self.lambda_orth = lambda_orth

    def compute_loss(self, domain_name, domain_data, replay_data, model, tokenizer):
        adapter_d = self.domain_adapters[domain_name]
        adapter_s = self.shared_adapter

        loss_d = 0.0
        for x, y in domain_data:
            input_ids = tokenizer(x).to(model.device)
            labels = torch.tensor(y).to(model.device)
            outputs = self.get_output(model, input_ids, adapter_d)
            loss_d += F.cross_entropy(outputs.logits, labels)

        loss_s = 0.0
        for samples in replay_data.values():
            for x, y in samples:
                input_ids = tokenizer(x).to(model.device)
                labels = torch.tensor(y).to(model.device)
                outputs = self.get_output(model, input_ids, adapter_s)
                loss_s += F.cross_entropy(outputs.logits, labels)

        orth_loss = self.orthogonal_loss(adapter_d.lora_A, adapter_d.lora_B,
                                         adapter_s.lora_A, adapter_s.lora_B)

        total_loss = loss_d + loss_s + self.lambda_orth * orth_loss
        return total_loss

    def get_output(self, base_model, input_ids, adapter):
        outputs = base_model(input_ids=input_ids)
        hiden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
        lora_output = adapter(hiden_states)
        return lora_output

class DomainKnowledgeWarmup:
    def __init__(self, shared_adapter, domain_adapters):
        self.shared_adapter = shared_adapter
        self.domain_adapters = domain_adapters

    def warmup(self, replay_data, model, tokenizer, optimizer,
               num_epochs=10, batch_size=16, log_path="warmup_log.txt"):
        # Prepare the replay dataset
        # replay_data: dict[DomainName -> list of (x, y)]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "a")

        replay_dataset = []
        for domain_name, samples in replay_data.items():
            for x, y in samples:
                replay_dataset.append((x, y, domain_name))

        dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(num_epochs):
            print(f"Warmup Epoch: {epoch + 1}/{num_epochs}")
            train_loss = 0.0

            for x_batch, y_batch, domain_batch in tqdm(dataloader):
                avg_train_batch_loss = 0.0
                # Train each sample in the batch to get the domain adapter
                for x, y, domain_name in zip(x_batch, y_batch, domain_batch):
                    adapter_d = self.domain_adapters[domain_name]
                    adapter_d.requires_grad_(False)
                    self.shared_adapter.requires_grad_(True)

                    input_ids = tokenizer(x).to(model.device)
                    labels = torch.tensor(y).to(model.device)
                    outputs = self.get_output(model, input_ids, adapter_d, self.shared_adapter)
                    loss = F.cross_entropy(outputs.logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_batch_loss += train_loss / len(domain_batch)
            avg_train_loss = avg_train_batch_loss / len(dataloader)
            print(f"Warmup Epoch {epoch + 1} Loss: {avg_train_loss}")
            log_file.write(f"Warmup Epoch {epoch + 1} Loss: {avg_train_loss}\n")
            log_file.flush()

        log_file.write("Warmup completed.\n")
        log_file.flush()
        log_file.close()

    def get_output(self, base_model, input_ids, adapter_domain, adapter_shared):
        # Get the output of the model with the given adapter
        outputs = base_model(input_ids=input_ids)
        hiden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.logits
        lora_output = adapter_domain(hiden_states) + adapter_shared(hiden_states)
        return lora_output