import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from DomainKnowledge.SelectKnowledge import DomainPositioning
from DomainKnowledge.AcquiringKnowledge import DomainKnowledgeDecoupler, DomainKnowledgeWarmup


class LLM_CL:
    def __init__(self, model, tokenizer, shared_adapter, domain_adapters,
                 lambda_orth=1e-6, warmup_epochs=5, decoupler_epochs=5, batch_size=16,
                 replay_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.shared_adapter = shared_adapter
        self.domain_adapters = domain_adapters # dict[DomainName -> Adapter]
        self.lambda_orth = lambda_orth
        self.warmup_epochs = warmup_epochs
        self.decoupler_epochs = decoupler_epochs
        self.batch_size = batch_size
        self.replay_size = replay_size

        self.decoupler = DomainKnowledgeDecoupler(
            shared_adapter=self.shared_adapter,
            domain_adapters=self.domain_adapters,
            lambda_orth=self.lambda_orth
        )
        self.warmup = DomainKnowledgeWarmup(
            shared_adapter=self.shared_adapter,
            domain_adapters=self.domain_adapters
        )
        self.positioner = DomainPositioning(
            model=self.model,
            domain_adapters=self.domain_adapters,
            shared_adapter=self.shared_adapter
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.replay_data = {}  # dict[DomainName -> list of (x, y)]

    def train_on_domain(self, domain_name, train_domain_data, val_domain_data, optimizer,
                        log_path="train_on_domain.txt"):

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "a")

        train_loader = DataLoader(train_domain_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_domain_data, batch_size=self.batch_size, shuffle=False)

        self.model.train()
        for epoch in range(self.decoupler_epochs):
            print(f"Training on domain: {domain_name}, Epoch: {epoch + 1}/{self.decoupler_epochs}")
            train_on_domain_loss = 0.0

            for x_batch, y_batch in tqdm(train_loader):
                batch_data = list(zip(x_batch, y_batch))

                optimizer.zero_grad()

                loss = self.decoupler.compute_loss(
                    domain_name=domain_name,
                    domain_data=batch_data,
                    replay_data=self.replay_data,
                    model=self.model,
                    tokenizer=self.tokenizer
                )

                loss.backward()
                optimizer.step()
                train_on_domain_loss += loss.item()

            avg_train_on_domain_loss = train_on_domain_loss / len(train_loader)
            print(f"Training on domain {domain_name} loss: {avg_train_on_domain_loss}")

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    batch_data = list(zip(x_batch, y_batch))

                    loss = self.decoupler.compute_loss(
                        domain_name=domain_name,
                        domain_data=batch_data,
                        replay_data=self.replay_data,
                        model=self.model,
                        tokenizer=self.tokenizer
                    )
                    val_loss += loss.item()

            avg_val_on_domain_loss = val_loss / len(val_loader)
            print(f"Validation on domain {domain_name} loss: {avg_val_on_domain_loss}")

            log_msg = f"[Domain: {domain_name}] Epoch {epoch + 1}: \
                        Train Loss = {avg_train_on_domain_loss:.4f}, \
                        Val Loss = {avg_val_on_domain_loss:.4f}"
            log_file.write(log_msg + "\n")
            log_file.flush()

        log_file.close()

        # Updating replay buffer (after training on the domain)
        self.replay_data[domain_name] = train_domain_data[:self.replay_size]

    def warmup_shared_adapter(self, optimizer):
        # ===> Warmup using all replay data to align invariant adapter
        self.warmup.warmup(
            replay_data=self.replay_data,
            model=self.model,
            tokenizer=self.tokenizer,
            optimizer=optimizer,
            num_epochs=self.warmup_epochs,
            batch_size=self.batch_size
        )

    def prepare_for_inference(self):
        # ===> Compute domain prototypes for domain positioning
        self.positioner.compute_prototypes(self.replay_data, self.tokenizer)

    def predict(self, x):
        # ===> Inference with automatic domain adapter selection
        best_domain, best_adapter = self.positioner.find_best_domain(x, self.tokenizer)
        input_ids, _ = self.tokenizer(x, return_tensors="pt", max_length=128, \
                                      truncation=True, padding=True).to(self.model.device)
        outputs = self.get_hidden(self.model, input_ids, best_adapter)
        return torch.argmax(outputs, dim=-1)

    def evaluate(self, test_data):
        preds = []
        labels = []
        for x, y in test_data:
            pred = self.predict(x)
            preds.append(pred.item())
            labels.append(y)
        return preds, labels