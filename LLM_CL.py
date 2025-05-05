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
                optimizer.step()
                train_on_domain_loss += loss.item()

            avg_train_on_domain_loss = train_on_domain_loss / len(train_loader)
            print(f"Training on domain {domain_name} loss: {avg_train_on_domain_loss}")

            print(f"Validation on domain: {domain_name}, Epoch: {epoch + 1}/{self.decoupler_epochs}")
            val_on_domain_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in tqdm(val_loader):
                    input_ids = self.tokenizer(x_val, return_tensors="pt").to(self.model.device)
                    labels = torch.tensor(y_val).to(self.model.device)
                    adapter_d = self.domain_adapters[domain_name]
                    outputs = self.get_hidden(self.model, input_ids, adapter_d)
                    print(f"Outputs: {outputs.shape}, Labels: {labels.shape}")
                    val_on_domain_loss += F.cross_entropy(outputs, labels).item()

            avg_val_on_domain_loss = val_on_domain_loss / len(val_loader)
            print(f"Validation on domain {domain_name} loss: {avg_val_on_domain_loss}")

            log_msg = f"[Domain: {domain_name}] Epoch {epoch + 1}: \
                        Train Loss = {avg_train_on_domain_loss:.4f}, \
                        Val Loss = {avg_val_on_domain_loss:.4f}"
            log_file.write(log_msg + "\n")
            log_file.flush()

        log_file.close()

        # Updating replay buffer (after training on the domain)
        self.replay_data[domain_name] = train_domain_data[:self.replay_size]

    def get_hidden(self, base_model, input_ids, adapter):
        # Get the output of the model with the given adapter
        outputs = base_model(**input_ids)

        hiden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs.pooler_output
        lora_output = adapter(hiden_states)
        return lora_output

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
        input_ids, _ = self.tokenizer(x, return_tensors="pt").to(self.model.device)
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

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from Adapters.LoRA import LoRAAdapter

    base_model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    log_on_domain_path = "on_domain.txt"

    shared_adapter = LoRAAdapter(in_features=732, out_features=3, rank=8) # Initialize your shared adapter here
    domain_adapters = {
        "restaurant": LoRAAdapter(in_features=732, out_features=3, rank=8),  # Initialize your restaurant adapter here
        "laptop": LoRAAdapter(in_features=732, out_features=3, rank=8),  # Initialize your laptop adapter here
        # Add more domain adapters as needed
    }

    llm_cl = LLM_CL(
        model=model,
        tokenizer=tokenizer,
        shared_adapter=shared_adapter,
        domain_adapters=domain_adapters,
        lambda_orth=1e-6,
        warmup_epochs=5,
        replay_size=8
    )

    # Dummy data for training and evaluation
    train_domain_data = {
        "restaurant": [("The food was great!", 1), ("Terrible service.", 0)],
        "laptop": [("Battery life is amazing.", 1), ("The screen is too dim.", 0)]
    }
    val_domain_data = {
        "restaurant": [("I will never come back.", 0), ("Loved the ambiance!", 1)],
        "laptop": [("Very fast performance.", 1), ("Not worth the price.", 0)]
    }
    test_data = [("I love this restaurant!", 1), ("This laptop is terrible.", 0)]

    for domain_name, data in train_domain_data.items():
        optimizer = torch.optim.AdamW(
            list(llm_cl.shared_adapter.parameters()) + list(llm_cl.domain_adapters[domain_name].parameters()),
            lr=1e-4
        )
        llm_cl.train_on_domain(domain_name, data, val_domain_data[domain_name], optimizer,
                               log_path=log_on_domain_path)

    os.makedirs(os.path.dirname(log_on_domain_path), exist_ok=True)
    log_file = open(log_on_domain_path, "a")
    log_file.write("END OF TRAINING\n")
    log_file.flush()
    log_file.close()
    optimizer = torch.optim.AdamW(llm_cl.shared_adapter.parameters(), lr=1e-4)
    llm_cl.warmup_shared_adapter(optimizer)
    llm_cl.prepare_for_inference()

    preds, labels = llm_cl.evaluate(test_data)
    print(f"Predictions: {preds}, Labels: {labels}")