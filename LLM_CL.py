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



import torch
import random
import torch.nn as nn
from Adapters.LoRA import LoRAAdapter

class LLM_CL(nn.Module):
    def __init__(self, model, tokenizer, domain_names, out_features=3, rank=8):
        super(LLM_CL, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.shared_adapter = LoRAAdapter(model.config.hidden_size, out_features=out_features, rank=rank).to(model.device)
        self.domain_adapters = {domain_name: LoRAAdapter(model.config.hidden_size, out_features=out_features, rank=rank).to(model.device)
                                for domain_name in domain_names}

        self.decoupler = DomainKnowledgeDecoupler(tokenizer)
        self.warmup = DomainKnowledgeWarmup(tokenizer)
        self.positioning = DomainPositioning(tokenizer)

        for param in self.model.parameters():
            param.requires_grad = False

    def domain_variant_hidden(self, x, domain_name):
        hidden = self.decoupler(
                            x, self.model,
                            self.domain_adapters[domain_name]
                        )
        return hidden
    def domain_invariant_hidden(self, x_replay):
        hidden = self.decoupler(
                            x_replay, self.model,
                            self.shared_adapter,
                        )
        return hidden

    def warmup_knowledge(self, x_replay, domain_name):
        hidden = self.warmup(
                            x_replay, self.model,
                            self.shared_adapter,
                            self.domain_adapters[domain_name]
                        )
        return hidden

    def prepare_finding(self, domain_data):
        self.positioning.compute_prototypes(domain_data, self.model, self.shared_adapter)

    def find_best_domain_name(self, test_input):
        return self.positioning.find_best_domain(test_input, self.model, self.shared_adapter)

class DomainKnowledgeDecoupler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, x, model, adapter):
        return self.forward(x, model, adapter)

    def forward(self, x, model, adapter):
        # Tokenize the input
        tokenized_input = self.tokenizer(x, return_tensors='pt').to(model.device)
        return self.get_hidden(tokenized_input, model, adapter)

    def get_hidden(self, tokenized_input, model, adapter):
        # Get the hidden states from the model
        outputs = model(**tokenized_input)
        hidden_states = outputs.pooler_output
        # Apply the adapter to the hidden states
        adapted_hidden_states = adapter(hidden_states)
        return adapted_hidden_states

    def orthogonal_constraint(self, domain_adapter, shared_adapter):
        A_orth = torch.matmul(domain_adapter.lora_A.T, shared_adapter.lora_A)
        B_orth = torch.matmul(domain_adapter.lora_B.T, shared_adapter.lora_B)

        A_orth = torch.norm(A_orth, p='fro')
        B_orth = torch.norm(B_orth, p='fro')

        orthogonal_loss = A_orth ** 2 + B_orth ** 2
        return orthogonal_loss

class DomainKnowledgeWarmup:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, x_replay, model, shared_adapter, domain_adapter):
        return self.forward(x_replay, model, shared_adapter, domain_adapter)

    def forward(self, x_replay, model, shared_adapter, domain_adapter):
        # Tokenize the input
        tokenized_input = self.tokenizer(x_replay, return_tensors='pt').to(model.device)
        return self.get_hidden(tokenized_input, model, shared_adapter, domain_adapter)

    def get_hidden(self, tokenized_input, model, shared_adapter, domain_adapter):
        # Get the hidden states from the model
        output = model(**tokenized_input).pooler_output
        hidden = shared_adapter(output) + domain_adapter(output)
        return hidden

class DomainPositioning:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.domain_prototypes = {}
        self.covariance = None

    def compute_prototypes(self, domain_data, model, shared_adapter):
        reps = []
        for domain_name, samples in domain_data.items():
            embeddings = []
            for x, y in samples:
                input_ids = self.tokenizer(x, return_tensors="pt").to(model.device)
                hidden_states = self.get_hidden(input_ids, shared_adapter)
                embeddings.append(hidden_states.mean(dim=1))  # Mean pooling
            domain_rep = torch.stack(embeddings).mean(dim=0)
            self.domain_prototypes[domain_name] = domain_rep
            reps.extend(embeddings)

        reps_tensor = torch.stack(reps)
        diffs = reps_tensor - reps_tensor.mean(dim=0)
        self.covariance = torch.matmul(diffs.T, diffs) / len(reps_tensor)

    def find_best_domain(self, test_input, model, shared_adapter):
        input_ids = self.tokenizer(**test_input).to(model.device)
        test_embed = self.get_hidden(model, input_ids, shared_adapter).mean(dim=1).squeeze()

        best_score = -float('inf')
        best_domain = None
        cov_inv = torch.linalg.pinv(self.covariance)

        for domain_name, proto in self.domain_prototypes.items():
            diff = test_embed - proto
            score = -diff.T @ cov_inv @ diff
            if score > best_score:
                best_score = score
                best_domain = domain_name

        return best_domain

    def get_hidden(self, model, input_ids, adapter):
        outputs = model(**input_ids).to(model.device)
        hidden_states = outputs.pooler_output
        lora_output = adapter(hidden_states)
        return lora_output

if __name__ == "__main__":
    # Example usage
    import os
    import torch
    import random
    import torch.nn as nn
    from tqdm import tqdm
    from transformers import BertTokenizer, BertModel
    from sklearn.metrics import accuracy_score, f1_score

        # Define constants
    EPOCHS_D = 30
    EPOCHS_W = 10
    RANK = 8
    LEARNING_RATE_D = 1e-6
    LEARNING_RATE_W = 1e-5
    LAMDA_ORTH = 1e-6
    REPLAY_SIZE = 10
    LOG_TRAINING_PATH = "training_log.txt"
    CHECKPOIN_PATH = ""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_metrics(preds, labels):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")  # hoặc "macro", tùy bài toán
        return acc, f1

    def embed_label(label, device):
        if label == 1:
            return torch.tensor([1]).to(device)
        elif label == -1:
            return torch.tensor([2]).to(device)
        elif label == 0:
            return torch.tensor([0]).to(device)
        else:
            raise ValueError("Invalid label. Label must be 1, -1, or 0.")

    def split_into_random_chunks(lst, chunk_size=3):
        shuffled = lst[:]         # sao chép danh sách gốc
        random.shuffle(shuffled)  # xáo trộn ngẫu nhiên

        chunks = [shuffled[i:i+chunk_size] for i in range(0, len(shuffled), chunk_size)]
        return chunks

    def filter_domains(data, selected_domains):
        return {domain: data[domain] for domain in selected_domains if domain in data}

    def traning_llm_cl(train_data, val_data, test_data, device=DEVICE):
        # Initialize tokenizer and model
        base_model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(base_model_name)
        model = BertModel.from_pretrained(base_model_name)
        model.to(device)

        test_data = [sample for _, sample in test_data.items()]

        # Replay data
        replay_data = {}

        # Initialize LLM_CL
        domain_names = list(train_data.keys())
        llm_cl = LLM_CL(model, tokenizer, domain_names, rank=RANK)
        model_path = os.path.join(CHECKPOIN_PATH, "_".join(domain_names) + ".pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            llm_cl.load_state_dict(checkpoint['model_state_dict'])
            BEST_F1 = checkpoint['best_f1']
        else:
            BEST_F1 = 0.0
        llm_cl.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer_decoupler = torch.optim.Adam(filter(lambda p: p.requires_grad, llm_cl.parameters()), lr=LEARNING_RATE_D)

        # Training loop
        if not os.path.exists(LOG_TRAINING_PATH):
            os.makedirs(LOG_TRAINING_PATH, exist_ok=True)
        log_file = open(LOG_TRAINING_PATH, "a")
        msg = f"\n================\n {'_'.join(domain_names)} \n================\n"
        print(msg)
        log_file.write(msg)

        # Step 1: Domain Knowledge Decoupling
        def handle_domain_knowledge_decoupling_step(model, domain_data, replay_data, extend_replay = True):

            mode = "Training"
            model.train()
            if not extend_replay:
                model.val()
                mode = "Validating"

            for domain_name, data in domain_data.items():
                loss_d = []
                for text, label in tqdm(data, desc=f"{mode} domain variant for {domain_name}"):
                    # Domain Knowledge Warmup
                    domain_variant_hidden = model.domain_variant_hidden(text, domain_name)
                    loss = criterion(domain_variant_hidden, label)
                    loss_d.append(loss.item())

                for domain_name, data in replay_data.items():
                    loss_s = []
                    for text, label in tqdm(data, desc=f"{mode} domain invariant for {domain_name}"):
                        # Domain Knowledge Warmup
                        domain_invariant_hidden = model.domain_invariant_hidden(text)
                        loss = criterion(domain_invariant_hidden, label)
                        loss_s.append(loss.item())

                if extend_replay:
                    replay_data[domain_name] = random.sample(data, k=REPLAY_SIZE)

            # Orthogonal constraint
            orthogonal_loss = model.decoupler.orthogonal_constraint(
                model.domain_adapters[domain_name],
                model.shared_adapter
            )
            loss = sum(loss_d) + sum(loss_s) + orthogonal_loss * LAMDA_ORTH

            return loss, replay_data

        for epoch in range(EPOCHS_D):
            optimizer_decoupler.zero_grad()
            loss, replay_data =handle_domain_knowledge_decoupling_step(
                                    llm_cl, train_data, replay_data, extend_replay = True)

            loss.backward()
            optimizer_decoupler.step()

            msg = f"Step1 - Epoch {epoch + 1}/{EPOCHS_D}, Loss train: {loss.item()}"

            with torch.no_grad():
                loss, replay_data =handle_domain_knowledge_decoupling_step(
                                    llm_cl, val_data, replay_data, extend_replay = False)

            msg += f"\t Loss val: {loss.item()}"
            print(msg)
            log_file.write(msg + "\n")

        # Step 2: Domain Knowledge Warmup
        for domain_name, data in train_data.items():
            for param in llm_cl.domain_adapters[domain_name].parameters():
                param.requires_grad = False

        for param in llm_cl.shared_adapter.parameters():
            param.requires_grad = True

        optimizer_warmup = torch.optim.Adam(filter(lambda p: p.requires_grad, llm_cl.parameters()), lr=LEARNING_RATE_W)

        for epoch in range(EPOCHS_W):
            total_warmup_loss = 0.0

            for domain_name, data in replay_data.items():
                loss_warmup = 0.0
                for text, label in tqdm(data, desc=f"Warming up domain invariant at {domain_name}"):
                    # Domain Knowledge Warmup
                    optimizer_warmup.zero_grad()
                    warmup_hidden = llm_cl.warmup_knowledge(text, domain_name)
                    loss = criterion(warmup_hidden, label)
                    loss.backward()
                    loss_warmup += loss.item()
                    optimizer_warmup.step()

                total_warmup_loss += loss_warmup

            msg = f"Step2 - Epoch {epoch + 1}/{EPOCHS_W}, Warmup Loss: {total_warmup_loss}"
            print(msg)
            log_file.write(msg + "\n")

        # Test step
        llm_cl.prepare_finding(train_data)
        loss_test = 0.0
        all_preds = []
        all_labels = []
        for text, label in tqdm(test_data, desc="Testing"):
            domain_name = llm_cl.find_best_domain_name(text)
            output = llm_cl.domain_variant_hidden(text, domain_name)
            loss_test += criterion(output, label).items()
            pred = torch.argmax(output, dim=1)
            all_preds.append(pred)
            all_labels.append(label)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc, f1 = compute_metrics(all_preds, all_labels)
        msg = f"Step 3 Loss test: {loss_test / len(test_data)} - Acc test: {acc} - F1_Macro: {f1}"
        print(msg)
        log_file.write(msg + "\n")

        if f1 > BEST_F1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_f1': f1
            }, model_path)

            msg = f" Saved best model with F1 = {f1:.4f} \n"
        else:
            msg = f" Don't save: F1 = {f1:.4f} < best_f1 = {BEST_F1} \n"

        print(msg)
        log_file.write(msg)
        log_file.close()


        # Data
    train_data = {
        'domain1': [('text1', 0), ('text2', 1), ('text3', -1)],
        'domain2': [('text4', -1), ('text5', 1), ('text6', 0)],
        'domain3': [('text7', 0), ('text8', 1), ('text9', -1)],
        'domain4': [('text10', 1), ('text11', -1), ('text12', 0)]
    }

    val_data = {
        'domain1': [('text1', 0), ('text2', 1), ('text3', -1)],
        'domain2': [('text4', -1), ('text5', 1), ('text6', 0)],
        'domain3': [('text7', 0), ('text8', 1), ('text9', -1)],
        'domain4': [('text10', 1), ('text11', -1), ('text12', 0)]
    }

    test_data = {
        'domain1': [('text1', 0), ('text2', 1), ('text3', -1)],
        'domain2': [('text4', -1), ('text5', 1), ('text6', 0)],
        'domain3': [('text7', 0), ('text8', 1), ('text9', -1)],
        'domain4': [('text10', 1), ('text11', -1), ('text12', 0)]
    }

    for domain_name, data in train_data.items():
        train_data[domain_name] = [(text, embed_label(label, DEVICE)) for text, label in data]

    for domain_name, data in val_data.items():
        val_data[domain_name] = [(text, embed_label(label, DEVICE)) for text, label in data]

    for domain_name, data in test_data.items():
        test_data[domain_name] = [(text, embed_label(label, DEVICE)) for text, label in data]

    domain_names = list(train_data.keys())
    chunk_domain_names = split_into_random_chunks(domain_names)

    for chunk in chunk_domain_names:
        if len(chunk) > 3:
            traning_llm_cl(filter_domains(train_data, chunk),
                        filter_domains(val_data, chunk),
                        filter_domains(test_data, chunk))






