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
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

class LLM_CL(nn.Module):
    def __init__(self, model, tokenizer, domain_names, out_features=3, rank=8, lora_alpha=16):
        super(LLM_CL, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        lora_share_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=["ffn.lin1", "ffn.lin2"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.shared_adapter = get_peft_model(model, lora_share_config)
        self.domain_adapters = {
            domain_name: get_peft_model(model, LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=["ffn.lin1", "ffn.lin2"],
                lora_dropout=0.1,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )) for domain_name in domain_names
        }
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, dropout=0.1)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(64, out_features)
        ).to(self.model.device)

        self.decoupler = DomainKnowledgeDecoupler(tokenizer, self.attention, self.classifier)
        self.warmup = DomainKnowledgeWarmup(tokenizer, self.attention, self.classifier)
        self.positioning = DomainPositioning(tokenizer, self.attention, self.classifier)

        for param in self.model.parameters():
            param.requires_grad = False

    def domain_variant_hidden(self, x, domain_name):
        hidden = self.decoupler(x, self.domain_adapters[domain_name])
        return hidden
    def domain_invariant_hidden(self, x_replay):
        hidden = self.decoupler(x_replay, self.shared_adapter)
        return hidden

    def warmup_knowledge(self, x_replay, domain_name):
        hidden = self.warmup(x_replay, self.shared_adapter, self.domain_adapters[domain_name])
        return hidden

    def prepare_finding(self, domain_data):
        return self.positioning.compute_prototypes(domain_data, self.shared_adapter)

    def find_best_domain_name(self, test_input):
        return self.positioning.find_best_domain(test_input, self.shared_adapter)

class DomainKnowledgeDecoupler:
    def __init__(self, tokenizer, attention, classifier):
        self.tokenizer = tokenizer
        self.attention = attention
        self.classifier = classifier

    def __call__(self, x, adapter):
        return self.forward(x, adapter)

    def forward(self, x, adapter):
        # Tokenize the input
        tokenized_input = self.tokenizer(x, return_tensors='pt', max_length=128, \
                                         truncation=True, padding=True).to(adapter.device)
        return self.get_hidden(tokenized_input, adapter)

    def get_hidden(self, tokenized_input, adapter):
        # Get the hidden states from the model
        outputs = adapter(**tokenized_input)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.permute(1, 0, 2)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = attn_output.permute(1, 0, 2)
        attn_pooled = attn_output.mean(dim=1)
        # Apply the adapter to the hidden states
        adapted_hidden_states = self.classifier(attn_pooled)
        return adapted_hidden_states

    def orthogonal_constraint(self, domain_adapter, shared_adapter):
        print(len([p for p in domain_adapter.parameters()]))
        domain_params = torch.cat([p.flatten() for p in domain_adapter.parameters()])
        shared_params = torch.cat([p.flatten() for p in shared_adapter.parameters()])
        orth_loss = torch.dot(domain_params, shared_params) ** 2
        return orth_loss

class DomainKnowledgeWarmup:
    def __init__(self, tokenizer, attention, classifier):
        self.tokenizer = tokenizer
        self.attention = attention
        self.classifier = classifier

    def __call__(self, x_replay, shared_adapter, domain_adapter):
        return self.forward(x_replay, shared_adapter, domain_adapter)

    def forward(self, x_replay, shared_adapter, domain_adapter):
        # Tokenize the input
        tokenized_input = self.tokenizer(x_replay, return_tensors='pt', max_length=128, \
                                         truncation=True, padding=True).to(shared_adapter.device)
        return self.get_hidden(tokenized_input, shared_adapter, domain_adapter)

    def get_hidden(self, tokenized_input, shared_adapter, domain_adapter):
        # Shared adapter output
        hidden_states = shared_adapter(**tokenized_input).last_hidden_state  # [batch_size, seq_len, 768]
        hidden_states = hidden_states.permute(1, 0, 2)  # [seq_len, batch_size, 768]
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, 768]
        attn_pooled = attn_output.mean(dim=1)  # [batch_size, 768]
        shared_output = self.classifier(attn_pooled)  # [batch_size, out_features]
        # Domain adapter output
        domain_hidden = domain_adapter(**tokenized_input).last_hidden_state  # [batch_size, seq_len, 768]
        domain_pooled = domain_hidden.mean(dim=1)  # [batch_size, 768]
        domain_output = self.classifier(domain_pooled)  # [batch_size, out_features]
        # Combine outputs
        hidden = shared_output + domain_output
        return hidden

class DomainPositioning:
    def __init__(self, tokenizer, attention, classifier):
        self.tokenizer = tokenizer
        self.attention = attention
        self.classifier = classifier
        self.domain_prototypes = {}
        self.covariance = None

    def compute_prototypes(self, domain_data, shared_adapter):
        reps = []
        for domain_name, samples in domain_data.items():
            embeddings = []
            for x, y in tqdm(samples, desc=f"Prepare finding for {domain_name}"):
                input_ids = self.tokenizer(x, return_tensors="pt", max_length=128, \
                                           truncation=True, padding=True).to(shared_adapter.device)
                hidden_states = self.get_hidden(input_ids, shared_adapter)
                embeddings.append(hidden_states.mean(dim=1))  # Mean pooling
            domain_rep = torch.stack(embeddings).mean(dim=0)
            self.domain_prototypes[domain_name] = domain_rep
            reps.extend(embeddings)

        reps_tensor = torch.stack(reps)
        diffs = reps_tensor - reps_tensor.mean(dim=0)
        self.covariance = torch.matmul(diffs.T, diffs) / len(reps_tensor)
        return self.covariance, self.domain_prototypes

    def find_best_domain(self, test_input, shared_adapter):
        input_ids = self.tokenizer(test_input, return_tensors="pt", max_length=128, \
                                   truncation=True, padding=True).to(shared_adapter.device)
        test_embed = self.get_hidden(input_ids, shared_adapter).mean(dim=1).squeeze()

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

    def get_hidden(self, input_ids, adapter):
        outputs = adapter(**input_ids)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.permute(1, 0, 2)
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = attn_output.permute(1, 0, 2)
        attn_pooled = attn_output.mean(dim=1)
        lora_output = self.classifier(attn_pooled)
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






