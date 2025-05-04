import torch

class DomainPositioning:
    def __init__(self, model, domain_adapters, shared_adapter):
        self.model = model
        self.domain_adapters = domain_adapters
        self.shared_adapter = shared_adapter
        self.domain_prototypes = {}
        self.covariance = None

    def compute_prototypes(self, domain_data, tokenizer):
        reps = []
        for domain_name, samples in domain_data.items():
            embeddings = []
            for x, y in samples:
                input_ids, _ = tokenizer(x, return_tensors="pt").to(self.model.device)
                hidden_states = self.model.get_hidden(input_ids, adapter=self.shared_adapter)
                embeddings.append(hidden_states.mean(dim=1))  # Mean pooling
            domain_rep = torch.stack(embeddings).mean(dim=0)
            self.domain_prototypes[domain_name] = domain_rep
            reps.extend(embeddings)

        reps_tensor = torch.stack(reps)
        diffs = reps_tensor - reps_tensor.mean(dim=0)
        self.covariance = torch.matmul(diffs.T, diffs) / len(reps_tensor)

    def find_best_domain(self, test_input, tokenizer):
        input_ids, _ = tokenizer(test_input, None)
        test_embed = self.model.get_hidden(input_ids, adapter=self.shared_adapter).mean(dim=1).squeeze()

        best_score = -float('inf')
        best_domain = None
        cov_inv = torch.linalg.pinv(self.covariance)

        for domain_name, proto in self.domain_prototypes.items():
            diff = test_embed - proto
            score = -diff.T @ cov_inv @ diff
            if score > best_score:
                best_score = score
                best_domain = domain_name

        return best_domain, self.domain_adapters[best_domain]
