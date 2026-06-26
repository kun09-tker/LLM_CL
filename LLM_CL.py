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
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import DebertaV2Tokenizer
from PLMs.DebertaV2 import MyDebertaV2ForSequenceClassification
from Utils.distances import mahalanobis_distance
from Utils.processors import AscProcessor

ASC = AscProcessor()

class LLM_CL(nn.Module):
    def __init__(self, domain_names, tokenizer_path, rank_domain=8, alpha_domain=16, rank_share=8, alpha_share=16, orth_lambda=0.1,
                 base_name = "yangheng/deberta-v3-base-absa-v1.1",
                 device='cpu'):
        super(LLM_CL, self).__init__()
        self.base = MyDebertaV2ForSequenceClassification.from_pretrained(
                base_name,
                domain_names = domain_names,
                rank_domain=rank_domain, alpha_domain=alpha_domain,
                rank_share=rank_share, alpha_share=alpha_share, orth_lambda=orth_lambda
            )
        self.base.to(device)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer_path)

        self.decoupler = DomainKnowledgeDecoupler(self.tokenizer)
        self.warmup = DomainKnowledgeWarmup(self.tokenizer)
        self.positioning = DomainPositioning(self.tokenizer)

    def get_emb(self, x_batch):
        texts = []
        for x in x_batch:
            texts.append(ASC.get_input_sep(x))
        tokenized_input = self.tokenizer(texts, max_length=512, return_tensors='pt', \
                                         truncation=True, padding=True).to(self.base.device)
        return self.base(**tokenized_input, return_emb=True)

    def domain_variant_hidden(self, x_batch, domain_name):
        hidden = self.decoupler(x_batch, self.base, domain_name)
        return hidden

    # def domain_invariant_hidden(self, x_replay):
    #     hidden = self.decoupler(x_replay, self.base)
    #     return hidden

    def prepare_warmup(self):
        self.warmup.prepare_warmup(self.base)

    def warmup_knowledge(self, x_replay):
        hidden = self.warmup(x_replay, self.base)
        return hidden

    def prepare_finding(self, domain_data):
        return self.positioning.compute_prototypes(domain_data, self.base)

    def find_best_domain_name(self, x_batch_test):
        return self.positioning.find_best_domain(x_batch_test, self.base)

    def predict(self, domain_name, sample):
        text = [ASC.get_input_sep(sample)]
        tokenized_input = self.tokenizer(text, max_length=512, return_tensors='pt', \
                                         truncation=True, padding=True).to(self.base.device)
        return self.base(**tokenized_input, domain_name=domain_name)


class DomainKnowledgeDecoupler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, x, model, domain_name=None):
        return self.forward(x, model, domain_name)

    def forward(self, x_batch, model, domain_name):
        texts = []
        labels = []
        for x in x_batch:
            texts.append(ASC.get_input_sep(x))
            labels.append(ASC.get_label_classifier(x))

        tokenized_input = self.tokenizer(texts, max_length=512, return_tensors='pt', \
                                         truncation=True, padding=True).to(model.device)
        labels = torch.tensor(labels).to(model.device)
        tokenized_input["labels"] = labels

        return self.get_hidden(tokenized_input, model, domain_name), labels

    def get_hidden(self, tokenized_input, model, domain_name):
        return model(**tokenized_input, domain_name=domain_name)

class DomainKnowledgeWarmup:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_warmup(self, model):
        for param in model.invariant_apdater.parameters():
            param.requires_grad = True
        for name in model.domain_names:
          for param in model.variant_apdater[name].parameters():
              param.requires_grad = False

    def __call__(self, x_replay, model):
        return self.forward(x_replay, model)

    def forward(self, x_replay_batch, model):
        texts = []
        labels = []
        for x in x_replay_batch:
            texts.append(ASC.get_input_sep(x))
            labels.append(ASC.get_label_classifier(x))

        tokenized_input = self.tokenizer(texts, max_length=512, return_tensors='pt', \
                                         truncation=True, padding=True).to(model.device)
        labels = torch.tensor(labels).to(model.device)
        tokenized_input["labels"] = labels

        return self.get_hidden(tokenized_input, model), labels

    def get_hidden(self, tokenized_input, model):
        return model(**tokenized_input)

class DomainPositioning:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.domain_prototypes = {}
        self.covariance = None

    def compute_prototypes(self, domain_data, model):
        reps = []
        for domain_name, samples in domain_data.items():
            embeddings = []
            for x in tqdm(samples, desc=f"Prepare finding for {domain_name}"):
                text = ASC.get_input_sep(x)
                tokenized_input = self.tokenizer(text, max_length=512, return_tensors='pt', \
                                                truncation=True, padding=True).to(model.device)

                hidden_states = self.get_hidden(tokenized_input, model)
                hidden_states = hidden_states[:, 0, :].cpu().detach().numpy()
                embeddings.append(hidden_states)

            mean = np.mean(np.concatenate(embeddings, axis=0), axis=0)
            self.domain_prototypes[domain_name] = mean
            reps.extend(embeddings)

        reps_tensor = np.concatenate(reps, axis=0)
        covariance = np.cov(reps_tensor.T)
        self.covariance = np.linalg.inv(covariance + 1e-6 * np.eye(covariance.shape[0]))
        return self.covariance, self.domain_prototypes

    def find_best_domain(self, test_input, model):
        text = ASC.get_input_sep(test_input)
        tokenized_input = self.tokenizer(text, max_length=512, return_tensors='pt', \
                                        truncation=True, padding=True).to(model.device)
        test_embed = self.get_hidden(tokenized_input, model)
        test_embed = test_embed[0, 0, :].cpu().detach().numpy()

        distances = {domain: mahalanobis_distance(test_embed, mean_i, self.covariance)
                 for domain, mean_i in self.domain_prototypes.items()}

        selected_domain = min(distances, key=distances.get)

        return selected_domain

    def get_hidden(self, tokenized_input, model):
        return model(**tokenized_input).hidden_states[-1]




