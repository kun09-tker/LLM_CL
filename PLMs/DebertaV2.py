import torch
import torch.nn as nn

from Layers.LoRA import LoRAApdater
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from transformers import DebertaV2Model, DebertaV2ForSequenceClassification

class MyDebertaV2Model(DebertaV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions \
          if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict \
          if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

        # Get embeddings
        embeddings_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds
        )

        # Pass through encoder
        encoder_outputs = self.encoder(
            embeddings_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )

        if not return_dict:
            encoded_layers = encoder_outputs[1]
            sequence_output = encoded_layers[-1]
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        encoded_layers = encoder_outputs.hidden_states
        sequence_output = encoded_layers[-1]
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class MyDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config, domain_names, rank_domain=8, alpha_domain=16, rank_share=8, alpha_share=16):
        super().__init__(config)
        self.config = config
        self.deberta = MyDebertaV2Model(config)
        self.domain_names = domain_names
        self.invariant_apdater = LoRAApdater("LoRA_share", in_features=self.config.hidden_size, out_features=self.config.hidden_size, rank=rank_share, alpha=alpha_share)
        self.variant_apdater = nn.ModuleDict({
            name: LoRAApdater(f"LoRA_{name}", in_features=self.config.hidden_size, out_features=self.config.hidden_size, rank=rank_domain, alpha=alpha_domain)
                for name in domain_names})
        # self.classifier = nn.ModuleDict({
        #     name: nn.Sequential(
        #       nn.Dropout(0.2),
        #       nn.Linear(self.config.hidden_size, 512),
        #       ACT2FN[self.config.pooler_hidden_act],
        #       nn.Linear(512, 128),
        #       nn.ReLU(),
        #       nn.Linear(128, self.config.num_labels)
        #     #   nn.Softmax(dim=1)
        #     )
        #     for name in domain_names})
        # self.classifier = nn.ModuleDict({
        #     name: nn.Sequential(
        #     #   nn.BatchNorm1d(self.config.hidden_size),
        #       nn.Dropout(0.2),
        #       nn.Linear(self.config.hidden_size, 512),
        #       nn.ReLU(),
        #       nn.Linear(512, 128),
        #       ACT2FN[self.config.pooler_hidden_act],
        #       nn.Linear(128, self.config.num_labels),
        #       nn.Softmax(dim=1)
        #     )
        #     for name in domain_names})
        # self.classifier_share = nn.Sequential(
        #     # nn.BatchNorm1d(self.config.hidden_size),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.config.hidden_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128),
        #     ACT2FN[self.config.pooler_hidden_act],
        #     nn.Linear(128, self.config.num_labels),
        #     nn.Softmax(dim=1)
        # )
        # self.classifier_share = nn.Sequential(
        #       nn.Dropout(0.2),
        #       nn.Linear(self.config.hidden_size, 512),
        #       ACT2FN[self.config.pooler_hidden_act],
        #       nn.Linear(512, 128),
        #       nn.ReLU(),
        #       nn.Linear(128, self.config.num_labels)
        #     #   nn.Softmax(dim=1)
        # )
        for param in self.parameters():
            param.requires_grad = False
        self.post_init()
    def freeze_or_unfreeze(self, backbone=False, finetun=True):
        print(f"Chek status:\n\t Pretraining trainable: {backbone}\n\t Finetuning trainable: {finetun}\n")
        for param in self.parameters():
            param.requires_grad = backbone

        for param in self.invariant_apdater.parameters():
            param.requires_grad = finetun
        for name in self.domain_names:
          for param in self.variant_apdater[name].parameters():
              param.requires_grad = finetun

        # for param in self.classifier.parameters():
        #     param.requires_grad = finetun
        # for param in self.classifier_share.parameters():
        #     param.requires_grad = finetun

    def freeze_backbone_unfreeze_finetun(self):
        self.freeze_or_unfreeze(backbone=False, finetun=True)

    def unfreeze_backbone_freeze_finetun(self):
        self.freeze_or_unfreeze(backbone=True, finetun=False)

    def unfreeze_backbone_unfreeze_finetun(self):
        self.freeze_or_unfreeze(backbone=True, finetun=True)

    def forward(self,
                domain_name=None,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_emb=False,
                return_dict=True):

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        if return_emb:
            return sequence_output[:, 0]
        # print(outputs)
        # reshape = sequence_output.squeeze(0)

        # Apply LoRA
        if domain_name is not None:
            for name in self.domain_names:
                if name != domain_name:
                    for param in self.variant_apdater[name].parameters():
                        param.requires_grad = False
                else:
                    for param in self.variant_apdater[name].parameters():
                        param.requires_grad = True
            lora_output = self.variant_apdater[domain_name](sequence_output)
        else:
            lora_output = self.invariant_apdater(sequence_output)

        # context_token = lora_output[:, 0]

        # if domain_name is not None:
        #     logits = self.classifier[domain_name](context_token)
        # else:
        #     logits = self.classifier_share(context_token)
        pooled_output = self.pooler(lora_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # print(logits.view(-1, self.num_labels))
            # print(labels.view(-1))
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
