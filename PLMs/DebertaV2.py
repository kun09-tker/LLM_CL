import torch
import torch.nn as nn

from Layers.LoRA import LoRAApdater
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from transformers import DebertaV2Model, DebertaV2ForSequenceClassification

class MyDebertaV2Model(DebertaV2Model):
    def __init__(self, deberta_model, domain_names, rank=8, alpha=16):
        super().__init__(deberta_model.config)
        self.config = deberta_model.config
        self.embeddings = deberta_model.embeddings
        self.encoder = deberta_model.encoder
        self.domain_names = domain_names
        self.invariant_apdater = LoRAApdater("LoRA_share", in_features=self.config.hidden_size, out_features=self.config.hidden_size, rank=rank, alpha=alpha)
        self.variant_apdater = nn.ModuleDict({
            name: LoRAApdater(f"LoRA_{name}", in_features=self.config.hidden_size, out_features=self.config.hidden_size, rank=rank, alpha=alpha)
                for name in domain_names})
        self.post_init()

    def forward(
        self,
        domain_name=None,
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

        # Apply LoRA
        if domain_name is not None:
            lora_output = self.variant_apdater[domain_name](embeddings_output)
        else:
            lora_output = self.invariant_apdater(embeddings_output)

        # Pass through encoder
        encoder_outputs = self.encoder(
            lora_output,
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
    def __init__(self, config, domain_names, rank=8, alpha=16):
        super().__init__(config)
        self.deberta = MyDebertaV2Model(self.deberta, domain_names, rank, alpha)
        self.post_init()

        # Đóng băng tất cả các tham số
        for param in self.parameters():
            param.requires_grad = False

        # Mở khóa các tham số của LoRA
        for param in self.deberta.invariant_apdater.parameters():
            param.requires_grad = True
        for name in domain_names:
          for param in self.deberta.variant_apdater[name].parameters():
              param.requires_grad = True

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
                return_dict=True):

        outputs = self.deberta(
            domain_name=domain_name,
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
        # sequence_output = self.dropout(sequence_output)
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
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
