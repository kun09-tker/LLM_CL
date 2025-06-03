import torch.nn as nn

from Layers.LoRA import LoRAApdater
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import DebertaV2Model

class MyDebertaV2Model(DebertaV2Model):
    def __init__(self, config, domain_names, rank=8, alpha=16):
        super().__init__(config)
        self.domain_names = domain_names
        self.invariant_apdater = LoRAApdater(
                "LoRA_share",
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                rank=rank, alpha=alpha
            )
        self.variant_apdater = nn.ModuleDict({
                name: LoRAApdater(
                        f"LoRA_{name}",
                        in_features=config.hidden_size,
                        out_features=config.hidden_size,
                        rank=rank, alpha=alpha
                    )
                for name in domain_names
            })
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Đóng băng tất cả các tham số
        for param in self.parameters():
            param.requires_grad = False

        # Mở khóa các tham số của LoRA
        for param in self.invariant_apdater.parameters():
            param.requires_grad = True
        for name in domain_names:
          for param in self.variant_apdater[name].parameters():
              param.requires_grad = True

        self.post_init()


    def forward(
        self,
        domain_name=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Default to config if return_dict is not specified
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=sequence_output if output_hidden_states else None,
                attentions=encoder_outputs.attentions if output_attentions else None
            )
        else:
            output = (logits,) + sequence_output[1:]
            return ((loss,) + output) if loss is not None else output