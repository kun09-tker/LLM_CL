
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertAttention, BertSelfAttention, BertSelfOutput, BertOutput
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from Adapters.Bert import BertAdapterMask

class MyBertOutput(BertOutput):
    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(config)
        self.adapter_mask = BertAdapterMask(ntasks, bert_hidden_size, bert_adapter_size)

    def forward(self, hidden_states, input_tensor,**kwargs):

        # add parameters --------------
        s, t = None, None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------


        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.adapter_mask(hidden_states,t,s)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MyBertSelfOutput(BertSelfOutput):
    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(config)
        self.adapter_mask = BertAdapterMask(ntasks, bert_hidden_size, bert_adapter_size)

    def forward(self, hidden_states, input_tensor,**kwargs):

        # add parameters --------------
        s,t=None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.adapter_mask(hidden_states,t,s)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MyBertAttention(BertAttention):
    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(config)
        self.output = MyBertSelfOutput(config, ntasks, bert_hidden_size, bert_adapter_size)
        self.self = BertSelfAttention(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,**kwargs):

        # add parameters --------------
        s,t=None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------


        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions
        )

        attention_output = self.output(self_outputs[0], hidden_states,t=t,s=s)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class MyBertLayer(BertLayer):
    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(config)
        self.attention = MyBertAttention(config, ntasks, bert_hidden_size, bert_adapter_size)
        self.crossattention = MyBertAttention(config, ntasks, bert_hidden_size, bert_adapter_size)
        self.output = MyBertOutput(config, ntasks, bert_hidden_size, bert_adapter_size)
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,**kwargs
    ):

        # add parameters --------------
        s,t=None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            t=t,s=s,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
            t=t,s=s,
        )

        outputs = (layer_output,) + outputs
        return outputs

class MyBertEncoder(BertEncoder):
    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000):
        super().__init__(config)
        self.layer = nn.ModuleList([MyBertLayer(config, ntasks, bert_hidden_size, bert_adapter_size) for _ in range(config.num_hidden_layers)])
    def compute_layer_outputs(self,
                              output_attentions,layer_module,
                              hidden_states,attention_mask,layer_head_mask,
                              encoder_hidden_states,encoder_attention_mask,**kwargs):

        # add parameters --------------

        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------

        layer_outputs = layer_module(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            t=t,s=s
        )

        return layer_outputs,x_list,h_list

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,**kwargs
    ):


        # add parameters --------------
        s,t=None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs,x_list,h_list = self.compute_layer_outputs(
                          output_attentions,layer_module,
                          hidden_states,attention_mask,layer_head_mask,
                          encoder_hidden_states,encoder_attention_mask,
                          t=t,s=s,x_list=x_list,h_list=h_list
                        )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions,cross_attentions=all_cross_attentions
        )

class MyBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, ntasks, bert_hidden_size = 768, bert_adapter_size = 2000, add_pooling_layer=True):
        super().__init__(config)
        self.encoder = MyBertEncoder(config, ntasks, bert_hidden_size, bert_adapter_size)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,**kwargs
    ):


        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """

        # add parameters --------------
        s,t,=None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        # other parameters --------------

        x_list = [] #accumulate for every forward pass
        h_list = []

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,  token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs,x_list,h_list = self.compute_encoder_outputs(
                                embedding_output,extended_attention_mask,head_mask,
                                encoder_hidden_states,encoder_extended_attention_mask,output_attentions,
                                output_hidden_states,return_dict,t=t,s=s,x_list=x_list,h_list=h_list)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def compute_encoder_outputs(self,
                                embedding_output,extended_attention_mask,head_mask,
                                encoder_hidden_states,encoder_extended_attention_mask,output_attentions,
                                output_hidden_states,return_dict,**kwargs):


        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,t=t,s=s
        )

        return encoder_outputs,x_list,h_list

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Lấy các tham số thêm
        ntasks = kwargs.pop("ntasks", 1)
        bert_hidden_size = kwargs.pop("bert_hidden_size", 768)
        bert_adapter_size = kwargs.pop("bert_adapter_size", 2000)

        # Lấy config
        config = kwargs.get("config", None)
        if config is None:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Khởi tạo mô hình
        model = cls(config, ntasks, bert_hidden_size, bert_adapter_size)

        # Load trọng số pretrained
        from transformers.modeling_utils import load_state_dict
        from transformers.file_utils import cached_path, WEIGHTS_NAME
        import os

        model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME) if os.path.isdir(pretrained_model_name_or_path) else pretrained_model_name_or_path
        state_dict = load_state_dict(model_file)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            print("⚠️ Missing keys:", missing_keys)
        if len(unexpected_keys) > 0:
            print("⚠️ Unexpected keys:", unexpected_keys)

        return model

from typing import Callable

def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors,**kwargs
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    # add parameters --------------
    s, t = None, None
    if 't' in kwargs: t = kwargs['t']
    if 's' in kwargs: s = kwargs['s']
    # other parameters --------------


    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    # num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    # assert num_args_in_forward_chunk_fn == len(
    #     input_tensors
    # ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
    #     num_args_in_forward_chunk_fn, len(input_tensors)
    # )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk,t=t,s=s) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors, t=t,s=s)

def feed_forward_chunk(self, attention_output, t=None,s=1,):
    intermediate_output = self.intermediate(attention_output)

    layer_output = self.output(intermediate_output, attention_output, t=t,s=s,)
    return layer_output