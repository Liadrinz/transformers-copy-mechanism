import torch

from torch import nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any, List
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, ModelOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GenerationMixinWithCopyMech(GenerationMixin):
    
    def _get_src_hidden_states(self, outputs: ModelOutput):
        raise NotImplementedError
    
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "src_hidden_states", "src_ids_for_copy"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs
    
    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder)
        # feed `src_hidden_states` to next decoding step
        model_kwargs["src_hidden_states"] = self._get_src_hidden_states(outputs)
        return model_kwargs
    
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            input_ids, expand_size, is_encoder_decoder, attention_mask, encoder_outputs, **model_kwargs)
        # expand `src_ids_for_copy` when doing beam search
        if "src_ids_for_copy" in model_kwargs:
            src_ids_for_copy = model_kwargs["src_ids_for_copy"]
            model_kwargs["src_ids_for_copy"] = src_ids_for_copy.index_select(0, expanded_return_idx)
        return input_ids, model_kwargs
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs) -> dict:
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        inputs["src_ids_for_copy"] = kwargs.get("src_ids_for_copy", None)
        inputs["src_hidden_states"] = kwargs.get("src_hidden_states", None)
        return inputs


@dataclass
class CausalLMOutputWithCrossAttentionsAndCopyMechanism(CausalLMOutputWithCrossAttentions):
    
    src_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class GPT2LMHeadModelWithCopyMech(GPT2LMHeadModel, GenerationMixinWithCopyMech):
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.p_gen_head = nn.Sequential(
            nn.Linear(config.n_embd * 3, 1),
            nn.Sigmoid(),
        )
        self.post_init()
    
    def _get_src_hidden_states(self, outputs: ModelOutput):
        return outputs.src_hidden_states
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        src_ids_for_copy: Optional[torch.LongTensor] = None,
        src_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentionsAndCopyMechanism]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        
        if src_ids_for_copy is None:
            src_ids_for_copy = input_ids
        
        if labels is None and src_hidden_states is None:
            # Inference, encoding step
            logits = lm_logits
        else:
            embeds = transformer_outputs.hidden_states[0]
            attentions = transformer_outputs.attentions[-1].mean(dim=1)
            if labels is not None:
                # Training
                is_source = (labels == -100).long()
                is_target = 1 - is_source
                context_vectors = (attentions * is_source[:, None, :]) @ (hidden_states * is_target[:, :, None])
            else:
                # Inference, decoding step
                src_len = src_ids_for_copy.size(1)
                attentions = attentions[:, :, :src_len]
                context_vectors = attentions @ src_hidden_states
            total_states = torch.cat((context_vectors, hidden_states, embeds), dim=-1)
            p_gen = self.p_gen_head(total_states)
            p_copy = 1 - p_gen
            input_one_hot = src_ids_for_copy.new_zeros(src_ids_for_copy.size(0), src_ids_for_copy.size(1), self.config.vocab_size)
            input_one_hot.scatter_(-1, src_ids_for_copy[:, :, None], 1)
            input_one_hot = input_one_hot.float()
            cp_logits = attentions @ input_one_hot
            logits = p_gen * lm_logits + p_copy * cp_logits
        
        if src_hidden_states is None:
            src_hidden_states = hidden_states

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:] + (src_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentionsAndCopyMechanism(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            src_hidden_states=src_hidden_states
        )


class BartForConditionalGenerationWithCopyMech(BartForConditionalGeneration, GenerationMixinWithCopyMech):
    
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.p_gen_head = nn.Sequential(
            nn.Linear(config.d_model * 3, 1),
            nn.Sigmoid(),
        )
        self.post_init()
    
    def _get_src_hidden_states(self, outputs: ModelOutput):
        return outputs.encoder_last_hidden_state
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        src_ids_for_copy: Optional[torch.LongTensor] = None,
        src_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        
        if src_ids_for_copy is None:
            src_ids_for_copy = input_ids
        
        if labels is None and src_hidden_states is None:
            # Inference, encoding step
            logits = lm_logits
        else:
            embeds = outputs.decoder_hidden_states[0]
            attentions = outputs.cross_attentions[-1].sum(dim=1)
            if labels is not None:
                # Training
                context_vectors = attentions @ hidden_states
            else:
                # Inference, decoding step
                context_vectors = attentions @ src_hidden_states
            total_states = torch.cat((context_vectors, hidden_states, embeds), dim=-1)
            p_gen = self.p_gen_head(total_states)
            p_copy = 1 - p_gen
            input_one_hot = src_ids_for_copy.new_zeros(src_ids_for_copy.size(0), src_ids_for_copy.size(1), self.config.vocab_size)
            input_one_hot.scatter_(-1, src_ids_for_copy[:, :, None], 1)
            input_one_hot = input_one_hot.float()
            cp_logits = attentions @ input_one_hot
            logits = p_gen * lm_logits + p_copy * cp_logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
