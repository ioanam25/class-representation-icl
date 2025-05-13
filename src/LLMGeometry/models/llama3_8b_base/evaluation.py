import torch
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from LLMGeometry.utils import create_causal_mask, take_last_from_attention_mask, expand_KV

import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from termcolor import colored
from typing import List, Dict, Any

# ----------------------- Code for manual forward pass (legacy) ----------------------- 

# def manual_attention(attention_module, hidden_states,attention_mask, position_ids=None):
#     '''
#         Manual computation of the attention module. Used to extract intermediate states.
#     '''
#     if position_ids is None:
#         position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).expand(hidden_states.shape[0],-1)
        
#     if attention_mask.ndim==2:
#         attention_mask_expanded = create_causal_mask(attention_mask).to(hidden_states.device)
#     else:
#         attention_mask_expanded = attention_mask
#     bsz, q_len, _ = hidden_states.size()

#     query_states = attention_module.q_proj(hidden_states)
#     key_states = attention_module.k_proj(hidden_states)
#     value_states = attention_module.v_proj(hidden_states)

#     query_states = query_states.view(bsz, q_len, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)
#     key_states = key_states.view(bsz, q_len, attention_module.num_key_value_heads, attention_module.head_dim).transpose(1, 2)
#     value_states = value_states.view(bsz, q_len, attention_module.num_key_value_heads, attention_module.head_dim).transpose(1, 2)

#     cos, sin = attention_module.rotary_emb(value_states, position_ids)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#     key_states = repeat_kv(key_states, attention_module.num_key_value_groups)
#     value_states = repeat_kv(value_states, attention_module.num_key_value_groups)

#     attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attention_module.head_dim) ) + attention_mask_expanded[:, :, :, : key_states.shape[-2]]

#     # upcast attention to fp32
#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#     attn_weights = nn.functional.dropout(attn_weights, p=attention_module.attention_dropout, training=attention_module.training)
#     attn_output = torch.matmul(attn_weights, value_states)

#     attn_output = attn_output.transpose(1, 2).contiguous()

#     attn_displacements = attn_output
#     attn_output = attn_output.reshape(bsz, q_len, -1)
#     attn_output = attention_module.o_proj(attn_output)

#     return {
#         'output': attn_output,
#         'attention_weights': attn_weights,
#         'key_states': key_states,
#         'value_states': value_states,
#         'query_states': query_states,
#         'attn_displacements': attn_displacements
#     }



# def manual_decoder_layer(hidden_states, position_ids, attention_mask, layer, 
#                          return_MLP_intermediate=False, return_attention_weights=True, return_QKV=True):
#     '''
#         Manual computation of a single LlamaDecoder layer.

#         Parameters:
#         ----------
#         hidden_states: torch.Tensor
#             The input embeddings to the layer.
#         position_ids: torch.Tensor
#             The position IDs of the input embeddings.
#         attention_mask: torch.Tensor
#             The attention mask for the layer.
#         layer: LlamaDecoderLayer
#             The LlamaDecoder layer to run the forward pass on.
#         return_MLP_intermediate: bool, optional
#             If True, the intermediate MLP output is returned. Tensor of shape (batch_size, sequence_length, MLP_dim).
#         return_attention_weights: bool, optional
#             If True, the attention weights are returned. Tensor of shape (batch_size, num_heads, sequence_length, sequence_length).
#         return_QKV: bool, optional
#             If True, the query, key, and value states are returned. Tensors of shape (batch_size, sequence_length, num_heads, head_dim).
#     '''
#     residual = hidden_states

#     hidden_states = layer.input_layernorm(hidden_states)

#     # Self Attention
#     attn_output = manual_attention(layer.self_attn, hidden_states, attention_mask, position_ids)
#     hidden_states = residual + attn_output['output']

#     # Fully Connected
#     residual = hidden_states
#     hidden_states = layer.post_attention_layernorm(hidden_states)
#     # --- MLP
#     intermediate = layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states)
#     hidden_states = layer.mlp.down_proj(intermediate)

#     hidden_states = residual + hidden_states

#     return {
#         'hidden_states' : hidden_states,
#         'attention_weights' : attn_output['attention_weights'] if return_attention_weights else None,
#         'key_states' : attn_output['key_states'].transpose(1,2) if return_QKV else None,
#         'value_states' : attn_output['value_states'].transpose(1,2) if return_QKV else None,
#         'query_states' : attn_output['query_states'].transpose(1,2) if return_QKV else None,
#         'MLP_intermediate' : intermediate if return_MLP_intermediate else None
#     }


# def manual_forward_pass(concatenated_input_embeddings, model, attention_mask, correct_labels=None, return_MLP_intermediate=False, return_attention_weights=False, return_QKV=False):
#     '''
#         Forward pass is rewritten as manual sequential computation across layers for flexibility of embedding extractions.

#         Parameters:
#             - concatenated_input_embeddings – Tensor with input embeddings (batch_size, sequence_length, embedding_dim)
#             - model – Callable() model
#             – attention_mask – Tensor (batch_size, sequence_length) of 0/1, specifying which tokens are masked for attention calculation.
#             – correct_labels – Tensor (batch_size, sequence_length) ID's of correct tokes to compute loss. Tokens that are not contributing to the loss are equal to -100.
#     '''
    
#     # --- Model forward pass
#     loss = None
#     attention_mask = create_causal_mask(attention_mask).to(model.device)
#     position_ids = torch.arange(concatenated_input_embeddings.shape[1], device=model.device).expand(concatenated_input_embeddings.shape[0],-1)


#     states = concatenated_input_embeddings
#     all_outputs = tuple()

#     for layer in model.model.layers:
#         # --- Iterating over LllamaDecoder layers
#         layer_output = manual_decoder_layer(
#             states, 
#             position_ids=position_ids,
#             attention_mask=attention_mask,
#             layer=layer,
#             return_MLP_intermediate=return_MLP_intermediate,
#             return_attention_weights=return_attention_weights,
#             return_QKV=return_QKV
#         )
#         all_outputs+=(layer_output,)
#         states = layer_output['hidden_states']

#     # --- Computing logits
#     states = model.model.norm(states)
#     logits = model.lm_head(states)

#     if correct_labels is not None:
#         loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         loss=loss_fn(logits.permute(0,2,1), correct_labels)


#     return {
#         'logits' : logits,
#         'outputs' : all_outputs,
#         'hidden_states' : [i['hidden_states'] for i in all_outputs],
#         'loss': loss,
#     }


class ActivationsHookLlama3:
    def __init__(self, model, add_positional_encodings=False):
        self.model = model
        self.hooks = []
        self.activations: List[Dict[str, Any]] = []
        self.num_key_value_heads = model.config.num_key_value_heads
        self.num_key_value_groups = model.model.layers[0].self_attn.num_key_value_groups
        self.num_heads = model.model.layers[0].self_attn.num_heads
        self.head_dim = model.model.layers[0].self_attn.head_dim
        self.add_positional_encodings = add_positional_encodings
        self._register_hooks()

    def clear_activations(self):
        self.activations = [{} for _ in range(len(self.model.model.layers))]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _register_hooks(self):

        def hidden_states_hook(module, input, output, layer_idx):
            # Extract hidden states (should be attached to the layer module)
            self.activations[layer_idx]['hidden_states'] = output[0]

        def MLP_intermediate_hook(act_fn_module, input, output, layer_idx):
            # Extract SwiGLU activations (should be attached to the layer.mlp.act_fn module)
            self.activations[layer_idx]['MLP_intermediate'] = output

        def keys_hook(k_proj_module, input, output, layer_idx):
            # Extract Key vectors (should be attached to the layer.self_attn.k_proj module)
            key_states = output
            bsz = key_states.shape[0]
            q_len = key_states.shape[1]
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if self.add_positional_encodings:
                raise NotImplementedError("Positional encodings are not yet supported")
            self.activations[layer_idx]['keys'] = repeat_kv(key_states, self.num_key_value_groups).transpose(1, 2)

        def values_hook(v_proj_module, input, output, layer_idx):
            # Extract Value vectors (should be attached to the layer.self_attn.v_proj module)
            value_states = output
            bsz = value_states.shape[0]
            q_len = value_states.shape[1]
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if self.add_positional_encodings:
                raise NotImplementedError("Positional encodings are not yet supported")
            self.activations[layer_idx]['values'] = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2) 

        def queries_hook(q_proj_module, input, output, layer_idx):
            # Extract Query vectors (should be attached to the layer.self_attn.q_proj module)
            query_states = output
            bsz = query_states.shape[0]
            q_len = query_states.shape[1]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim) # (bsz, q_len, num_heads, head_dim)
            self.activations[layer_idx]['queries'] = query_states 


        for idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(
                layer.register_forward_hook(lambda module, input, output, layer_idx=idx: hidden_states_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.mlp.act_fn.register_forward_hook(lambda module, input, output, layer_idx=idx: MLP_intermediate_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.k_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: keys_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.v_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: values_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.q_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: queries_hook(module, input, output, layer_idx)))


    def __call__(self, *args, **kwargs):
        self.clear_activations()
        output = self.model(*args, **kwargs)
        return output, self.activations


def hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=None, add_positional_encodings=False, **kwargs):
    '''
        Forward pass of the model with hooks to extract intermediate states.

        Parameters:
        ----------
            model - LlamaForCausalLM model
            concatenated_input_embeddings - Tensor with input embeddings (batch_size, sequence_length, embedding_dim)
            attention_mask - Tensor (batch_size, sequence_length) of 0/1, specifying which tokens are masked for attention calculation.
            correct_labels - Tensor (batch_size, sequence_length) ID's of correct tokes to compute loss. Tokens that are not contributing to the loss are equal to -100.
            add_positional_encodings - bool, optional - if True, positional encodings are added to the extracted K and V states.
    '''
    
        
    # --- Model forward pass
    loss = None
    hook = ActivationsHookLlama3(model, add_positional_encodings=add_positional_encodings)
    output, activations = hook(inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs)
    hook.remove_hooks()

    if correct_labels is not None:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss=loss_fn(output.logits.permute(0,2,1), correct_labels)

    return {
        'output': output,
        'activations': activations,
        'loss' : loss,
    }

