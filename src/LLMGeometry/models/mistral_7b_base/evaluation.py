import torch
from typing import List, Dict, Any

class ActivationsHookMisral7B:
    def __init__(self, model, add_positional_encodings=False):
        self.model = model
        self.hooks = []
        self.activations: List[Dict[str, Any]] = []
        self.num_attention_heads = model.config.num_attention_heads
        self.num_key_value_heads = model.config.num_key_value_heads
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
            self.activations[layer_idx]['hidden_states'] = output[0]

        def MLP_intermediate_hook(gate_proj_module, input, output, layer_idx):
            self.activations[layer_idx]['MLP_intermediate'] = output

        def queries_hook(q_proj_module, input, output, layer_idx):
            bsz, q_len, _ = output.shape
            queries = output.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            self.activations[layer_idx]['queries'] = queries

        def keys_hook(k_proj_module, input, output, layer_idx):
            bsz, q_len, _ = output.shape
            keys = output.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            self.activations[layer_idx]['keys'] = keys

        def values_hook(v_proj_module, input, output, layer_idx):
            bsz, q_len, _ = output.shape
            values = output.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            self.activations[layer_idx]['values'] = values

            if self.add_positional_encodings:
                raise NotImplementedError("Positional encodings are not yet supported")

        for idx, layer in enumerate(self.model.model.layers):
            self.hooks.append(
                layer.register_forward_hook(lambda module, input, output, layer_idx=idx: hidden_states_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.mlp.gate_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: MLP_intermediate_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.q_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: queries_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.k_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: keys_hook(module, input, output, layer_idx)))
            self.hooks.append(
                layer.self_attn.v_proj.register_forward_hook(lambda module, input, output, layer_idx=idx: values_hook(module, input, output, layer_idx)))

    def __call__(self, *args, **kwargs):
        self.clear_activations()
        output = self.model(*args, **kwargs)
        return output, self.activations

def hooked_forward_pass(model, input_embeds, attention_mask, correct_labels=None, add_positional_encodings=False, **kwargs):
    '''
    Forward pass of the Gemma 2 model with hooks to extract intermediate states.

    Parameters:
    ----------
        model - GemmaForCausalLM model
        input_embeds - Tensor with input embeddings (batch_size, sequence_length, embedding_dim)
        attention_mask - Tensor (batch_size, sequence_length) of 0/1, specifying which tokens are masked for attention calculation.
        correct_labels - Tensor (batch_size, sequence_length) ID's of correct tokens to compute loss. Tokens that are not contributing to the loss are equal to -100.
        add_positional_encodings - bool, optional - if True, positional encodings are added to the extracted K and V states.
    '''
    
    loss = None
    hook = ActivationsHookMisral7B(model, add_positional_encodings=add_positional_encodings)
    output, activations = hook(inputs_embeds=input_embeds, attention_mask=attention_mask, **kwargs)
    hook.remove_hooks()

    if correct_labels is not None:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(output.logits.permute(0, 2, 1), correct_labels)

    return {
        'output': output,
        'activations': activations,
        'loss': loss,
    }