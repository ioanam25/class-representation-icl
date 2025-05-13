import torch

def embed_tokens(input_tokens, model):
    '''
        Embeds the input tokens using the model's embedding layer.

        Parameters:
        ------------
            input_tokens (Tensor or list) – the input tokens to be embedded. Can be a Tensor of token IDs or a list of Tensors.
            model (Gemma2ForCausalLM) – the model to be used for the token embedding.

        Returns:
        --------
            embeds (Tensor or list) – the embedded tokens. If the input is a list, the output is a list of Tensors.
    '''

    with torch.no_grad():
        if isinstance(input_tokens, list):
            embeds = [model.model.embed_tokens(
                t.to(int).to(model.device)
            ) for t in input_tokens]
        else:
            embeds = model.model.embed_tokens(input_tokens.to(int).to(model.device))
    return embeds



def prepare_batch(batch, model, tokenizer, soft_prompt_embeds=None, soft_prompt_location='before', prompt_text_field='text', answer_field=None, verbose=False, add_special_tokens=True):
    '''
        Creates a batch of input–output tokens and attention masks.

        Parameters:
            - batch (Iterable) an iterable of dictionaries. Each dictionary should contain the prompt and the correct output token.
            - model (Gemma2ForCausalLM) – the model to be used for the prompt tuning.
            - tokenizer (PreTrainedTokenizer) – the tokenizer to be used for the prompt tuning.
            – soft_prompt_embeds (Tensor) – the soft prompt embeddings of shape (1, soft_sequence_length, embedding_dim). If None, no soft prompt is used.
            - soft_prompt_location (str) – the location of the soft prompt embeddings relative to the  Can be 'before' or 'after'. Ignored if soft_prompt_embeds is None.
            - prompt_text_field (str) – a name of the batch dictionary field that contains a string to be used as the prompt.
            - answer_field (str) – a name of the batch dictionary field that specifies desired response of the model. If None, corresponding correct label is set to -100.
    '''

    prompt_tokens = [tokenizer.encode(b[prompt_text_field], add_special_tokens=False, return_tensors='pt') for b in batch]
    if answer_field is not None:
        answer_tokens = [tokenizer.encode(b[answer_field], add_special_tokens=False, return_tensors='pt')[:,[0]] for b in batch] # only the first token is used as the answer
    else:
        answer_tokens = [-100]*len(batch)
        if verbose:
            print("Warning: No answer field is specified. Correct labels are set to -100")

    if soft_prompt_embeds is None:
        if verbose:
            print("Warning: No soft prompt is specified")
        soft_prompt_embeds = torch.zeros((1,0,model.config.hidden_size)).to(model.device)

    if soft_prompt_location=='after': # <bos> + PROMPT_TOKENS + SOFT_PROMPT + PAD
        if add_special_tokens:
            input_token_ids_unpadded = [
                torch.cat([
                    tokenizer.encode('<bos>', add_special_tokens=False, return_tensors='pt'),
                    prompt_tokens[k],
                    torch.ones((1,soft_prompt_embeds.shape[1]))*(-100),
                ], dim=1) for k in range(len(batch))
            ]

            input_embeds_unpadded = [
                torch.cat([
                    embed_tokens(tokenizer.encode('<bos>', add_special_tokens=False, return_tensors='pt'), model),
                    embed_tokens(prompt_tokens[k], model),
                    soft_prompt_embeds,
                ], dim=1) for k in range(len(batch))
            ]
        else:
            input_token_ids_unpadded = [
                torch.cat([
                    prompt_tokens[k],
                    torch.ones((1,soft_prompt_embeds.shape[1]))*(-100),
                ], dim=1) for k in range(len(batch))
            ]

            input_embeds_unpadded = [
                torch.cat([
                    embed_tokens(prompt_tokens[k], model),
                    soft_prompt_embeds,
                ], dim=1) for k in range(len(batch))
            ]

        max_length = max([i.shape[1] for i in input_embeds_unpadded])


        input_token_ids_padded = torch.cat([
            torch.cat([i, torch.ones((1,max_length-i.shape[1]))*(-100)], dim=1) for i in input_token_ids_unpadded
        ])

        input_embeds_padded = torch.cat([
            torch.cat([i, torch.zeros((1,max_length-i.shape[1],i.shape[2])).to(i.device)], dim=1) for i in input_embeds_unpadded
        ])
        attention_mask = torch.cat([torch.cat([torch.ones((1,i.shape[1])), torch.zeros((1,max_length-i.shape[1]))], dim=1) for i in input_embeds_unpadded])

    if soft_prompt_location=='before': # <bos> + SOFT_PROMPT + PROMPT_TOKENS + PAD
        if add_special_tokens:
            input_token_ids_unpadded = [
                torch.cat([
                    tokenizer.encode('<bos>', add_special_tokens=False, return_tensors='pt'),
                    torch.ones((1,soft_prompt_embeds.shape[1]))*(-100),
                    prompt_tokens[k],
                ], dim=1) for k in range(len(batch))
            ]

            input_embeds_unpadded = [
                torch.cat([
                    embed_tokens(tokenizer.encode('<bos>', add_special_tokens=False, return_tensors='pt'), model),
                    soft_prompt_embeds,
                    embed_tokens(prompt_tokens[k], model),
                ], dim=1) for k in range(len(batch))
            ]
        else:
            input_token_ids_unpadded = [
                torch.cat([
                    torch.ones((1,soft_prompt_embeds.shape[1]))*(-100),
                    prompt_tokens[k],
                ], dim=1) for k in range(len(batch))
            ]

            input_embeds_unpadded = [
                torch.cat([
                    soft_prompt_embeds,
                    embed_tokens(prompt_tokens[k], model),
                ], dim=1) for k in range(len(batch))
            ]
            
        max_length = max([i.shape[1] for i in input_embeds_unpadded])

        input_token_ids_padded = torch.cat([
            torch.cat([i, torch.ones((1,max_length-i.shape[1]))*(-100)], dim=1) for i in input_token_ids_unpadded
        ])
        input_embeds_padded = torch.cat([
            torch.cat([i, torch.zeros((1,max_length-i.shape[1],i.shape[2])).to(i.device)], dim=1) for i in input_embeds_unpadded
        ])
        attention_mask = torch.cat([torch.cat([torch.ones((1,i.shape[1])), torch.zeros((1,max_length-i.shape[1]))], dim=1) for i in input_embeds_unpadded])

    # --- Creating correct labels tensor
    correct_labels = torch.ones_like(attention_mask)*(-100)
    for k in range(len(batch)):
        idx_of_last = torch.argmin(attention_mask[k], dim=0).item()-1
        correct_labels[k][idx_of_last] = answer_tokens[k]

    return {
        "input_ids": input_token_ids_padded.to(torch.long).to(model.device),
        "input_embeds": input_embeds_padded.to(torch.bfloat16).to(model.device),
        "attention_mask": attention_mask.to(model.device),
        "correct_labels" : correct_labels.to(torch.long).to(model.device)
    }