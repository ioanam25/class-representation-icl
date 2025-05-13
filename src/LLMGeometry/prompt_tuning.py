import torch
from LLMGeometry.evaluation import hooked_forward_pass
from LLMGeometry.preprocessing import prepare_batch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np

import torch
import pandas as pd
from termcolor import colored


TRAINING_CONFIG = dict(
    NUM_EPOCHS = 30,
    BATCH_SIZE = 16,
    INITAL_LR = 3e-4, # Initial learning rate
    LR_DECAY = 0.9, # Exponential decay factor for the learning rate (every DECREASE_LR_EVERY_N_EPOCHS epochs).
    ACC_GRADIENT_EVERY_N_BATCHES = 1, # Accumulate gradients every ACC_GRADIENT_EVERY_N_BATCHES batches.
    LOG_TRAIN_EVERY_N_BATCHES = 1, # Log training statistics and save soft prompt every LOG_TRAIN_EVERY_N_BATCHES batches.
    DECREASE_LR_EVERY_N_EPOCHS = 1, # Decrease the learning rate every DECREASE_LR_EVERY_N_EPOCHS epochs.
)


def train_soft_prompt(dataloader_train, model, tokenizer, prompt_text_field='text', answer_field='category', soft_embeds=None, SOFT_PROMPT_LENGTH=None):
    '''
        Trains the soft prompt for a given model and dataloader.

        Parameters:
        ----------
        - dataloader_train : DataLoader object
            DataLoader object containing the training data.
        - model : LlamaForCausalLM object to be passed into manual_forward_pass.
        - tokenizer : PreTrainedTokenizer object.
        - SOFT_PROMPT_LENGTH : int
            Length of the soft prompt.
        - prompt_text_field : str
            Name of the field in the dataset that contains the input text. Passed into prepare_batch as the prompt_text_field parameter.
        - answer_field : str
            Name of the field in the dataset that contains the target token. Passed into prepare_batch as the answer_field parameter.

        Returns:
        -------
        - pd.DataFrame containing the training statistics and the soft prompt tensor at each logging step.
    '''

    # --- Initializing the soft prompt. This tensor will be optimized during the training loop.
    if soft_embeds is None:
        soft_embeds = initialize_soft_prompt(SOFT_PROMPT_LENGTH, model, tokenizer, soft_prompt_initial_text='Task', random_init=False)
        print(colored("Initialized the soft prompt with initial text 'Task'.", 'green'))
    else:
        if soft_embeds.shape[1] != SOFT_PROMPT_LENGTH:
            print(colored(f"SOFT_PROMPT_LENGTH is {SOFT_PROMPT_LENGTH}, but the provided soft_embeds tensor has length {soft_embeds.shape[1]}.", 'red'))

    # ------------ Prompt training loop ------------
    optimizer = torch.optim.Adam([soft_embeds], lr=TRAINING_CONFIG['INITAL_LR'])
    scheduler = ExponentialLR(optimizer, TRAINING_CONFIG['LR_DECAY'])
    model.model.requires_grad_(False) # Since we're not optimizing the model params, this is to save memory

    batch_counter=-1
    acc_train = []

    for epoch in tqdm(range(TRAINING_CONFIG['NUM_EPOCHS'])):
        for batch in dataloader_train:
            batch_counter+=1
            
            # --- Training loop
            tokenized_batch = prepare_batch(batch, model, tokenizer, soft_prompt_embeds=soft_embeds, soft_prompt_location='before', prompt_text_field=prompt_text_field, answer_field=answer_field)
            attention_mask = tokenized_batch['attention_mask']
            correct_labels = tokenized_batch['correct_labels']

            output = hooked_forward_pass(model,tokenized_batch['input_embeds'],
                                                    attention_mask=attention_mask,
                                                    correct_labels=correct_labels,
                                                )
            loss = output['loss']
                
            # --- Logging
            if (batch_counter % TRAINING_CONFIG['LOG_TRAIN_EVERY_N_BATCHES'])==0:
                acc_train.append({
                    'epoch' : epoch,
                    'train_batch' :batch_counter-1,
                    'train_loss_total' : loss.item(),
                    'soft_embeds_tensor' : soft_embeds.detach().clone().cpu()
                })
                
            # --- Optimizing
            loss.backward()
            if (batch_counter % TRAINING_CONFIG['ACC_GRADIENT_EVERY_N_BATCHES'])==0:
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"batch {batch_counter} | loss: {loss.item()}")
        # --- Decreasing the learning rate
        if epoch % TRAINING_CONFIG['DECREASE_LR_EVERY_N_EPOCHS'] == 0:
            scheduler.step()
    return pd.DataFrame(acc_train)


def initialize_soft_prompt(soft_prompt_length,
                    model, tokenizer,
                    soft_prompt_initial_text = 'Please',
                    random_init=False
                   ):
    '''
        Initializes the soft prompt embeddings.

        Parameters:
        ------------
            soft_prompt_length (int) – the length of the soft prompt in tokens.
            model (LlamaForCausalLM) – the model to be used for the prompt tuning.
            tokenizer (PreTrainedTokenizer) – the tokenizer to be used for the prompt tuning.
            soft_prompt_initial_text (str) – the initial text of the soft prompt (if random_init=False). If the text is longer than 1 token, the embeddings will be averaged and expanded to the soft_prompt_length.
            random_init (bool) – whether to initialize the soft prompt randomly.

        Returns:
        --------
            soft_embeds (Tensor) – the soft prompt embeddings of shape (1, soft_prompt_length, embedding_dim).
    '''
    if random_init:
        # Random initialization
        soft_embeds = torch.randn((1,soft_prompt_length,model.config.hidden_size)).to(model.device)
        soft_embeds.requires_grad = True
        return soft_embeds
    
    # Initialization with a text
    soft_prompt = tokenizer.encode(soft_prompt_initial_text, add_special_tokens=False, return_tensors='pt')
    with torch.no_grad():
        soft_embeds = model.model.embed_tokens(
            soft_prompt.to(int).to(model.device)
        )
        soft_embeds = torch.mean(soft_embeds,dim=1).expand(1,soft_prompt_length,-1).clone()
    soft_embeds.requires_grad = True
    return soft_embeds
