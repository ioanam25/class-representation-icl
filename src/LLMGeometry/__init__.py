import LLMGeometry.models
import LLMGeometry.models.llama3_8b_base
import LLMGeometry.models.llama3_1b_base
import LLMGeometry.models.llama3_70b_instruct
import LLMGeometry.models.gemma2_2b_base
import LLMGeometry.models.mistral_7b_base
import LLMGeometry.models.qwen2_7b_base

def load_model_and_tokenizer(model_name):
    if model_name == 'llama3.1_base':
        return LLMGeometry.models.llama3_8b_base.load_model_and_tokenizer()
    if model_name == 'llama3.1_1b_base':
        return LLMGeometry.models.llama3_1b_base.load_model_and_tokenizer()
    if model_name == 'llama3.1_70b_instruct':
        return LLMGeometry.models.llama3_70b_instruct.load_model_and_tokenizer()
    if model_name == 'gemma2_2b_base':
        return LLMGeometry.models.gemma2_2b_base.load_model_and_tokenizer()
    if model_name == 'mistral_7b_base':
        return LLMGeometry.models.mistral_7b_base.load_model_and_tokenizer()
    if model_name == 'qwen2_7b_base':
        return LLMGeometry.models.qwen2_7b_base.load_model_and_tokenizer()
    raise ValueError(f"Model {model_name} not supported")


def load_tokenizer(model_name):
    if model_name == 'llama3.1_base':
        return LLMGeometry.models.llama3_8b_base.load_tokenizer()
    if model_name == 'llama3.1_1b_base':
        return LLMGeometry.models.llama3_1b_base.load_tokenizer()
    if model_name == 'llama3.1_70b_instruct':
        return LLMGeometry.models.llama3_70b_instruct.load_tokenizer()
    if model_name == 'gemma2_2b_base':
        return LLMGeometry.models.gemma2_2b_base.load_tokenizer()
    if model_name == 'mistral_7b_base':
        return LLMGeometry.models.mistral_7b_base.load_tokenizer()
    if model_name == 'qwen2_7b_base':
        return LLMGeometry.models.qwen2_7b_base.load_tokenizer()
    
    raise ValueError(f"Model {model_name} not supported")

def get_number_of_hidden_layers(model_or_name):
    if isinstance(model_or_name, str):
        model_name = model_or_name
        if model_name == 'meta-llama/Meta-Llama-3.1-8B' or model_name == 'llama3.1_base':
            return 32
        if model_name == 'meta-llama/Llama-3.2-1B' or model_name == 'llama3.1_1b_base':
            return 16
        if model_name == 'meta-llama/Meta-Llama-3.1-70B-Instruct' or model_name == 'llama3.1_70b_instruct':
            return 80
        if model_name == 'google/gemma-2-2b' or model_name == 'gemma2_2b_base':
            return 26
        if model_name == 'mistralai/Mistral-7B-v0.3' or model_name == 'mistral_7b_base':
            return 32
        if model_name == 'Qwen/Qwen2.5-7B' or model_name == 'qwen2_7b_base':
            return 28
        raise ValueError(f"Model {model_name} not supported")
    else:
        model = model_or_name
        model_name = model.name_or_path
        if model_name == 'meta-llama/Meta-Llama-3.1-8B' or model_name == 'llama3.1_base':
            return len(model.model.layers)
        if model_name == 'meta-llama/Llama-3.2-1B' or model_name == 'llama3.1_1b_base':
            return len(model.model.layers)
        if model_name == 'meta-llama/Meta-Llama-3.1-70B-Instruct' or model_name == 'llama3.1_70b_instruct':
            return len(model.model.layers)
        if model_name == 'google/gemma-2-2b' or model_name == 'gemma2_2b_base':
            return len(model.model.layers)
        if model_name == 'mistralai/Mistral-7B-v0.3' or model_name == 'mistral_7b_base':
            return len(model.model.layers)
        if model_name == 'Qwen/Qwen2.5-7B' or model_name == 'qwen2_7b_base':
            return len(model.model.layers)
    raise ValueError(f"Model {model_name} not supported")



def get_decoder_block(model, index):
    model_name = model.name_or_path
    if model_name == 'meta-llama/Meta-Llama-3.1-8B' or model_name == 'llama3.1_base':
        return model.model.layers[index]
    if model_name == 'meta-llama/Llama-3.2-1B' or model_name == 'llama3.1_1b_base':
        return model.model.layers[index]
    if model_name == 'meta-llama/Meta-Llama-3.1-70B-Instruct' or model_name == 'llama3.1_70b_instruct':
        return model.model.layers[index]
    if model_name == 'google/gemma-2-2b' or model_name == 'gemma2_2b_base':
        return model.model.layers[index]
    if model_name == 'mistralai/Mistral-7B-v0.3' or model_name == 'mistral_7b_base':
        return model.model.layers[index]
    if model_name == 'Qwen/Qwen2.5-7B' or model_name == 'qwen2_7b_base':
        return model.model.layers[index]
    raise ValueError(f"Model {model_name} not supported")


def get_embedding_dim(model_or_name):
    if isinstance(model_or_name, str):
        model_name = model_or_name
        if model_name == 'meta-llama/Meta-Llama-3.1-8B' or model_name == 'llama3.1_base':
            return 4096
        if model_name == 'meta-llama/Llama-3.2-1B' or model_name == 'llama3.1_1b_base':
            return 2048
        if model_name == 'meta-llama/Meta-Llama-3.1-70B-Instruct' or model_name == 'llama3.1_70b_instruct':
            return 8192
        if model_name == 'google/gemma-2-2b' or model_name == 'gemma2_2b_base':
            return 2304
        if model_name == 'mistralai/Mistral-7B-v0.3' or model_name == 'mistral_7b_base':
            return 4096
        if model_name == 'Qwen/Qwen2.5-7B' or model_name == 'qwen2_7b_base':
            return 3584
    else:
        model = model_or_name
        model_name = model.name_or_path

        if model_name == 'meta-llama/Meta-Llama-3.1-8B' or model_name == 'llama3.1_base':
            return model.config.hidden_size
        if model_name == 'meta-llama/Llama-3.2-1B' or model_name == 'llama3.1_1b_base':
            return model.config.hidden_size
        if model_name == 'meta-llama/Meta-Llama-3.1-70B-Instruct' or model_name == 'llama3.1_70b_instruct':
            return model.config.hidden_size
        if model_name == 'google/gemma-2-2b' or model_name == 'gemma2_2b_base':
            return model.config.hidden_size
        if model_name == 'mistralai/Mistral-7B-v0.3' or model_name == 'mistral_7b_base':
            return model.config.hidden_size
        if model_name == 'Qwen/Qwen2.5-7B' or model_name == 'qwen2_7b_base':
            return model.config.hidden_size
    raise ValueError(f"Model {model_name} not supported")
