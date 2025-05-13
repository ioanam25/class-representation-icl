from LLMGeometry.utils import generate_random_samples
import json

class ICL_Template():
    def __init__(self, name, demonstrations_enabled, task_instruction_enabled, demonstrations_template=None, task_instruction=None, test_template="{}"):
        self.name = name
        self.demonstrations_enabled = demonstrations_enabled
        self.task_instruction_enabled = task_instruction_enabled
        self.demonstrations_template = demonstrations_template
        self.task_instruction = task_instruction
        self.test_template = test_template
        self.ICL_prompt = None
        self.train_idx = None
    
    def create_ICL_prompt(self, dataframe, prompt_text_field='text', answer_field='category', N_samples=4, shuffle=False, seed=None):
        '''
            Create ICL prompt from the train dataframe. The prompt will be saved in self.ICL_prompt attribute.

            Parameters:
            ----------
            dataframe: pd.DataFrame
                The dataframe containing the training data. The dataframe should have the following columns:
                - prompt_text_field: The field containing the text prompt
                - answer_field: The field containing the answer
            prompt_text_field: str
                The field containing the text prompt
            answer_field: str
                The field containing the answer
            N_samples: int
                Number of samples to choose from the dataframe as demonstrations
        '''
        if self.task_instruction_enabled:
            prompt = self.task_instruction + "\n"
        else:
            prompt = ""

        if self.demonstrations_enabled:
            chosen_indices = generate_random_samples(dataframe[answer_field], N_samples, seed=seed)
            chosen_sentences = dataframe.loc[chosen_indices,prompt_text_field]
            chosen_labels = dataframe.loc[chosen_indices, answer_field]
            self.train_idx = chosen_indices

            if shuffle:
                chosen_labels = chosen_labels.sample(frac=1).reset_index(drop=True) # Shuffle the labels

            for sentence, label in zip(chosen_sentences, chosen_labels):
                prompt += self.demonstrations_template.format(sentence, label) + "\n"
                
        self.ICL_prompt = prompt

        return prompt, self.train_idx

    def apply(self, x):
        '''
            Apply the ICL prompt to the test data.

            Parameters:
            ----------
            x: str
                The text prompt (test sentence)
        '''
        if self.ICL_prompt is None:
            raise ValueError("ICL prompt is not created. Run create_ICL_prompt method first.")
        
        return self.ICL_prompt + self.test_template.format(x)

    def apply_to_test(self, x):
        return self.test_template.format(x)
    
    def create_dataframes(self, train_dataframe, test_dataframe, prompt_text_field, answer_field, N_samples=4, add_space_to_answer='auto', tokenizer=None):
        if add_space_to_answer=='auto':
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided to determine whether to add space to the answer")
            if tokenizer.name_or_path in ['mistralai/Mistral-7B-v0.3']:
                add_space_to_answer = False # Mistral uses sentencepiece tokenizer, which does not require space before the answer 
            else:
                add_space_to_answer = True

        self.create_ICL_prompt(train_dataframe, prompt_text_field=prompt_text_field, answer_field=answer_field, N_samples=N_samples)
        test_dataframe = test_dataframe.copy()
        test_dataframe['ICL_prompt'] = test_dataframe[prompt_text_field].apply(lambda x: self.apply(x, prompt_text_field))
        if add_space_to_answer:
            test_dataframe[answer_field] = " " + test_dataframe[answer_field]
        return test_dataframe
    
    def __repr__(self):
        return f"ICL_Template(name={self.name}, demonstrations_template={self.demonstrations_template}, task_instruction={self.task_instruction})"
    
def load_ICL_template(name, json_path=None, dataset_name=None):
    if dataset_name is not None:
        json_path = f"/mnt/home/akirsanov/LLMGeometry/experiments/ICL/templates/{dataset_name}.json"
    else:
        if json_path is None:
            raise ValueError("Either json_path or dataset_name must be provided.")
        
    with open(json_path, 'r') as f:
        templates = json.load(f)
    for template in templates:
        if template['name'] == name:
            return ICL_Template(**template)
    raise ValueError(f'Template {name} not found in {json_path}')


def list_ICL_templates_in_json(json_path):
    with open(json_path, 'r') as f:
        templates = json.load(f)
    return [template['name'] for template in templates]