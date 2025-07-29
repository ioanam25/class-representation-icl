import pickle

# Modify the paths to the datasets as per your directory structure
def load_dataset_by_name(dataset_name, **kwargs):
    '''
        Load dataset by name

        Parameters:
        -----------
            dataset_name: str, one of ['TREC_coarse', 'ag_news', 'claude_multitask']
            **kwargs: additional arguments for dataset loading function
    '''
    if dataset_name == 'TREC_coarse':
        return pickle.load(open('datasets/TREC_coarse/TREC_coarse.pickle', 'rb'))
    if dataset_name == 'ag_news':
        return pickle.load(open('datasets/ag_news/ag_news.pickle', 'rb'))
    if dataset_name == 'claude_multitask':
        return pickle.load(open('datasets/claude_multitask/claude_multitask.pickle', 'rb'))
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented")