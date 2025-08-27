import json
from pathlib import Path

def generate_configs():
    base_config = {
        'MODEL_NAME': 'llama3.1_base',
        'DATASET_NAME': 'claude_multitask',
        'num_classes': 3,
        'prefix_type': 'demos',
        'keyword': 'Category',
        'answer_field': 'emotion_letter',
        'N_RUNS': 10,
        'root_folder': "learning_curves_relabel_demos",
        'ensemble_assignment': False,
        'ensemble_method': 'logit_averaging',
        'ensemble_temperature': 0,
        'top_tokens': 128256,
        'whole_words_only': True,
        'base_seed': 42
    }
    
    configs = []
    
    # For each n_relabel (10, 20, ..., 100)
    for n_relabel in range(10, 101, 10):
        # For each n_examples (0 to 100)
        for n_examples in range(0, 101):
            config = base_config.copy()
            config['n_relabel'] = n_relabel
            config['n_examples'] = n_examples
            configs.append(config)
    
    # Save configurations
    output_dir = Path('experiments')
    # output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'icl_configs.json'
    with open(output_file, 'w') as f:
        json.dump(configs, f, indent=2)
    
    print(f"Generated {len(configs)} configurations")
    print(f"Saved to {output_file}")
    print(f"\nTotal jobs: {len(configs)}")
    print(f"n_relabel values: {list(range(10, 101, 10))}")
    print(f"n_examples values: {list(range(0, 101))}")
    print("\nExample config:")
    print(json.dumps(configs[0], indent=2))

if __name__ == "__main__":
    generate_configs() 