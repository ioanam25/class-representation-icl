#!/usr/bin/env python3
"""
Generate fake relabeling pickle files using hand-picked dictionary synonyms
instead of optimized tokens. These can be used with the existing ICL pipeline
via fixed_relabeling_path.

Usage:
    python experiments/generate_synonym_relabelings.py
"""

import os
import sys
import pickle
import transformers

# ── Qwen tokenizer setup ────────────────────────────────────────────────────
def load_qwen_tokenizer():
    safe_name = 'Qwen/Qwen2.5-7B'.replace('/', '--')
    refs_main = os.path.join('tokenizers', f'models--{safe_name}', 'refs', 'main')
    with open(refs_main) as f:
        commit_hash = f.read().strip()
    snap = os.path.join('tokenizers', f'models--{safe_name}', 'snapshots', commit_hash)
    return transformers.AutoTokenizer.from_pretrained(snap, local_files_only=True)

# ── Synonym definitions ─────────────────────────────────────────────────────
# 3-class: Joy (A), Anger (C), Fear (D)
SYNONYMS_3CLASS = {
    'gold':  {'A': 'joy',       'C': 'anger',      'D': 'fear'},
    'syn1':  {'A': 'happiness', 'C': 'rage',        'D': 'anxiety'},
    'syn2':  {'A': 'delight',   'C': 'fury',        'D': 'dread'},
    'syn3':  {'A': 'cheerful',  'C': 'wrath',       'D': 'panic'},
    'syn4':  {'A': 'pleased',   'C': 'irritation',  'D': 'terror'},
}

# 5-class: Joy (A), Sadness (B), Anger (C), Fear (D), Surprise (E)
SYNONYMS_5CLASS = {
    'gold':  {'A': 'joy',       'B': 'sadness',  'C': 'anger', 'D': 'fear',  'E': 'surprise'},
    'syn1':  {'A': 'happiness', 'B': 'grief',    'C': 'rage',  'D': 'anxiety', 'E': 'startled'},
    'syn2':  {'A': 'delight',   'B': 'sorrow',   'C': 'fury',  'D': 'dread',   'E': 'awe'},
    'syn3':  {'A': 'cheerful',  'B': 'misery',   'C': 'wrath', 'D': 'panic',   'E': 'shock'},
}

def make_relabeling_pkl(tokenizer, synonym_dict, set_name, num_classes, output_dir):
    """Create a relabeling pickle file mimicking the format of optimize_tokens output."""
    new_labels = {}
    for cls_key, word in synonym_dict.items():
        token_str = 'Ġ' + word
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        # Validate it's a real single token
        encoded = tokenizer.encode(' ' + word, add_special_tokens=False)
        if len(encoded) != 1 or token_id is None:
            raise ValueError(
                f"'{word}' is NOT a single token for Qwen! "
                f"token_str={token_str}, id={token_id}, encoded={encoded}. "
                f"Pick a different synonym."
            )
        new_labels[cls_key] = (token_str, token_id)

    # Build pickle in the same format as generate_relabelings produces
    data = {
        'config': {
            'MODEL_NAME': 'qwen2_7b_base',
            'DATASET_NAME': 'claude_multitask',
            'num_classes': num_classes,
            'top_tokens': 128256,
            'whole_words_only': True,
            'ensemble_assignment': False,
            'ensemble_method': 'voting',
            'synonym_set': set_name,  # extra metadata
        },
        'relabelings': [{
            'labels': new_labels,
            'objective': 0.0,  # no optimization was run
        }],
    }

    os.makedirs(output_dir, exist_ok=True)
    fname = f"qwen2_7b_base_relabelings_{num_classes}classes_synonym_{set_name}.pkl"
    path = os.path.join(output_dir, fname)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✓ {path}")
    return path


def main():
    print("Loading Qwen tokenizer...")
    tokenizer = load_qwen_tokenizer()

    output_dir = "qwen2_7b_base_relabelings_synonyms"

    print("\n=== 3-class synonym relabelings ===")
    for set_name, syn_dict in SYNONYMS_3CLASS.items():
        make_relabeling_pkl(tokenizer, syn_dict, set_name, num_classes=3, output_dir=output_dir)

    print("\n=== 5-class synonym relabelings ===")
    for set_name, syn_dict in SYNONYMS_5CLASS.items():
        make_relabeling_pkl(tokenizer, syn_dict, set_name, num_classes=5, output_dir=output_dir)

    print("\nDone! All synonym relabeling files saved to:", output_dir)


if __name__ == '__main__':
    main()
