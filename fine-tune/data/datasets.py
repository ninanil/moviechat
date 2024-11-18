# data/datasets.py
import os
from datasets import load_dataset
from transformers import PreTrainedTokenizer

def load_datasets(artifact_dir, tokenizer: PreTrainedTokenizer):
    # Load the dataset
    dataset = load_dataset('json', data_dir=artifact_dir)

    # Filter splits
    train_dataset = dataset['train'].filter(lambda x: x['split'] == 'train')
    val_dataset = dataset['train'].filter(lambda x: x['split'] == 'val')

    # Map formatting function
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer, tokenizer.eos_token),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer, tokenizer.eos_token),
        batched=True
    )

    # Remove unnecessary columns
    columns_to_remove = ['split', 'type', 'instruction', 'input', 'context', 'response']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    return train_dataset, val_dataset

def formatting_prompts_func(examples, tokenizer: PreTrainedTokenizer, eos_token: str):
    # Implement your formatting logic here
    # ...
    return {"text": formatted_texts}
