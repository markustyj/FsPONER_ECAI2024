

'''
from typing import Dict
from typing import Tuple
from typing import Union
from functools import cmp_to_key
from datasets import Dataset
'''
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import Trainer
from datasets import load_dataset


train_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/train.jsonl")
val_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/val.jsonl")
test_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/test.jsonl")

#output_model='models/thin-film-finetuned',
finetuned_path = "/home/z004r5cc/Documents/master_thesis/salmon_ner/models/thin-film-finetuned"
tokenizer_path='/home/z004r5cc/Documents/master_thesis/salmon_ner/models/thin-film-finetuned'
tokenizer_add_prefix_space = True


raw_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
            "test": test_file,
        },
    )
"""
model_loaded = AutoModelForTokenClassification.from_pretrained(finetuned_path)

tokenizer = AutoTokenizer.from_pretrained( tokenizer_path )
tokenized_datasets = tokenizer(raw_datasets)
"""

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenized_datasets = tokenizer(raw_datasets)
#tokenized_datasets = tokenizer(raw_datasets["test"])

model_loaded = AutoModelForTokenClassification.from_pretrained(finetuned_path)

print(tokenized_datasets)
full_test_dataset = tokenized_datasets['tokens'][0]


test_results = model_loaded(full_test_dataset)
print(test_results)


