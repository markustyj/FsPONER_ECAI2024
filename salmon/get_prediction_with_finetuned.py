import torch
import numpy as np
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from functools import cmp_to_key
from datasets import Dataset
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments

from transformers import Trainer
from datasets import load_dataset


train_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/train.jsonl")
val_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/val.jsonl")
test_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/test.jsonl")

#output_model='models/thin-film-finetuned',
finetuned_path = "/home/z004r5cc/Documents/models/thin-film-finetuned"
tokenizer_path='distilroberta-base'
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

def create_label_list(dataset_train: Dataset, dataset_val: Dataset, dataset_test: Dataset) -> Tuple:
    """
    Create a list of label from the dataset and return an encoding Dictionary.

    Parameters
    ----------
    dataset: HuggingFace dataset.

    Returns
    -------
    A tuple of a list of labels, and a dictionary of label encodings.
    """
    labels_set = set()

    dataset_train.map(lambda x: labels_set.update(x["ner_tags"]))
    dataset_val.map(lambda x: labels_set.update(x["ner_tags"]))
    dataset_test.map(lambda x: labels_set.update(x["ner_tags"]))

    label_list = sorted(list(labels_set), key=cmp_to_key(compare_labels))
    label_encoding_dict = {}
    for i, label in enumerate(label_list):
        label_encoding_dict[label] = i
    return label_list, label_encoding_dict

def tokenize_and_align_labels(
    examples: Any, tokenizer: Any, label_encoding_dict: Dict
) -> Any:
    """
    Align tokens to BPE/WordPiece tokens and adjust the labels accordingly.

    Parameters
    ----------
    examples: a sample/batch of samples from a dataset.
    tokenizer: a HuggingFace tokenizer.
    label_encoding_dict: a dictionary that contains the labels as keys and their
    encoded integer id as values.

    Returns
    -------
    Tokenized and adjusted sample(s).
    """
    label_all_tokens = True
    tokenized_inputs = tokenizer(
        list(examples["tokens"]), truncation=True, is_split_into_words=True, padding = "max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(
                    label_encoding_dict[label[word_idx]]
                    if label_all_tokens
                    else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compare_labels(item1: str, item2: str) -> int:
    """
    Compare function to sort labels so that their indices can be used as IDs.

    Parameters
    ----------
    item1: label name string.
    item2: label name string.

    Returns
    -------
    -1 or 1 or 0 according to the order.
    """
    # pylint: disable=too-many-return-statements
    if len(item1) == 1 and len(item2) == 1:
        if item1 > item2:
            return -1
        if item1 < item2:
            return 1
        return 0

    if len(item1) == 1:
        return -1
    if len(item2) == 1:
        return 1

    if item1 > item2:
        return -1
    if item1 < item2:
        return 1
    return 0






label_list, label_encoding_dict = create_label_list(raw_datasets["train"],raw_datasets["validation"],raw_datasets["test"])
#print(label_list, label_encoding_dict)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    use_fast=True,
    add_prefix_space=tokenizer_add_prefix_space, truncation=True
)

def tokenize_function(examples):
    return tokenize_and_align_labels(
        examples, tokenizer, label_encoding_dict
    )

tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True  , num_proc=4
)
#print(tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding = True)

model_loaded = AutoModelForTokenClassification.from_pretrained(finetuned_path)


training_args = TrainingArguments("test-trainer")
trainer = Trainer(
        model=model_loaded,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        #compute_metrics=lambda x: compute_ner_metrics(x, label_list),
    )

#Although data_collator in Trainer, not in test_dataset.
predictions = trainer.predict(tokenized_datasets["test"])
#print(predictions.predictions.shape, predictions.label_ids)
preds = np.argmax(predictions.predictions, axis=-1)
for i,row in enumerate(preds):
    for j,element in enumerate(row):
        if element > 0.1 and j > 0.1 and j<= len(tokenized_datasets["test"]["tokens"][i]):
            #print(i,j)
            #print(element)
            print(tokenized_datasets["test"]["tokens"][i][j-1])


'''
full_test_dataset = tokenized_datasets["test"]
#print(full_test_dataset["labels"][:3])
#print(full_test_dataset["input_ids"][0:3])
input_from_model = {"input_ids":  torch.Tensor (full_test_dataset["input_ids"][:2]).to(torch.int64),  'attention_mask': torch.Tensor(full_test_dataset["attention_mask"][:2]).to(torch.int64)}
#print(input_from_model)
logits = model_loaded(**input_from_model).logits
predicted_token_class_ids = logits.argmax(-1)
predicted_tokens_classes = [model_loaded.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
#print(predicted_tokens_classes)
'''


    

