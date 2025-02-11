import os
import random
import logging
from typing import Union, List, Optional

from datasets import load_dataset

EOT_TOKEN = "[EOT]"


def tokenize_text(
    text_samples: Union[str, List[str]],
    tokenizer,
    max_length=None,
    truncation=True,
    use_encode=True,
    padding=True,
    return_tensors=None,
):
    if use_encode:
        return tokenizer.encode(
            text_samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )

    return tokenizer(
        text_samples,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def format_prompt_func(examples, eot_token=EOT_TOKEN):
    output_texts = []
    conversations = examples["conversations"]
    for i in range(len(conversations)):
        text = f" {eot_token}".join(conversations[i]) + f" {eot_token}"
        output_texts.append(text)

    return output_texts


def load_lima_dataset(
    dataset_path: str, split: str = "train", sub_split_size: float = None, **kwargs
):
    dataset = load_dataset(dataset_path, split=split, **kwargs)
    if isinstance(sub_split_size, float) and 0.0 < sub_split_size <= 1.0:
        dataset = dataset.train_test_split(test_size=sub_split_size)["test"]

    return dataset
