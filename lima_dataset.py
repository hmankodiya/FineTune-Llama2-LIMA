import os
import random
import logging
from typing import Union, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
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


def format_prompt_func_each(text_field, eot_token=EOT_TOKEN):
    text_field = [f"{prompt}{eot_token}" for prompt in text_field]
    return text_field


def format_prompt_func(examples, eot_token=EOT_TOKEN):
    output_texts = []
    for i, example in enumerate(examples["conversations"]):
        text_field = f"{eot_token}".join(example) + f"{eot_token}"
        output_texts.append(text_field)

    return output_texts
    # f"{prompt}{eot_token}" for prompt in text_field


def load_lima_dataset(
    dataset_path: str, split: str = "train", sub_split_size: float = None, **kwargs
):
    dataset = load_dataset(dataset_path, split=split, **kwargs)
    if isinstance(sub_split_size, float) and 0.0 < sub_split_size <= 1.0:
        dataset = dataset.train_test_split(test_size=sub_split_size)["test"]

    return dataset


class InstructionDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        split_type,
        sub_split_size,
        dataset_kwargs,
        tokenizer,
        format_prompt_func=None,
        text_field="conversations",
        return_tensors=None,
        truncation=True,
        max_length=1024,
        padding=True,
        ignore_index=-100,
    ):
        self.dataset_path = dataset_path
        self.split_type = split_type
        self.sub_split_size = sub_split_size
        self.dataset_kwargs = dataset_kwargs
        self.dataset = load_lima_dataset(
            self.dataset_path,
            split=self.split_type,
            sub_split_size=sub_split_size,
            **self.dataset_kwargs,
        )
        self.text_field = text_field
        self.format_prompt_func = format_prompt_func
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.ignore_index = ignore_index

    def __len__(self):
        # return 10
        return len(self.dataset)

    def __getitem__(self, index):
        instance = self.dataset[index]
        text_field_val = instance[self.text_field]

        if self.format_prompt_func is not None:
            instance = self.format_prompt_func(text_field_val)
        else:
            instance = text_field_val

        tokenized_instance = tokenize_text(
            instance,
            self.tokenizer,
            truncation=self.truncation,
            max_length=self.max_length,
            padding=self.padding,
            use_encode=False,
            return_tensors=self.return_tensors,
        )

        return tokenized_instance
