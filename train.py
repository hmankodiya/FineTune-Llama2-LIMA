import os
import random
from argparse import ArgumentParser
import logging

import torch
from trl import SFTConfig, SFTTrainer

from lima_dataset import load_lima_dataset, tokenize_text, format_prompt_func, EOT_TOKEN
from utils import (
    read_yaml,
    get_model_config,
    get_tokenizer_config,
    get_split_config,
    get_dataset_config,
    get_trainer_config,
    get_lora_config,
    _handle_seed,
)
from model import (
    load_model,
    load_tokenizer,
    load_lora_model,
)


logging.basicConfig(
    filename="./logs.txt",
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    level=logging.INFO,
    filemode="w",
)
logger = logging.getLogger(__name__)  # Logger for the main script


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        default="./train_config.yaml",
        help="Path to the configuration file (YAML or JSON format).",
    )
    arg_parser.add_argument(
        "--use_lora",
        type=bool,
        required=False,
        default=True,
        help="Lora for model.",
    )

    args = arg_parser.parse_args()
    config = read_yaml(args.config)

    tokenizer_name, tokenizer_path, tokenizer_config = get_tokenizer_config(config)
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_path=tokenizer_path,
        tokenizer_config=tokenizer_config,
    )
    tokenizer.add_tokens([EOT_TOKEN])

    dataset_desc, (train_split_config, val_split_config, test_split_config) = (
        get_split_config(config)
    )

    train_dataset_path, train_sub_split_size, train_dataset_config = get_dataset_config(
        train_split_config
    )
    train_dataset = load_lima_dataset(
        train_dataset_path, "train", train_sub_split_size, **train_dataset_config
    )
    logger.info(
        f"Loaded Train Dataset: {dataset_desc}, Dataset Length: {len(train_dataset)} with sub_split_size {train_sub_split_size if train_sub_split_size else None}."
    )

    model_name, model_path, base_model_path, model_config = get_model_config(config)
    model = load_model(
        model_string=model_name,
        model_path=model_path,
        base_model_path=base_model_path,
        model_config=model_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        lora_config = get_lora_config(config)
        model = load_lora_model(model, lora_config)

    trainer_config = get_trainer_config(config)
    trainer_config["logging_dir"] = os.path.join(
        trainer_config["output_dir"], "runs", trainer_config["run_name"]
    )
    save_trained_model = trainer_config.pop("save_trained_model", True)
    resume_from_checkpoint = trainer_config.pop("resume_from_checkpoint", None)
    sft_trainer_args = SFTConfig(**trainer_config)
    sft_trainer = SFTTrainer(
        model,
        args=sft_trainer_args,
        train_dataset=train_dataset,
        formatting_func=format_prompt_func,
        processing_class=tokenizer,
    )

    logger.info("Training started.")
    training_outs = sft_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training finished.")

    if save_trained_model:
        logger.info(f'Saving model at {trainer_config["logging_dir"]}')
        model.save_pretrained(os.path.join(trainer_config["logging_dir"], "model"))
