import os
import shutil
import yaml
import logging
import random
import re
import html

import torch

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SPLIT_KEYS = ["train", "val", "test"]
# DEVICE = "cpu"


def _handle_seed(seed_val=None):
    if isinstance(seed_val, int):
        return seed_val

    return random.randint(0, 2**32 - 1)


def write_yaml(filepath, dictionary):
    """
    Writes a Python dictionary to a YAML file.

    Args:
        filepath (str): The path of the YAML file where the dictionary will be written.
        dictionary (dict): The Python dictionary to be written into the YAML file.

    Returns:
        str: Success message if the operation is successful.
        str: Error message if an exception occurs.
    """
    try:
        logger.debug(
            f"Attempting to write YAML file at '{filepath}' with data: {dictionary}"
        )
        with open(filepath, "w") as f:
            yaml.safe_dump(dictionary, f)
        logger.info(
            f"YAML file '{filepath}' successfully created and parameters saved."
        )
        return f"YAML file '{filepath}' successfully created and parameters saved."
    except Exception as e:
        logger.error(f"Failed to write YAML file '{filepath}'. Error: {e}")
        return str(e)


def read_yaml(filepath):
    """
    Reads and parses a YAML file into a Python dictionary.

    Args:
        filepath (str): The path of the YAML file to read.

    Returns:
        dict: Parsed contents of the YAML file as a Python dictionary.
    """
    try:
        logger.debug(f"Attempting to read YAML file from '{filepath}'")
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        logger.info(f"YAML file '{filepath}' loaded successfully.")
        logger.debug(f"Configuration loaded: {data}")
        return data
    except Exception as e:
        logger.error(f"Failed to read YAML file '{filepath}'. Error: {e}")
        return str(e)


def make_dir(paths):
    """
    Creates a directory or directories.

    If the directory/directories exist, it replaces them by deleting and recreating them.

    Args:
        paths (str or list): A string representing a single directory path or
                             a list of strings for multiple directories.

    Returns:
        None
    """
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        try:
            if os.path.exists(path):
                logger.debug(f"Directory '{path}' already exists. Removing it.")
                shutil.rmtree(path)
                logger.info(f"Existing directory '{path}' removed.")
            os.makedirs(path)
            logger.info(f"Directory '{path}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create directory '{path}'. Error: {e}")


def get_split_config(config):
    """
    Extracts dataset split configuration from a given config dictionary.

    Args:
        config (dict): A dictionary containing dataset configuration and split keys.

    Returns:
        Tuple[str, Tuple[Any, Any, Any]]:
            - A description of the dataset (str).
            - A tuple containing configurations for train, val, and test splits.

    Raises:
        ValueError: If `dataset_config` is missing or empty, or if no split keys are found.
        KeyError: If split keys are not valid.
    """
    dataset_config = config.get("dataset_config", {})

    # Validate dataset_config
    if not dataset_config:
        raise ValueError(
            "The `dataset_config` key is either empty or not present in the config."
        )

    # Pop description with a default message if not provided
    description = dataset_config.pop("desc", "Dataset description not provided.")

    # Get the split keys and validate
    split_keys = [key for key in dataset_config.keys() if key in SPLIT_KEYS]
    if not split_keys:
        raise ValueError(f"At least one of the keys {SPLIT_KEYS} must be present.")
    if not set(split_keys).issubset(SPLIT_KEYS):
        raise KeyError(f"Dataset split keys must be a subset of {SPLIT_KEYS}.")

    # Extract configurations for splits
    train_split_config = dataset_config.get("train")
    val_split_config = dataset_config.get("val")
    test_split_config = dataset_config.get("test")

    return description, (train_split_config, val_split_config, test_split_config)


def get_dataset_config(dataset_config: dict):
    """
    Extracts and validates the dataset configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'dataset_config'.

    Returns:
        tuple: A tuple containing the dataset path and description.

    Raises:
        KeyError: If 'dataset_config' is missing or required keys are not found.
        ValueError: If the dataset configuration contains invalid values.
    """
    try:
        # Extract dataset_config

        # Validate required fields
        if "dataset_path" not in dataset_config:
            raise KeyError("Missing required key in 'dataset_config': 'dataset_path'")

        dataset_path = dataset_config.pop("dataset_path")
        sub_split_size = dataset_config.pop("sub_split_size")
        # Log the configuration
        logger.info(f"Loaded dataset configuration: {dataset_config}")

        return dataset_path, sub_split_size, dataset_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'dataset_config': {e}")
        raise


def get_tokenizer_config(config):
    """
    Extracts and validates the tokenizer configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'tokenizer_config'.

    Returns:
        tuple: A tuple containing the tokenizer name, tokenizer path, and any additional configuration.

    Raises:
        KeyError: If 'tokenizer_config' is missing or required keys are not found.
        ValueError: If the tokenizer configuration contains invalid values.
    """
    try:
        # Extract tokenizer_config
        tokenizer_config = config.get("tokenizer_config", {})

        # Validate required fields
        if (
            "tokenizer_name" not in tokenizer_config
            or "tokenizer_path" not in tokenizer_config
        ):
            raise KeyError(
                "Missing required keys in 'tokenizer_config': 'tokenizer_name' and/or 'tokenizer_path'"
            )

        # Log the configuration
        logger.info(f"Loaded tokenizer configuration: {tokenizer_config}")

        tokenizer_name = tokenizer_config.pop("tokenizer_name")
        tokenizer_path = tokenizer_config.pop("tokenizer_path")

        return (
            tokenizer_name,
            tokenizer_path,
            tokenizer_config,
        )

    except Exception as e:
        logger.error(f"An error occurred while processing 'tokenizer_config': {e}")
        raise


def get_model_config(config, pad_token_id=None, tokenizer_length=None):
    """
    Extracts and validates the model configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'model_config'.

    Returns:
        dict: The processed model configuration.

    Raises:
        KeyError: If 'model_config' is missing or required keys are not found.
        ValueError: If the model configuration contains invalid values.
    """
    try:
        # Extract model_config
        model_config = config.get("model_config", {})

        if pad_token_id:
            model_config["pad_token_id"] = pad_token_id

        if tokenizer_length:
            model_config["tokenizer_length"] = tokenizer_length

        # Validate required fields
        if "model_name" not in model_config:
            raise KeyError("Missing required keys in 'model_config': 'model_name'")

        # Log the configuration
        logger.info(f"Loaded model configuration: {model_config}")

        model_name = model_config.pop("model_name")
        model_path = model_config.pop("model_path", None)
        base_model_path = model_config.pop("base_model_path", None)

        return (model_name, model_path, base_model_path, model_config)

    except Exception as e:
        logger.error(f"An error occurred while processing 'model_config': {e}")
        raise


def get_lora_config(config):
    lora_config = config.get("lora_config", {})

    if not lora_config:
        logger.info("`lora_config` is missing or empty. Returning None.")
        return None

    logger.info(f"Loaded trainer configuration: {lora_config}")
    return lora_config


def get_trainer_config(config):
    """
    Extracts and validates the trainer configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'trainer_config'.

    Returns:
        dict or None: The processed trainer configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract trainer_config
        trainer_config = config.get("sft_trainer_config", {})

        # Check if trainer_config is empty
        if not trainer_config:
            logger.info("`sft_trainer_config` is missing or empty. Returning None.")
            return None

        seed = trainer_config.get("seed", "none")
        if isinstance(seed, str):
            trainer_config["seed"] = _handle_seed(seed_val=None)

        logger.info(f"Loaded trainer configuration: {trainer_config}")
        return trainer_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'trainer_config': {e}")
        raise


def get_generation_samples(config):
    """
    Extracts and validates the generation configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'generation_config'.

    Returns:
        dict or None: The processed generation configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract generation_config
        samples = config.get("generation_samples", {})

        # Check if generation_config is empty
        if not samples:
            return None

        logger.info(f"Loaded generation configuration: {samples}")

        return samples

    except Exception as e:
        logger.error(f"An error occurred while processing 'samples': {e}")
        raise


def get_generation_config(config):
    """
    Extracts and validates the generation configuration from the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing the configuration, including 'generation_config'.

    Returns:
        dict or None: The processed generation configuration if it exists and is not empty; otherwise, None.
    """
    try:
        # Extract generation_config
        generation_config = config.get("generation_config", {})

        # Check if generation_config is empty
        if not generation_config:
            return None

        logger.info(f"Loaded generation configuration: {generation_config}")

        return generation_config

    except Exception as e:
        logger.error(f"An error occurred while processing 'generation_config': {e}")
        raise
