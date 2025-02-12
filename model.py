import os
import random
import logging
from typing import Union, List, Optional

import torch
import torch.nn as nn
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
)

from utils import DEVICE
from lima_dataset import EOT_TOKEN, tokenize_text

ADDITIONAL_TOKENS_MAP = {"EOT_TOKEN": EOT_TOKEN}

# import evaluate
# from sentiment_dataset import tokenize_text, _INDEX2LABEL
# METRICS_DICT = dict(
#     accuracy=evaluate.load("accuracy"), auc=evaluate.load("roc_auc", "multiclass")
# )

# Configure logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set logger to capture DEBUG and above messages


def manage_tokens(tokenizer, special_token_kwargs):
    pad_token = special_token_kwargs.get("pad_token", tokenizer.eos_token)
    if pad_token == "eos_token":
        pad_token = tokenizer.eos_token

    tokenizer.pad_token = pad_token
    additional_tokens = special_token_kwargs.get("additional_tokens", [])
    if len(additional_tokens):
        additional_tokens = list(
            map(lambda x: ADDITIONAL_TOKENS_MAP[x], additional_tokens)
        )
        tokenizer.add_tokens(additional_tokens)


def load_pretrained_llama2_tokenizer(
    tokenizer_path="meta-llama/Llama-2-7b-hf", **kwargs
):
    """
    Loads a pre-trained Llama2 tokenizer and adds a special padding token.
    """
    tokenizer_config = kwargs.pop("config", {})
    special_token_kwargs = kwargs.pop("special_token_kwargs", {})

    if not isinstance(tokenizer_config, dict):
        logger.error(
            f"Found tokenizer_config of type: {type(tokenizer_config)}; expected config of type: Dict"
        )
    try:
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, **tokenizer_config)
        manage_tokens(tokenizer, special_token_kwargs)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_tokens([EOT_TOKEN])
        logger.debug("Special tokens added to the tokenizer.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load Llama2 tokenizer from {tokenizer_path}: {e}")
        raise


def load_pretrained_base_llama2_model(
    base_model_path="meta-llama/Llama-2-7b-hf",
    # pad_token_id=None,
    # tokenizer_length=None,
    **kwargs,
):
    config = kwargs.pop("config", {})
    bnb_config = config.pop("bnb_config", {})
    pad_token_id = config.pop("pad_token_id", None)
    tokenizer_length = config.pop("tokenizer_length", None)

    if isinstance(bnb_config, dict):
        bnb_config = BitsAndBytesConfig(**bnb_config)

    elif isinstance(bnb_config, BitsAndBytesConfig):
        pass

    else:
        raise ValueError(
            f"Expected type dict() or BitsAndBytesConfig() for bnb_config found {type(bnb_config)}"
        )

    if not isinstance(kwargs, dict):
        logger.error(
            f"Found model_config of type: {type(kwargs)}; expected config of type: Dict"
        )
    try:
        model = LlamaForCausalLM.from_pretrained(
            base_model_path, quantization_config=bnb_config, **config
        )

        if pad_token_id is not None:
            model.config.pad_token_id = pad_token_id
        if tokenizer_length is not None:
            model.resize_token_embeddings(tokenizer_length)

        logger.debug("Base model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load base Llama2 model from {base_model_path}: {e}")
        raise


def load_finetuned_llama2_model(
    model_path,
    **kwargs,
):
    if "base_model_path" not in kwargs:
        kwargs.update("base_model_path", "meta-llama/Llama-2-7b-hf")

    try:
        base_model = load_pretrained_base_llama2_model(**kwargs)
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.debug("Finetuned Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load Llama2 model from {model_path}: {e}")
        raise


TOKENIZER_DICT = {
    "llama2": (
        load_pretrained_llama2_tokenizer,
        {
            "tokenizer_path": "meta-llama/Llama-2-7b-hf",
            "config": {},
            "special_token_kwargs": {
                "pad_token": "eos_token",
                "additional_tokens": ["EOT_TOKEN"],
            },
        },
    ),
}

MODEL_DICT = {
    "llama2-base": (
        load_pretrained_base_llama2_model,
        {
            "base_model_path": "meta-llama/Llama-2-7b-hf",
            "config": {
                "bnb_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, "float16"),
                    bnb_4bit_use_double_quant=False,
                ),
            },
        },
    ),
    "llama2-finetuned": (
        load_finetuned_llama2_model,
        {
            "model_path": None,
            "base_model_path": "meta-llama/Llama-2-7b-hf",
            "config": {
                "bnb_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=getattr(torch, "float16"),
                    bnb_4bit_use_double_quant=False,
                ),
            },
        },
    ),
}


def load_tokenizer(tokenizer_name, tokenizer_path=None, tokenizer_config=None):
    """
    Dynamically fetch and initialize a tokenizer based on the tokenizer string.

    Args:
        tokenizer_name (str): The key corresponding to the desired tokenizer in TOKENIZER_DICT.
        tokenizer_path (str, optional): Custom tokenizer path to override the default path in TOKENIZER_DICT.
        tokenizer_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT
    Returns:
        Tokenizer object initialized with the specified parameters.

    Raises:
        ValueError: If the tokenizer string is not registered in TOKENIZER_DICT.
    """
    if tokenizer_name in TOKENIZER_DICT:
        func, kwargs = TOKENIZER_DICT[tokenizer_name]

        # Dynamically update kwargs based on provided arguments
        if tokenizer_path is not None:
            kwargs["tokenizer_path"] = tokenizer_path

        if tokenizer_config is not None:
            kwargs["config"] = tokenizer_config

        logger.info(
            f"Initializing tokenizer '{tokenizer_name}' with arguments: {kwargs}"
        )
        return func(**kwargs)
    else:
        logger.error(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )
        raise ValueError(
            f"Tokenizer '{tokenizer_name}' is not registered in TOKENIZER_DICT."
        )


def load_model(
    model_string,
    model_path=None,
    base_model_path=None,
    model_config=None,
    # device=DEVICE,
):
    """
    Dynamically fetch and initialize a model based on the model string.

    Args:
        model_string (str): The key corresponding to the desired model in MODEL_DICT.
        model_path (str, optional): Custom model path to override the default path in MODEL_DICT.
        model_config (dict, optional): Custom configuration to override the default configuration in MODEL_DICT.

    Returns:
        Model object initialized with the specified parameters.

    Raises:
        ValueError: If the model string is not registered in MODEL_DICT.
    """
    if model_string in MODEL_DICT:
        func, kwargs = MODEL_DICT[model_string]

        # Dynamically update kwargs based on provided arguments
        if model_path is not None:
            kwargs["model_path"] = model_path

        if base_model_path is not None:
            kwargs["base_model_path"] = base_model_path

        if model_config is not None:
            kwargs["config"] = model_config

        logger.info(f"Initializing model '{model_string}' with arguments: {kwargs}")
        # return func(**kwargs).to(device=device)
        return func(**kwargs)
    else:
        logger.error(f"Model '{model_string}' is not registered in MODEL_DICT.")
        raise ValueError(f"Model '{model_string}' is not registered in MODEL_DICT.")


def load_lora_model(model, lora_config: dict):
    """
    Load a model with LoRA (Low-Rank Adaptation) configurations for efficient fine-tuning.

    This function prepares the model for low-bit (k-bit) training and applies LoRA configurations
    to integrate parameter-efficient fine-tuning (PEFT) capabilities.

    Args:
        model (PreTrainedModel):
            The base Hugging Face model to which LoRA configurations will be applied.
        lora_config (dict):
            A dictionary containing LoRA configuration parameters. Example keys include:
                - `r` (int): Rank of the low-rank decomposition.
                - `lora_alpha` (float): Scaling factor for LoRA updates.
                - `lora_dropout` (float): Dropout rate for LoRA layers.
                - `bias` (str): Bias handling strategy, e.g., "none", "all", or "lora_only".

    Returns:
        PreTrainedModel:
            The input model modified to support LoRA-based fine-tuning.

    """
    # Step 1: Create LoRA configuration using the provided dictionary
    peft_config = LoraConfig(**lora_config)

    # Step 2: Prepare the model for k-bit training (quantized training support)
    model = prepare_model_for_kbit_training(model)

    # Step 3: Apply the LoRA configuration to the model
    model = get_peft_model(model, peft_config)

    # Return the LoRA-modified model
    return model


def generate(
    model,
    tokenizer,
    prompt_samples: Union[str, List[str]],
    generation_config=None,
    use_encode: bool = False,
    return_tensors: str = "pt",
    device: str = DEVICE,
    eot_token=EOT_TOKEN,
):
    """
    Generates text.
    Iterates through each sample separately to ensure better handling.
    """
    if isinstance(prompt_samples, str):
        prompt_samples = [prompt_samples]
    elif not isinstance(prompt_samples, list) or not all(
        isinstance(p, str) for p in prompt_samples
    ):
        logger.error(
            "Invalid prompt provided. Prompt must be a non-empty string or list of strings."
        )
        raise ValueError("Prompt must be a non-empty string or list of strings.")

    logger.info(f"Generating text for {len(prompt_samples)} prompts.")
    results = []

    try:
        generation_config = (
            GenerationConfig(**generation_config) if generation_config else {}
        )
        for prompt in prompt_samples:
            prompt = f"{prompt}{eot_token}"
            # prompt = f"{prompt}"
            logger.debug(f"Tokenizing prompt: {prompt}")
            tokenized_prompt = tokenize_text(
                prompt,
                tokenizer,
                use_encode=use_encode,
                return_tensors=return_tensors,
            ).to(device=device)
            # tokenized_prompt = {
            #     key: value.to(device) for key, value in tokenized_prompt.items()
            # }

            logger.debug("Starting text generation.")
            generated_tokens = model.generate(
                tokenized_prompt,
                generation_config=generation_config,  # Pass config if available
            )

            decoded_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            results.append(decoded_text[0])  # Append the first generated result

        logger.info("Text generation completed successfully.")
        return results

    except Exception as e:
        logger.error(f"An error occurred during text generation: {e}")
        raise RuntimeError(f"An error occurred during text generation: {e}")


def compute_metrics(data, metrics):
    predictions, labels = data
    predict_scores = torch.softmax(torch.from_numpy(predictions), dim=-1).numpy()
    predict_labels = torch.argmax(torch.from_numpy(predictions), dim=-1).numpy()

    results = {}
    if "accuracy" in metrics:
        acc_metric = metrics["accuracy"].compute(
            predictions=predict_labels, references=labels
        )
        results["accuracy"] = round(acc_metric["accuracy"], 5)

    if "auc" in metrics:
        auc_metric = metrics["auc"].compute(
            prediction_scores=predict_scores, references=labels, multi_class="ovr"
        )
        results["auc"] = round(acc_metric["roc_acu"], 5)

    return results
