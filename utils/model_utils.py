import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Model configuration table for custom initialization
MODEL_CONFIG_TABLE = {
    1: {"hidden_size": 64, "intermediate_size": 256, "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2},
    2: {"hidden_size": 128, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2},
    3: {"hidden_size": 192, "intermediate_size": 768, "num_hidden_layers": 3, "num_attention_heads": 6, "num_key_value_heads": 3},
    4: {"hidden_size": 256, "intermediate_size": 1024, "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 4},
    5: {"hidden_size": 320, "intermediate_size": 1280, "num_hidden_layers": 5, "num_attention_heads": 8, "num_key_value_heads": 4},
    6: {"hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 6, "num_attention_heads": 8, "num_key_value_heads": 4},
    7: {"hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 7, "num_attention_heads": 10, "num_key_value_heads": 5},
    8: {"hidden_size": 640, "intermediate_size": 2560, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
    9: {"hidden_size": 704, "intermediate_size": 2816, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
    10: {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 8, "num_attention_heads": 12, "num_key_value_heads": 6},
}


def load_model_and_tokenizer(
    model_name,
    cache_dir,
    device,
    train_from_scratch=False,
    size_level=-1,
    use_digit_wise_tokenizer=False
):
    """
    Loads a model and tokenizer. Optionally initializes a new model from scratch
    with parameters defined by a given size_level and adds special tokens.
    Ensures the model and tokenizer vocabularies are consistent.
    """
    # ------------------------------------------------------------------
    # 1) Load/Initialize the Model
    # ------------------------------------------------------------------
    if train_from_scratch:
        if size_level == -1:
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_config(config).to(device)
            logging.info("Model initialized from scratch with default configuration.")
        else:
            if size_level not in MODEL_CONFIG_TABLE:
                raise ValueError(f"Invalid size level '{size_level}'. Available: 1 to 8")
            config_params = MODEL_CONFIG_TABLE[size_level]
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            config.update(config_params)
            model = AutoModelForCausalLM.from_config(config).to(device)
            logging.info(f"Model initialized from scratch with size level: {size_level}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            cache_dir=cache_dir
        ).to(device)
        logging.info("Pre-trained model loaded!")

    # ------------------------------------------------------------------
    # 2) Load the Tokenizer
    # ------------------------------------------------------------------
    if use_digit_wise_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=cache_dir
        )
        logging.info("Digit-wise tokenizer loaded from 'meta-llama/Llama-2-7b-hf'.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logging.info("Standard tokenizer loaded.")

    # ------------------------------------------------------------------
    # 3) Add Any Extra Special Tokens You Need
    # ------------------------------------------------------------------
    special_tokens_added = False
    if '[PAD]' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        special_tokens_added = True
    if '[NUM]' not in tokenizer.get_vocab():
        tokenizer.add_tokens(["[NUM]"])
        special_tokens_added = True

    # ------------------------------------------------------------------
    # 4) Expand the Smaller One (Tokenizer vs. Model Config)
    # ------------------------------------------------------------------
    config_vocab_size = getattr(model.config, "vocab_size", None)
    tokenizer_vocab_size = len(tokenizer)

    if config_vocab_size is not None:
        if config_vocab_size > tokenizer_vocab_size:
            missing = config_vocab_size - tokenizer_vocab_size
            dummy_tokens = [f"[DUMMY_{i}]" for i in range(missing)]
            tokenizer.add_tokens(dummy_tokens)
            logging.info(f"Expanded tokenizer by {missing} dummy tokens to match model vocab_size={config_vocab_size}.")
        elif config_vocab_size < tokenizer_vocab_size:
            logging.info(f"Expanding model config from {config_vocab_size} to {tokenizer_vocab_size} to match tokenizer.")
            model.config.vocab_size = tokenizer_vocab_size

    final_vocab_size = len(tokenizer)
    model.resize_token_embeddings(final_vocab_size)
    model.config.vocab_size = final_vocab_size

    # ------------------------------------------------------------------
    # 5) Log the Actual Model Size
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_millions = total_params / 1_000_000  # Convert to millions
    logging.info(f"Actual model size (total parameters): {total_params_in_millions:.2f}M")
    logging.info(f"Padding token ID: {tokenizer.pad_token_id}")
    if not use_digit_wise_tokenizer:
        num_id = tokenizer.convert_tokens_to_ids('[NUM]')
        logging.info(f"[NUM] token ID: {num_id}")

    return model, tokenizer
