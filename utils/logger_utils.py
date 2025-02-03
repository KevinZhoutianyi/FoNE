import logging
import os
from datetime import datetime
import torch
# Setup logger
def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get current date and time for the log file name
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(output_dir, f'training_log_{current_time}.log')

    # Configure the logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # Overwrite existing log file
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO  # Set logging level to INFO
    )
# Function to save model
def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokenizer saved to {output_dir}")

def get_output_folder(args):
    """Generates an output folder path based on arguments."""
    method_str = args.method  # Use method directly for clarity
    scratch_str = "train_from_scratch" if args.train_from_scratch else "pretrained"

    # Handle dataset name depending on whether it's a tuple
    if isinstance(args.dataset, tuple):
        dataset_name = f"{args.dataset[0].split('/')[-1]}_{args.dataset[1]}"
    else:
        dataset_name = args.dataset.split('/')[-1]

    # Move args.name as a folder before dataset name
    name_folder = args.name

    if args.method == 'fne':
        # Construct folder name based on arguments
        other_args = (
            f"period_base_list_{args.period_base_list}_batchsize_{args.batch_size}_"
            f"epochs_{args.epochs}_lr_{args.lr}_int_digit_len_{args.int_digit_len}_"
            f"frac_digit_len_{args.frac_digit_len}_seed{args.seed}_"
            f"num_train_samples_{args.num_train_samples}_model_size_level_{args.model_size_level}_lengensize_{args.len_gen_size}_addlinear_{not args.not_add_linear}"
        )
    elif args.method == 'regular':
        other_args = (
            f"batchsize_{args.batch_size}_"
            f"epochs_{args.epochs}_lr_{args.lr}_"
            f"seed{args.seed}_"
            f"num_train_samples_{args.num_train_samples}_model_size_level_{args.model_size_level}_usedigitwisetokenizer_{args.use_digit_wise_tokenizer}"
        )
    elif args.method == 'vanilla':
        other_args = (
            f"batchsize_{args.batch_size}_"
            f"epochs_{args.epochs}_lr_{args.lr}_int_digit_len_{args.int_digit_len}_"
            f"frac_digit_len_{args.frac_digit_len}_seed{args.seed}_"
            f"num_train_samples_{args.num_train_samples}_model_size_level_{args.model_size_level}_lengensize_{args.len_gen_size}_addlinear_{not args.not_add_linear}"
        )
    elif args.method == 'xval':
        other_args = (
            f"batchsize_{args.batch_size}_"
            f"epochs_{args.epochs}_lr_{args.lr}_"
            f"seed{args.seed}_"
            f"num_train_samples_{args.num_train_samples}_model_size_level_{args.model_size_level}"
        )

    # Combine all parts into a structured output folder path
    output_folder = os.path.join("result",  name_folder, scratch_str, args.model,  dataset_name, method_str, other_args)

    # Ensure the directory exists
    os.makedirs(output_folder, exist_ok=True)

    return output_folder

def get_embedding_dim(model):
    # Try different attribute names based on the model type
    if hasattr(model.config, 'n_embd'):  # GPT models
        return model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):  # LLaMA models
        return model.config.hidden_size
    else:
        raise AttributeError(f"Unknown model configuration: {model.config}")
def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
    print(f"Allocated: {allocated:.4f} GB, Reserved: {reserved:.4f} GB")
