import argparse
import os
import logging
import torch
import wandb

from utils.logger_utils import setup_logger, get_output_folder
from utils.model_utils import load_model_and_tokenizer
from train.train_pipeline import (
    parse_period_base_list,
    create_dataloader_and_train
)

# Set environment variables for Hugging Face and WandB tokens
os.environ["HF_TOKEN"] = "{your_HF_token}"
os.environ["WANDB_API_KEY"] = "{your_WandB_API_key}"
wandb.login()

def main():
    parser = argparse.ArgumentParser(
        description="Training LLMs with custom embeddings and Fourier loss on TS datasets"
    )
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--int_digit_len', type=int, default=5, help='Number of digits for integer part')
    parser.add_argument('--frac_digit_len', type=int, default=5, help='Number of digits for fractional part')
    parser.add_argument('--len_gen_size', type=int, default=0, help='FNE: add k 0s after numbers to len gen')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--name', type=str, default='', help='Log name')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--dataset', type=str, default='Onlydrinkwater/language_math_10base', help='Dataset name')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from scratch without pre-trained weights')
    parser.add_argument('--use_digit_wise_tokenizer', action='store_true', help='Whether to use digit-wise tokenizer')
    parser.add_argument('--num_train_samples', type=int, default=None, help='Number of training samples to use')
    parser.add_argument('--num_test_samples', type=int, default=None, help='Number of test samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_size_level', type=int, default=-1, help='From 1 to 8, choose the model size for training from scratch')
    parser.add_argument('--method', type=str, choices=['regular', 'fne', 'xval', 'vanilla'], default='fne', help='Training method: regular, fne, xval, or vanilla')
    parser.add_argument('--scheduler_name', type=str, default='cosine', help='Name of the learning rate scheduler (e.g., linear, constant, cosine, etc.)')
    parser.add_argument('--period_base_list', type=str, nargs='+', default=['10'], help='List of period bases for Fourier embedding (e.g., 2, 5, 1/3)')
    parser.add_argument('--clip', action='store_true', help='Enable clipping')
    parser.add_argument('--not_add_linear', action='store_true', help='Do not add linear layer after FNE')
    
    args = parser.parse_args()
    
    # Convert period_base_list strings to floats (handling fractions)
    args.add_linear = not args.not_add_linear
    args.period_base_list = parse_period_base_list(args.period_base_list)
    
    run_name = f"{args.name}{args.method}_{args.model}_{args.dataset}_seed{args.seed}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    wandb.init(
        project="FoNE_TS",
        config=vars(args),
        name=run_name
    )
    
    output_folder = get_output_folder(args)
    setup_logger(output_folder)
    
    logging.info(output_folder)
    logging.info(args)
    
    if ',' in args.dataset:
        args.dataset = tuple(args.dataset.split(','))
        logging.info(f"Dataset specified as tuple: {args.dataset}")
    else:
        logging.info(f"Dataset specified as single name: {args.dataset}")
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        cache_dir="/home/tianyizhou/hg_cache",
        device=device,
        train_from_scratch=args.train_from_scratch,
        size_level=args.model_size_level,
        use_digit_wise_tokenizer=args.use_digit_wise_tokenizer
    )
    
    # Run the training pipeline
    create_dataloader_and_train(args, model, tokenizer, device)
    
    wandb.finish()

if __name__ == "__main__":
    main()
