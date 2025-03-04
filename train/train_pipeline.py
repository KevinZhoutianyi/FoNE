import logging
import re
from fractions import Fraction
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
from datasets import load_dataset

from train.train import (
    train_fne, 
    train_regular,
    train_xval, 
    train_vanilla, 
)
from train.eval import (
    evaluate_fne,
    evaluate_regular,
    evaluate_xval,
    evaluate_vanilla
)
from utils.data_utils import collate_fn
from number_encoders.FNE import FNE
from number_encoders.XVAL import XVAL
from number_encoders.vanilla import VanillaEmbedding
from utils.logger_utils import get_embedding_dim

# --- Standard Helper Functions ---

def extract_max_num_from_dataset(dataset_name):
    """
    Extracts the maximum number from the given dataset.
    """
    dataset = load_dataset(dataset_name)
    max_number = float('-inf')
    
    def find_max_number(example):
        all_values = ' '.join(map(str, example.values()))
        numbers = [float(num) for num in re.findall(r'\d+\.?\d*', all_values)]
        return {"max_number": max(numbers, default=None)}
    
    for split in dataset.keys():
        max_in_split = dataset[split].map(find_max_number)
        max_numbers = [num["max_number"] for num in max_in_split if num["max_number"] is not None]
        if max_numbers:
            max_number = max(max_number, max(max_numbers))
    
    if max_number == float('-inf'):
        raise ValueError(f"Could not extract max number from dataset: {dataset_name}")
    
    logging.info(f"max number : {max_number}")
    return max_number

def parse_period_base_list(period_base_list):
    """
    Converts a list of string representations (including fractions) into floats.
    """
    return [float(Fraction(base)) for base in period_base_list]



def load_and_prepare_data(args, tokenizer):
    """
    Loads and preprocesses the dataset for data.
    """
    from utils.data_utils import load_and_preprocess_dataset  # Local import if needed
    train_data, test_data = load_and_preprocess_dataset(
        args.dataset, tokenizer, args.num_train_samples, args.num_test_samples, method=args.method
    )
    logging.info(f"2 data example: {train_data[:2]}")
    num_token = tokenizer.convert_tokens_to_ids("[NUM]") if args.method in ['fne', 'xval', 'vanilla'] else None
    return train_data, test_data, num_token

def create_data_loaders(train_data, test_data, tokenizer, num_token, args):
    """
    Creates DataLoaders for training and testing (for text-based datasets).
    """
    if args.num_train_samples == 0:
        logging.info("Skipping training as num_train_samples is set to 0.")
        train_loader = None
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer, num_token, method=args.method)
        )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, num_token, method=args.method)
    )
    return train_loader, test_loader

# --- Optimizer & Scheduler ---

def initialize_optimizer_and_scheduler(model, train_loader, args):
    """
    Initializes the optimizer and learning rate scheduler.
    """
    from transformers import get_scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.2 * total_steps)
    scheduler = get_scheduler(
        name=args.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

# --- Training & Evaluation Functions (unchanged) ---

def run_epoch(model, train_loader, test_loader, optimizer, scheduler, number_encoder, args, epoch, device, tokenizer=None):
    """
    Runs one training epoch and evaluates the model.
    """
    if args.method == 'regular':
        train_loss = train_regular(model, train_loader, optimizer, scheduler, device, args)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_regular(
            model, test_loader, tokenizer, device, print_labels=True, max_print_examples=5
        )
    elif args.method == 'fne':
        train_loss = train_fne(model, train_loader, number_encoder, optimizer, scheduler, args,
                               args.int_digit_len, args.frac_digit_len, args.len_gen_size, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_fne(
            model, test_loader, number_encoder, args.int_digit_len, args.frac_digit_len, device,
            print_labels=True, max_print=5
        )
    elif args.method == 'xval':
        train_loss = train_xval(model, train_loader, number_encoder, optimizer, scheduler, args, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_xval(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    elif args.method == 'vanilla':
        train_loss = train_vanilla(model, train_loader, number_encoder, optimizer, scheduler, args, device)
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_vanilla(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    else:
        raise ValueError(f"Unsupported method '{args.method}'.")
    
    logging.info(f"{'Epoch':<40}{epoch + 1:<10}")
    logging.info(f"{'Train Loss':<40}{train_loss:.4f}")
    logging.info(f"{'Test Loss':<40}{test_loss:.4f}")
    logging.info(f"{'Whole Num Acc':<40}{whole_number_accuracy * 100:.6f}%")
    logging.info(f"{'Digit-wise Acc':<40}{digit_wise_accuracy * 100:.6f}%")
    logging.info(f"{'MSE':<40}{mse:.6f}")
    logging.info(f"{'R^2':<40}{r2:.15f}")
    logging.info(f"{'Learning Rate':<40}{scheduler.get_last_lr()[0]:.6f}")
    
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "whole_number_accuracy": whole_number_accuracy * 100,
        "digit_wise_accuracy": digit_wise_accuracy * 100,
        "mse": mse,
        "r2": r2,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    return whole_number_accuracy

def evaluate_model(model, test_loader, tokenizer, number_encoder, args, device, stage="Initial"):
    """
    Evaluates the model on the test set.
    """
    logging.info(f"Starting {stage} evaluation.")
    
    if args.method == 'regular':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_regular(
            model, test_loader, tokenizer, device, print_labels=True, max_print_examples=10
        )
    elif args.method == 'fne':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_fne(
            model, test_loader, number_encoder, args.int_digit_len, args.frac_digit_len, device,
            print_labels=True, max_print=5
        )
    elif args.method == 'xval':
        test_loss, (whole_number_accuracy, digit_wise_accuracy), mse, r2 = evaluate_xval(
            model, test_loader, number_encoder, device, print_labels=True, max_print=5
        )
    else:
        raise ValueError(f"Unsupported method '{args.method}'.")
    
    logging.info(f"{stage} Test Results:")
    logging.info(f"{'Test Loss':<40}{test_loss:.4f}")
    logging.info(f"{'Whole Num Accuracy':<40}{whole_number_accuracy * 100:.6f}%")
    logging.info(f"{'Digit-wise Accuracy':<40}{digit_wise_accuracy * 100:.6f}%")
    logging.info(f"{'MSE':<40}{mse:.6f}")
    logging.info(f"{'R^2':<40}{r2:.15f}")
    
    wandb.log({
        f"{stage.lower()}_test_loss": test_loss,
        f"{stage.lower()}_whole_number_accuracy": whole_number_accuracy * 100,
        f"{stage.lower()}_digit_wise_accuracy": digit_wise_accuracy * 100,
        f"{stage.lower()}_mse": mse,
        f"{stage.lower()}_r2": r2
    })
    
    return test_loss, whole_number_accuracy, digit_wise_accuracy, mse, r2

# --- Main DataLoader & Training Pipeline ---

def create_dataloader_and_train(args, model, tokenizer, device):
    """
    Prepares data loaders and executes the training and evaluation pipeline.
    """

    # Load text (or tabular) data using your existing function
    train_data, test_data, num_token = load_and_prepare_data(args, tokenizer)
    train_loader, test_loader = create_data_loaders(train_data, test_data, tokenizer, num_token, args)
    tok = tokenizer

    # Initialize the appropriate number encoder based on the method
    number_encoder = None
    embedding_dim = get_embedding_dim(model)
    if args.method == 'fne':
        number_encoder = FNE(
            embedding_dim,
            int_digit_len=args.int_digit_len,
            frac_digit_len=args.frac_digit_len,
            period_base_list=args.period_base_list,
            add_linear=args.add_linear,
            device=device
        ).to(device)
    elif args.method == 'vanilla':
        number_encoder = VanillaEmbedding(
            embedding_dim,
            int_digit_len=args.int_digit_len,
            frac_digit_len=args.frac_digit_len,
            device=device
        ).to(device)
    elif args.method == 'xval':
        max_num = extract_max_num_from_dataset(args.dataset)
        number_encoder = XVAL(embedding_dim=embedding_dim, max_num=max_num, device=device).to(device)
    
    # If no training samples are specified, run evaluation only
    if args.num_train_samples == 0:
        evaluate_model(model, test_loader, tok, number_encoder, args, device, stage="Single Evaluation")
        return
    
    optimizer, scheduler = initialize_optimizer_and_scheduler(model, train_loader, args)
    
    for epoch in range(args.epochs):
        logging.info('-' * 100)
        logging.info(f"Starting Epoch {epoch + 1}/{args.epochs}")
        whole_number_accuracy = run_epoch(model, train_loader, test_loader, optimizer, scheduler, number_encoder, args, epoch, device, tok)
        if whole_number_accuracy == 1.0:
            logging.info("Stopping early as whole number accuracy reached 100%.")
            break
    
    evaluate_model(model, test_loader, tok, number_encoder, args, device, stage="Final")
