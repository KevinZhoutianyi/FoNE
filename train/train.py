import torch
import logging
from train.utils import get_regular_embeddings, handle_nan_loss

def train_fne(model, train_loader, fne, optimizer, scheduler, args, int_digit_len, frac_digit_len, len_gen_size, device):
    """
    Training loop for Fourier Neural Embedding (FNE) based models.
    """
    model.train()
    fne.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        len_gen = torch.randint(0, len_gen_size+1, (1,), device=device).item()
        
        regular_embeddings = get_regular_embeddings(model, input_ids)
        fourier_embeddings = fne(scatter_tensor, len_gen=len_gen)
        input_embeddings = regular_embeddings + fourier_embeddings

        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        before_decoder = outputs.hidden_states[-1]
        last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = fne.fourier_compute_loss(last_token_hidden_state, labels, int_digit_len, frac_digit_len, len_gen=len_gen)
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(fne.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_regular(model, dataloader, optimizer, scheduler, device, args):
    """
    Regular training loop for models without additional embedding modules.
    """
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()

        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)

def train_xval(model, train_loader, xval, optimizer, scheduler, args, device):
    """
    Training loop for models using the xval module.
    """
    model.train()
    xval.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)

        regular_embeddings = get_regular_embeddings(model, input_ids)
        input_embeddings = xval(scatter_tensor, regular_embeddings)
        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        before_decoder = outputs.hidden_states[-1]
        last_token_hidden_state = (before_decoder * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = xval.compute_loss(last_token_hidden_state, labels)

        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(xval.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    logging.info(f"avg Loss: {total_loss / len(train_loader)}")
    return total_loss / len(train_loader)

def train_vanilla(model, train_loader, vanilla_model, optimizer, scheduler, args, device):
    """
    Training loop for models using a vanilla embedding module.
    """
    model.train()
    vanilla_model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        scatter_tensor = batch['scatter_tensor'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        last_token_mask = batch['last_token_mask'].to(device)
        
        regular_embeddings = get_regular_embeddings(model, input_ids)
        vanilla_embeddings = vanilla_model(scatter_tensor)
        input_embeddings = regular_embeddings + vanilla_embeddings

        outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden_state = (last_hidden_state * last_token_mask.unsqueeze(-1)).sum(dim=1)

        loss = vanilla_model.compute_loss(last_token_hidden_state, labels)
        
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vanilla_model.parameters(), max_norm=1.0)
            
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    logging.info(f"Training Loss: {avg_loss}")
    return avg_loss
