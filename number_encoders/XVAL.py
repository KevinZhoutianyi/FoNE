import logging  
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
import pdb

# Fourier Embedding Class
class XVAL(nn.Module):
    def __init__(self, embedding_dim, max_num=1000, device='cuda'):
        """
        Args:
            embedding_dim: Dimension of the word embeddings.
            max_num: Maximum value of numbers in the dataset (used for scaling).
            device: Device to use ('cuda' or 'cpu').
        """
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.max_num = max_num  # Maximum number for scaling
        self.scale_factor = 5 / math.log(1 + max_num)  # Scaling factor for range [-5, 5]
        
        # Linear layer to predict numbers based on embeddings
        self.num_head = nn.Linear(embedding_dim, 1).to(device)
        
    def forward(self, number_scatter, word_embedding):
        """
        Compute Fourier-based embedding with a linear transformation.

        Args:
            number_scatter: Tensor containing the numbers to be embedded (batch_size, seq_len).
            word_embedding: Word embedding tensor (batch_size, seq_len, embedding_dim).
        
        Returns:
            Scaled embeddings.
        """
        # Ensure both tensors are in float32
        number_scatter = number_scatter.to(torch.float32)
        word_embedding = word_embedding.to(torch.float32)
        # Scale numbers to range [-5, 5]
        scaled_numbers = self.scale_factor * torch.log(1 + number_scatter.abs()) * torch.sign(number_scatter)

        # Create the mask and expand number_scatter
        mask = (scaled_numbers != 0).unsqueeze(-1)
        scaled_numbers_expanded = scaled_numbers.unsqueeze(-1).expand_as(word_embedding)

        # Use entry-wise product where scatter != 0, otherwise use the original embedding
        result = torch.where(mask, scaled_numbers_expanded * word_embedding, word_embedding)
        return result

    def compute_loss(self, before_decoder, label):
        """
        Compute the mean squared error loss for number predictions.

        Args:
            before_decoder: Input embeddings to the num_head layer 
                            (batch_size, seq_len, embedding_dim).
            label: True numbers (batch_size, seq_len).
        
        Returns:
            loss_num: Computed MSE loss.
        """
        # Scale labels to range [-5, 5]
        label = label.to(torch.float32)
        scaled_label = self.scale_factor * torch.log(1 + label.abs()) * torch.sign(label)
        
        num_preds = self.num_head(before_decoder)  # Shape: (batch_size, seq_len, 1)
        # Compute MSE loss
        loss_num = F.mse_loss(num_preds.squeeze(-1), scaled_label, reduction="mean")
  
        return loss_num

    def compute_prediction(self, before_decoder):
        """
        Predict numbers by extracting the most probable digit using argmax.

        Args:
            before_decoder: Input embeddings to the num_head layer 
                            (batch_size, seq_len, embedding_dim).
        
        Returns:
            predictions: Predicted numbers as integers (batch_size, seq_len).
        """
        # Pass through the linear layer to compute logits for each digit
        predictions = self.num_head(before_decoder).squeeze(-1)
        
        # Scale back predictions to the original number range
        predictions = (torch.exp(predictions.abs() / self.scale_factor) - 1) * torch.sign(predictions)
        return predictions
