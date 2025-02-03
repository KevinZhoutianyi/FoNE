import logging  
import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
import math
import numpy as np
import time
import pdb

# Fourier Embedding Class
class FNE(nn.Module):
    def __init__(self, embedding_dim, int_digit_len=5, frac_digit_len=5, period_base_list=[2, 5], add_linear=True, device='cuda'):
        super().__init__()
        print(f'---- \n fourier embedding initing... int_digit_len = {int_digit_len},  frac_digit_len = {frac_digit_len}, period_base_list = {period_base_list}, embedding dim= {embedding_dim}')
        self.add_linear = add_linear
        # Initialize period list for Fourier embedding
        self.period_list = self._get_period_list(period_base_list, minvalue=10**(-frac_digit_len), maxvalue=10**(int_digit_len+1)-1)
        logging.info(f"period_list: {self.period_list}")
        self.period_base_list_len = len(period_base_list)
        # Create and register the precomputed cosine/sine matrix as a buffer
        with torch.no_grad():
            precomputed_cos_sin_matrix = self._create_precomputed_cos_sin_matrix(period_base_list).T
        self.register_buffer("precomputed_cos_sin_matrix", precomputed_cos_sin_matrix.to(device))
        # Precompute the cosine/sine frequencies and register as a buffer
        period_tensor = torch.tensor(self.period_list, dtype=torch.float64, device=device)
        self.register_buffer("period_tensor", period_tensor)
        frequencies = (2 * torch.pi / period_tensor).to(device)
        self.register_buffer("frequencies", frequencies)
        
        # Initialize other attributes
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-5).to(device)
        self.device = device
        
        # Precompute powers of ten for digit manipulation and register as a buffer
        self.max_num_digits = int_digit_len + frac_digit_len
        powers_of_ten = torch.pow(10, torch.arange(self.max_num_digits, device=device)).long()[-(int_digit_len+frac_digit_len):]
        self.register_buffer("powers_of_ten", powers_of_ten)
        self.register_buffer("powers_of_ten_flipped", powers_of_ten.flip(0))
        logging.info(f"self.powers_of_ten: {self.powers_of_ten}")
        

        #for weighted loss
        self.digit_weights = torch.arange(1, self.max_num_digits + 1).to(device)
    def forward(self, number_scatter, len_gen=None):
        """Compute Fourier-based embedding with a linear transformation."""
        fourier_embedding = self.fourier_embedding(number_scatter, len_gen=len_gen)
        # sum_except_last = torch.sum(fourier_embedding[..., :-1], dim=-1, keepdim=True)
        # fourier_embedding[..., -1] = -sum_except_last.squeeze(-1)
        if self.add_linear:
            fourier_embedding = self.linear(fourier_embedding)
        return fourier_embedding

    def fourier_embedding(self, number_scatter, len_gen=None):
        """Apply Fourier embedding transformation to the input number scatter."""
        if len_gen is None:
            len_gen = 0
        # Flatten the number scatter tensor for processing
        flattened_number_scatter = number_scatter.flatten()
        
        # Transform numbers into cos/sin-based embeddings
        number_in_hidden_space = self._turn_numbers_to_cosxsinx(flattened_number_scatter, len_gen)

        # Apply mask to ignore padding (zeros in number_scatter)
        mask = (number_scatter != 0).unsqueeze(-1)
        masked_number_in_hidden_space = number_in_hidden_space.view(*number_scatter.shape, -1) * mask
        return masked_number_in_hidden_space

    def _get_period_list(self,  period_base_list=[2, 5], minvalue=1e-5,maxvalue=1e5):
        """
        Generates a list of periods based on the given period base list, a maximum number,
        and a minimum fraction threshold for the smallest fractional period.
        """
        period_list = set()
        # Generate fractional periods
        current_value = 1.0
        while current_value >= minvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue:
                    period_list.add(period)
            current_value /= 10
        # Generate integer periods
        current_value = 1
        while current_value <= maxvalue:
            for base in period_base_list:
                period = current_value * base
                if period <= maxvalue:
                    period_list.add(period)
            current_value *= 10
        return sorted(period_list)

    def _turn_numbers_to_cosxsinx(self, numbers, len_gen):
        """
        Use precomputed frequencies to turn numbers into cos/sin embeddings.
        """
        # Ensure numbers are on the correct device and in torch.float64
        numbers = numbers.to(self.device, dtype=torch.float64)*(10**len_gen)
        # Calculate cos and sin values with broadcasting
        cos_values = torch.cos(numbers.unsqueeze(1) * self.frequencies)
        sin_values = torch.sin(numbers.unsqueeze(1) * self.frequencies)
        # Concatenate cos and sin values interleaved
        # Stack cos and sin values along a new dimension
        cos_sin_stacked = torch.stack((cos_values, sin_values), dim=-1)

        # Flatten the last two dimensions to interleave cos and sin
        cos_sin_interleaved = cos_sin_stacked.view(cos_values.size(0), -1)

        # Initialize the result tensor and populate it
        result = torch.zeros((len(numbers), self.embedding_dim), dtype=torch.float32, device=self.device)
        result[:, :cos_sin_interleaved.shape[1]] = cos_sin_interleaved

        return result

    def _create_precomputed_cos_sin_matrix(self, period_base_list=[2, 5]):
        """
        Creates a precomputed cos/sin matrix for the given period base list and number of positions.
        """
        # Convert string periods to floats if necessary
        period_base_list = [base if type(base) != str else eval(base) for base in period_base_list]
        num_positions = 10  # Modify as needed for desired number of positions
        base_positions = torch.arange(num_positions)
        cos_sin_list = []
        # Compute cos and sin values for each period
        for period in period_base_list:
            w = 2 * torch.pi / period
            cos_sin_list.append(torch.cos(w * base_positions))
            cos_sin_list.append(torch.sin(w * base_positions))
        
        # Stack all values to form a matrix
        cos_sin_matrix = torch.stack(cos_sin_list, dim=1)
        return cos_sin_matrix

    def fourier_compute_loss(self, before_decoder, label, int_digit_len, frac_digit_len, len_gen=None):
        """
        Compute loss using cross-entropy for the predicted Fourier embeddings.
        """
        if len_gen is None:
            # As a fallback, you could default to zeros
            len_gen = 0

        before_decoder = self.layer_norm(before_decoder)
        num_digits = int_digit_len + frac_digit_len
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0]
        start_idx = 2*self.period_base_list_len*len_gen
        end_idx = 2*self.period_base_list_len*len_gen + (num_digits * num_cos_sin_per_digit)        

        # Reshape to match digits and cos/sin pairs
        slices_all_digits = before_decoder[..., start_idx:end_idx]#bs, 10*2
        slices_all_digits = slices_all_digits.view(-1, num_digits, num_cos_sin_per_digit) #bs,10,4

        # Compute logits
        logits_all_digits = torch.matmul(slices_all_digits, self.precomputed_cos_sin_matrix) #bs,10,4 * 4*10
        
        # Use precomputed powers of ten
        powers_of_ten = self.powers_of_ten  # Select required powers
        # Scale labels and extract digits
        
        scaled_labels = (torch.round(label * (10 ** frac_digit_len))).long()
        digit_labels = (scaled_labels.unsqueeze(1) // powers_of_ten) % 10

        loss_per_digit = F.cross_entropy(
            logits_all_digits.view(-1, 10),  # Flatten to [batch_size * num_digits, 10]
            digit_labels.view(-1),           # Flatten to [batch_size * num_digits]
            reduction='none'                 # Compute individual losses
        )  # [batch_size * num_digits]

        # Repeat weights for batch size
        # digit_weights = self.digit_weights.repeat(loss_per_digit.size(0) // self.max_num_digits)  # Match shape with loss_per_digit

        # Apply weights to the loss per digit
        weighted_loss_per_digit = loss_per_digit #* digit_weights  # Element-wise multiplication
     
        # Compute the total loss
        total_loss = weighted_loss_per_digit.sum() / loss_per_digit.size(0)  # Normalize by the total number of samples

        return total_loss
    
    
    def fourier_compute_prediction(self, before_decoder, int_digit_len, frac_digit_len):
        """
        Predict numbers by extracting the most probable digit using argmax.
        """
        before_decoder = self.layer_norm(before_decoder)
        batch_size = before_decoder.shape[0]
        num_cos_sin_per_digit = self.precomputed_cos_sin_matrix.shape[0]
        num_digits = int_digit_len + frac_digit_len
        predicted_number = torch.zeros(batch_size, dtype=torch.float64).to(before_decoder.device)
        # Add fractional part
        for i in range(frac_digit_len):
            start_idx = i * num_cos_sin_per_digit
            end_idx = start_idx + num_cos_sin_per_digit
            slice_i = before_decoder[..., start_idx:end_idx]
            logits_i = torch.matmul(slice_i, self.precomputed_cos_sin_matrix)
            predicted_digit = torch.argmax(logits_i, dim=1).to(torch.float64)
            predicted_number += predicted_digit * (10 ** -(frac_digit_len - i))
        # Add integer part
        for j in range(int_digit_len):
            start_idx = (frac_digit_len + j) * num_cos_sin_per_digit
            end_idx = start_idx + num_cos_sin_per_digit
            slice_j = before_decoder[..., start_idx:end_idx]
            logits_j = torch.matmul(slice_j, self.precomputed_cos_sin_matrix)
            predicted_digit = torch.argmax(logits_j, dim=1).to(torch.float64)
            predicted_number += predicted_digit * (10 ** j)
        
        return predicted_number
import torch.optim as optim

def main():
    # Set up parameters for testing
    embedding_dim = 1600
    period_base_list = [2, 5]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 200
    int_digit_len, frac_digit_len = 5, 5  # Adjusted to match max_number

    # Instantiate FNE model
    model = FNE(embedding_dim, int_digit_len, frac_digit_len, period_base_list, device).to(device)

    # Generate random data for testing
    max_int_part = 10 ** int_digit_len - 1
    max_frac_part = 10 ** frac_digit_len - 1
    int_part = torch.randint(low=0, high=max_int_part + 1, size=(batch_size,)).float()
    frac_part = torch.randint(low=0, high=max_frac_part + 1, size=(batch_size,)).float()
    frac_part = frac_part / (10 ** frac_digit_len)
    label = (int_part + frac_part).to(device)

    # Generate 'before_decoder' tensor for loss and prediction testing
    before_decoder = torch.randn(batch_size, embedding_dim).to(device)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Forward pass (loss computation) timing
    forward_start_time = time.time()
    total_loss = model.fourier_compute_loss(before_decoder, label, int_digit_len, frac_digit_len)
    forward_time = time.time() - forward_start_time
    print(f"Forward pass (loss computation) took: {forward_time:.4f} seconds")

    # Backward pass timing
    backward_start_time = time.time()
    total_loss.backward()  # Compute gradients
    backward_time = time.time() - backward_start_time
    print(f"Backward pass took: {backward_time:.4f} seconds")

    # Print total time for forward and backward
    print(f"Total forward and backward pass time: {forward_time + backward_time:.4f} seconds")

if __name__ == "__main__":
    main()
