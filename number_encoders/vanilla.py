import torch
import torch.nn as nn

class VanillaEmbedding(nn.Module):
    def __init__(self, embedding_dim=1600, int_digit_len=5, frac_digit_len=5, device='cuda'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.int_digit_len = int_digit_len
        self.frac_digit_len = frac_digit_len
        self.max_num_digits = int_digit_len + frac_digit_len
        self.device = device

        # Precompute powers of ten for digit extraction
        self.powers_of_ten = torch.pow(10, torch.arange(self.max_num_digits, device=device)).long()
        self.powers_of_ten_flipped = self.powers_of_ten

    def forward(self, number_scatter):
        """Compute vanilla embedding for the input number scatter."""
        return self.vanilla_embedding(number_scatter)

    def vanilla_embedding(self, number_scatter):
        """Convert numbers into vanilla embeddings, padded to embedding_dim."""
        # Flatten the input tensor
        flattened_number_scatter = number_scatter.flatten()

        # Convert numbers to vanilla embeddings
        vanilla_embeddings = self._turn_numbers_to_vanilla(flattened_number_scatter)

        # Pad embeddings with zeros to match embedding_dim
        padded_embeddings = torch.zeros((len(flattened_number_scatter), self.embedding_dim), device=self.device)
        padded_embeddings[:, :self.max_num_digits] = vanilla_embeddings

        # Reshape to match the original input shape
        padded_embeddings = padded_embeddings.view(*number_scatter.shape, self.embedding_dim)

        # Apply mask to ignore padding (zeros in number_scatter)
        mask = (number_scatter != 0).unsqueeze(-1)
        masked_padded_embeddings = padded_embeddings * mask

        return masked_padded_embeddings

    def _turn_numbers_to_vanilla(self, numbers):
        """Convert numbers into vanilla embeddings by extracting digits."""
        # Scale numbers to extract all digits
        scaled_numbers = (numbers * (10 ** self.frac_digit_len)).long()

        # Extract digits using powers of ten
        digits = (scaled_numbers.unsqueeze(1) // self.powers_of_ten) % 10

        # Convert digits to float32 for embedding
        vanilla_embeddings = digits.float()

        return vanilla_embeddings

    def compute_loss(self, last_hidden_state, label):
        """Compute RMS loss between each digit entry and label's digits."""
        # Convert labels to digit embeddings [batch_size, max_num_digits]
        labels_digits = self._turn_numbers_to_vanilla(label).squeeze(1)  # Remove unnecessary dimension
        
        # Get predicted digits from first max_num_digits entries
        predicted_digits = last_hidden_state[:, :self.max_num_digits]
        
        # Calculate RMS loss
        squared_errors = (predicted_digits - labels_digits) ** 2
        mean_squared_error = torch.mean(squared_errors)
        rms_loss = torch.sqrt(mean_squared_error)
        
        return rms_loss

    def compute_prediction(self, last_hidden_state):
        """Predict numbers by taking the i-th entry of the last hidden state as the i-th digit."""
        # Extract digits from the first max_num_digits entries of the last hidden state
        predicted_digits = torch.round(last_hidden_state[:, :self.max_num_digits]).long() % 10

        # Combine digits to form the predicted number
        predicted_number = torch.sum(predicted_digits * self.powers_of_ten_flipped, dim=-1) / (10 ** self.frac_digit_len)

        return predicted_number


# Example Usage
if __name__ == "__main__":
    # Set up parameters
    embedding_dim = 1600
    int_digit_len, frac_digit_len = 5, 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate the model
    model = VanillaEmbedding(embedding_dim, int_digit_len, frac_digit_len, device).to(device)

    # Generate random data for testing
    batch_size = 10
    max_int_part = 10 ** int_digit_len - 1
    max_frac_part = 10 ** frac_digit_len - 1
    int_part = torch.randint(low=0, high=max_int_part + 1, size=(batch_size,)).float()
    frac_part = torch.randint(low=0, high=max_frac_part + 1, size=(batch_size,)).float()
    frac_part = frac_part / (10 ** frac_digit_len)
    label = (int_part + frac_part).to(device)

    # Generate last hidden state for testing
    last_hidden_state = torch.randn(batch_size, embedding_dim).to(device)

    # Compute loss
    loss = model.compute_loss(last_hidden_state, label)
    print(f"Loss: {loss.item()}")

    # Compute prediction
    predicted_number = model.compute_prediction(last_hidden_state)
    print(f"Predicted Number: {predicted_number}")