import torch
from torch_geometric.nn import TemporalEncoding, PositionalEncoding


class RoPE1D(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.dim = out_channels

    def forward(self, features: torch.Tensor, timestamps: torch.Tensor):
        """
        Apply 1D Rotary Positional Embedding (RoPE) to an input feature sequence.

        Args:
            features (torch.Tensor): Tensor of shape (seq_len, dim)
            timestamps: (torch.Tensor): Normalized timestamps
        
        Returns:
            torch.Tensor: RoPE-encoded features of the same shape (seq_len, dim).
        """
        assert self.dim % 2 == 0, "RoPE requires even dimension size for pairing."

        # Generate frequencies for RoPE
        theta = 10000 ** (-torch.arange(0, self.dim, 2).float().to(features.device) / self.dim)  # Shape: (dim//2,)
        
        # Normalize timestamps to range [0, 1]
        angles = timestamps.unsqueeze(1) * theta  # Shape: (seq_len, dim//2)

        # Compute sin and cos
        sin_vals = torch.sin(angles)  # Shape: (seq_len, dim//2)
        cos_vals = torch.cos(angles)  # Shape: (seq_len, dim//2)

        # Split feature tensor into even and odd indices for rotation
        features_even = features[:, 0::2]  # Shape: (seq_len, dim//2)
        features_odd = features[:, 1::2]  # Shape: (seq_len, dim//2)

        # Apply RoPE rotation
        rotated_even = features_even * cos_vals - features_odd * sin_vals
        rotated_odd = features_even * sin_vals + features_odd * cos_vals

        # Reconstruct rotated tensor
        encoded_features = torch.zeros_like(features)
        encoded_features[:, 0::2] = rotated_even
        encoded_features[:, 1::2] = rotated_odd

        return encoded_features

load_encoding = {
    'ape': PositionalEncoding,
    'tpe': TemporalEncoding,
    'rpe': RoPE1D,
}
