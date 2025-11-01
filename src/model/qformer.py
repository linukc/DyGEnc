import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class QFormerLayer(nn.Module):
    """
    A single Q-Former block:
      1) Self-attention among queries (fixed length, no padding needed).
      2) Cross-attention (queries attend to 'features', which may be padded).
      3) Feed-forward, residuals, etc.
    """
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        
        # We use MultiheadAttention with batch_first=True
        self.self_attn = MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, features, features_mask=None):
        """
        queries:        (B, num_query_tokens, hidden_dim) - fixed-length, no pad
        features:       (B, seq_len, hidden_dim)          - potentially padded
        features_mask:  (B, seq_len)  True=valid, False=pad 
        """

        # 1) Self-Attention among the queries
        q_norm = self.norm1(queries)
        attn_out, _ = self.self_attn(
            q_norm, q_norm, q_norm  # Q, K, V
        )
        queries = queries + attn_out  # residual

        # 2) Cross-Attention (queries attend to features)
        # For MultiheadAttention with batch_first=True:
        #   key_padding_mask expects shape (B, seq_len) with True where we want to ignore
        # Our features_mask is True=valid, so we invert it:
        key_padding_mask = None
        if features_mask is not None:
            key_padding_mask = ~features_mask  # now True=ignore

        q_norm = self.norm2(queries)
        cross_out, _ = self.cross_attn(
            q_norm,                 # queries as Q
            features,               # K
            features,               # V
            key_padding_mask=key_padding_mask
        )
        queries = queries + cross_out  # residual

        # 3) Feed-Forward
        q_norm = self.norm3(queries)
        ff_out = self.ff(q_norm)
        queries = queries + ff_out  # residual

        return queries


class QFormer(nn.Module):
    """
    A multi-layer Q-Former that produces updated query embeddings from
    (possibly padded) features.
    """
    def __init__(
        self,
        num_query_tokens=8,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    ):
        super().__init__()
        
        # Learned query embeddings
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))

        # Stack of Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, features, features_mask=None):
        """
        features:       (B, seq_len, hidden_dim)
        features_mask:  (B, seq_len) True=valid, False=pad
        Returns:
            queries: (B, num_query_tokens, hidden_dim)
        """
        B = features.size(0)
        
        # Expand the learned queries for the batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, num_query_tokens, hidden_dim)

        # Pass queries + features through each Q-Former layer
        for layer in self.layers:
            queries = layer(queries, features, features_mask=features_mask)

        return queries
