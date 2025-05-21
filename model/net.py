import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import distributions as pyd

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device: torch.device = None) -> torch.Tensor:
    """
    Precompute frequency bands for rotary position embeddings.
    
    Args:
        dim (int): Dimension of the embeddings
        seq_len (int): Maximum sequence length
        theta (float): Base for frequency computation
        device (torch.device): Device to compute on
        
    Returns:
        torch.Tensor: Complex tensor of shape (seq_len, dim//2)
    """
    # Compute frequency bands
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)).to(device)
    
    # Generate sequence indices
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs).float().to(device)
    
    # Convert to polar form
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(device)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    
    Args:
        xq (torch.Tensor): Query tensor of shape (batch_size, seq_len, dim)
        xk (torch.Tensor): Key tensor of shape (batch_size, seq_len, dim)
        freqs_cis (torch.Tensor): Precomputed frequency bands
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors
    """
    # Reshape for complex multiplication
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Convert to complex
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # Apply rotation and convert back to real
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class MaskedCausalAttention(nn.Module):
    """
    Masked causal attention module with optional rotary position embeddings.
    """
    def __init__(
        self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        num_inputs: int,
        use_rope: bool = False,
        mgdt: bool = False,
        dt_mask: bool = False,
        att_mask: torch.Tensor = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.max_T = max_T
        self.num_inputs = num_inputs
        self.device = device
        
        # Attention layers
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)
        
        # Dropout layers
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        
        # Rotary position embeddings
        self.use_rope = use_rope
        if use_rope:
            self.freqs_cis = precompute_freqs_cis(h_dim, max_T, device=device)
        
        # Attention mask
        if att_mask is not None:
            mask = att_mask
        else:
            ones = torch.ones((max_T, max_T), device=device)
            mask = torch.tril(ones).view(1, 1, max_T, max_T)
            if mgdt and not dt_mask:
                period = num_inputs
                ret_order = 2
                ret_masked_rows = torch.arange(
                    period + ret_order-1, max_T, period,
                    device=device
                ).long()
                mask[:, :, :, ret_masked_rows] = 0
        
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, h_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, h_dim)
        """
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        
        # Compute query, key, value
        if not self.use_rope:
            q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
            k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
            v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        else:
            q = self.q_net(x)
            k = self.k_net(x)
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis)
            q = q.view(B, T, N, D).transpose(1, 2)
            k = k.view(B, T, N, D).transpose(1, 2)
            v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        
        # Compute attention weights
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float("-inf"))
        normalized_weights = F.softmax(weights, dim=-1)
        
        # Apply attention
        attention = self.att_drop(normalized_weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        
        return self.proj_drop(self.proj_net(attention))

class TransformerBlock(nn.Module):
    """
    Transformer block with masked causal attention and MLP.
    """
    def __init__(
        self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        num_inputs: int,
        use_rope: bool = False,
        mgdt: bool = False,
        dt_mask: bool = False,
        att_mask: torch.Tensor = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        
        # Attention layer
        self.attention = MaskedCausalAttention(
            h_dim=h_dim,
            max_T=max_T,
            n_heads=n_heads,
            drop_p=drop_p,
            num_inputs=num_inputs,
            use_rope=use_rope,
            mgdt=mgdt,
            dt_mask=dt_mask,
            att_mask=att_mask,
            device=device,
        )
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, h_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, h_dim)
        """
        # Attention block
        x = x + self.attention(x)
        x = self.ln1(x)
        
        # MLP block
        x = x + self.mlp(x)
        x = self.ln2(x)
        
        return x

class BaseActor(nn.Module):
    """
    Base actor network with Gaussian policy.
    """
    def __init__(self, hidden_dim: int, act_dim: int):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, obs: torch.Tensor) -> pyd.Normal:
        """
        Forward pass of the actor network.
        
        Args:
            obs (torch.Tensor): Observation tensor
            
        Returns:
            pyd.Normal: Gaussian distribution with mean and std
        """
        mu = torch.tanh(self.mu(obs))
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return pyd.Normal(mu, std)