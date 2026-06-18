"""Contains class ScribeLM, a transformer-based language model with multiheaded attention"""

import torch
from torch import nn
from torch.nn.parameter import Parameter
import math

class ScribeLM(nn.Module):
    """Decoder-only transformer (LLM) with multiheaded attention.

    Implemented from scratch without nn.MultiheadAttention or any of pytorch's nn.Transformer layers,
    to study the transformer architecture.
    Implements post-normalization, sinusoidal positional encodings, and causal masking

    Args:
        context_length (int): Maximum size of context window
        vocab_size (int): Size of model's token vocabulary
        n (int): Number of attention layers
        d_model (int): Embedding dimension size
        d_ff (int): Feedforward layer hidden dimension size
        h (int): Number of attention heads per layer
    """
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 n: int=6,
                 d_model: int=512,
                 d_ff: int=2048,
                 h: int=8,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.n = n
        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.d_k = self.d_v = d_model//h

        assert self.d_model%h==0, "d_model must be divisible by h (number of heads)"
        
        # Uses same sinusoidal (not learned) positional encodings as the paper Attention is All You Need:
        # PE(pos, 2i) = sin(pos/10000^(2i/dmodel))
        # PE(pos, 2i+1) = cos(pos/10000^(2i/dmodel))
        even_indices = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        self.register_buffer("positional_encodings", torch.zeros(self.context_length, self.d_model))
        for pos in range(self.context_length):
            self.positional_encodings[pos, ::2] = torch.sin(pos / 10_000 ** (even_indices / self.d_model))
            self.positional_encodings[pos, 1::2] = torch.cos(pos / 10_000 ** (even_indices / self.d_model))

        self.register_buffer("attn_mask", torch.triu(torch.ones((1, 1, self.context_length, self.context_length)), diagonal=1).bool())

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        #nn.init.xavier_uniform_(self.embedding.weight)
        
        self.w_q = Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.h, self.d_model, self.d_k)))
        self.w_k =  Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.h, self.d_model, self.d_k)))
        self.w_v =  Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.h, self.d_model, self.d_v)))
        self.w_o =  Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.h*self.d_v, self.d_model)))
        self.ff_1 =  Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.d_model, self.d_ff)))
        self.ff_2 =  Parameter(nn.init.xavier_uniform_(torch.empty(self.n, self.d_ff, self.d_model)))
        self.b_1 =  Parameter(torch.zeros((self.n, self.d_ff), requires_grad=True))
        self.b_2 =  Parameter(torch.zeros((self.n, self.d_model), requires_grad=True))
        self.layernorm_1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n)])
        self.layernorm_2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n)])

    def forward(self, input_tokens: torch.Tensor):
        tokens = input_tokens

        assert tokens.dim() <= 2, "input_tokens should be shape (batch, seq) or (seq,)"

        # Add batch dimension if input is unbatched
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        tokens = tokens[:, -self.context_length:]

        batch_size = tokens.shape[0]
        seq_len = tokens.shape[-1]

        hidden_state = self.embedding(tokens)
        assert hidden_state.shape == (batch_size, seq_len, self.d_model)

        hidden_state = torch.add(hidden_state, self.positional_encodings[:seq_len])

        for layer_idx in range(self.n):
            hidden_per_head = hidden_state.unsqueeze(1).repeat(1, self.h, 1, 1)
            assert hidden_per_head.shape == (batch_size, self.h, seq_len, self.d_model)

            q = hidden_per_head @ self.w_q[layer_idx, :, :, :].unsqueeze(0)
            k = hidden_per_head @ self.w_k[layer_idx, :, :, :].unsqueeze(0)
            v = hidden_per_head @ self.w_v[layer_idx, :, :, :].unsqueeze(0)
            assert q.shape == (batch_size, self.h, seq_len, self.d_k)
            
            # Use causal mask to prevent later tokens from attending to ealier ones
            # Mask set to -inf because when passed through softmax -inf becomes 0
            # Divide q @ k.T by sqrt(d_k) to prevent very large dot products
            attn_out = nn.functional.softmax(((q @ k.transpose(-1, -2))/math.sqrt(self.d_k)).masked_fill(self.attn_mask[:, :, :seq_len, :seq_len], -torch.inf), dim=-1) @ v
            assert attn_out.shape == (batch_size, self.h, seq_len, self.d_v)
            
            attn_out = torch.reshape(attn_out.transpose(-2, -3), (batch_size, seq_len, -1))
            assert attn_out.shape == (batch_size, seq_len, self.d_v*self.h)

            hidden_state = self.layernorm_1[layer_idx](hidden_state + attn_out @ self.w_o[layer_idx, :, :].unsqueeze(0))
            assert hidden_state.shape == (batch_size, seq_len, self.d_model)

            attn_out = torch.nn.functional.relu(hidden_state @ self.ff_1[layer_idx, :].unsqueeze(0) + self.b_1[layer_idx, :])

            hidden_state = self.layernorm_2[layer_idx](hidden_state + attn_out @ self.ff_2[layer_idx, :].unsqueeze(0) + self.b_2[layer_idx, :])
            assert hidden_state.shape == (batch_size, seq_len, self.d_model)
        
        # Use the same weights as embedding matrix for unembedding to reduce parameter count and improve performance
        return hidden_state @ self.embedding.weight.T.unsqueeze(0)