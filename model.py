import torch
from torch import nn
from torch.nn.parameter import Parameter
from utils import xavier_uniform
import math

class ScribeLM(nn.Module):
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
        assert self.d_model%h==0
        self.d_k = self.d_v = d_model//h

        self.embedding =  nn.Embedding(self.vocab_size, self.d_model, sparse=True)

        self.positional_encodings = torch.tile(torch.arange(self.d_model, dtype=torch.float32, device=next(self.parameters()).device), (self.context_length, 1))
        for pos in range(self.context_length):
            self.positional_encodings[pos, ::2] = torch.sin((pos/10_000)**(2*self.positional_encodings[pos, ::2]/self.d_model))
            self.positional_encodings[pos, 1::2] = torch.cos((pos/10_000)**(2*self.positional_encodings[pos, 1::2]/self.d_model))

        self.w_q = Parameter(xavier_uniform((self.n, self.h, self.d_model, self.d_k), self.d_model, self.d_k))
        self.w_k =  Parameter(xavier_uniform((self.n, self.h, self.d_model, self.d_k), self.d_model, self.d_k))
        self.w_v =  Parameter(xavier_uniform((self.n, self.h, self.d_model, self.d_v), self.d_model, self.d_v))
        self.w_o =  Parameter(xavier_uniform((self.n, self.h*self.d_v, self.d_model), self.h*self.d_v, self.d_model))
        self.ff_1 =  Parameter(xavier_uniform((self.n, self.d_model, self.d_ff), self.d_model, self.d_ff))
        self.ff_2 =  Parameter(xavier_uniform((self.n, self.d_ff, self.d_model), self.d_ff, self.d_model))
        self.b_1 =  Parameter(torch.zeros((self.n, self.d_ff), requires_grad=True))
        self.b_2 =  Parameter(torch.zeros((self.n, self.d_model), requires_grad=True))
        self.layernorm_1 = list([nn.LayerNorm(self.d_model) for _ in range(self.n)])
        self.layernorm_2 = list([nn.LayerNorm(self.d_model) for _ in range(self.n)])
    def forward(self, x: torch.Tensor):
        tokens = x.detach().clone()

        assert tokens.dim() <= 2

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        tokens = tokens[:, :self.context_length]
        batch_size = tokens.shape[0]
        context = tokens.shape[-1]
        assert tokens.shape == (batch_size, context)

        embedded = self.embedding(tokens)
        assert embedded.shape == (batch_size, context, self.d_model)
        embedded = torch.add(embedded, self.positional_encodings[:context])

        for layer in range(self.n):
            tiled_embedded = embedded.unsqueeze(1).repeat(1, self.h, 1, 1)
            assert tiled_embedded.shape == (batch_size, self.h, context, self.d_model)
            q = tiled_embedded @ self.w_q[layer, :, :, :].unsqueeze(0)
            k = tiled_embedded @ self.w_k[layer, :, :, :].unsqueeze(0)
            v = tiled_embedded @ self.w_v[layer, :, :, :].unsqueeze(0)
            assert q.shape == (batch_size, self.h, context, self.d_k)
            mask = torch.triu(torch.ones((1, 1, context, context)), diagonal=1).bool().to(next(self.parameters()).device)
            y = nn.functional.softmax(((q @ k.transpose(-1, -2))/math.sqrt(self.d_k)).masked_fill(mask, -torch.inf), dim=-1) @ v
            assert y.shape == (batch_size, self.h, context, self.d_v)
            y = torch.reshape(y.transpose(-2, -3), (batch_size, context, -1))
            assert y.shape == (batch_size, context, self.d_v*self.h)
            embedded = self.layernorm_1[layer](embedded + y @ self.w_o[layer, :, :].unsqueeze(0))
            assert embedded.shape == (batch_size, context, self.d_model)
            y = torch.maximum(torch.tensor(0), embedded @ self.ff_1[layer, :].unsqueeze(0) + self.b_1[layer, :])
            embedded = self.layernorm_2[layer](embedded + y @ self.ff_2[layer, :].unsqueeze(0) + self.b_2[layer, :])
            assert embedded.shape == (batch_size, context, self.d_model)
        return embedded @ self.embedding.weight.T.unsqueeze(0)