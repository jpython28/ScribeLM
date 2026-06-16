## Methodology

**Model**: ScribeLM uses manually implemented multi-head attention instead of `nn.Transformer` or `nn.MultiheadAttention` for the purpose of better understanding the transformer architecture. It uses post-normalization, sinusoidal positional encodings, causal masking, and weight tying between embedding and unembedding layers.

**Dataset**: WikiText-103, flattened and tokenized with GPT-2's BPE tokenizer, and then split into non-overlapping chunks.

**Ablation Design**: The ablation involved training four configurations, with different context lengths (64, 128, 256, and 512), but the same batch size, optimizer (AdamW), learning rate, and random seed. The configurations were trained for the same number of steps (20,000), as opposed to the same number of epochs. This is because non-overlapping chunks means that the number of chunks in an epoch is approximately `total_tokens/context_length`. This means that one epoch with a high context length gives the model less optimizer steps than a model with a lower context length.