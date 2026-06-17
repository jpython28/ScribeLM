# ScribeLM: A Context Length Ablation

## Abstract

ScribeLM is a decoder-only transformer with about 65 million parameters, making it a very small model. It is implemented from scratch and trained on WikiText-103. This report studies the effect of context length on perplexity. All other variables, such as architecture, optimization, and training duration were held constant. Perplexity was found to decrease as context length increases, but with diminishing returns at higher context lengths.

## Background

Transformers use self-attention, a mechanism that lets tokens within a certain context window "attend" to each other, enriching individual tokens' meaning. The larger the context window, the more the model can take advantage of long-range context. This report measures that benefit with a small implementation, rather than assuming it from theory alone.

## Methodology

**Model**: Many choices regarding model architecture were influenced by "Attention Is All You Need". ScribeLM uses manually implemented multi-head attention instead of `nn.Transformer` or `nn.MultiheadAttention` for the purpose of better understanding the transformer architecture. It uses post-normalization, sinusoidal positional encodings, causal masking, and weight tying between embedding and unembedding layers. All weights use Xavier initialization, excluding the embedding matrix, which was initialized with a standard normal distribution. Interestingly, when Xavier initialization was used for embeddings, the model plateaued within a few hundred steps. This happened because when Xavier initialization is applied to a large matrix, it creates very small row magnitudes. The embedding weights are tied to the unembedding weights, which means the output projection creates small dot products, decreasing the model's ability to make a confident prediction. These very small weights cause a plateau because embedding rows only get an update when their corresponding token appears in a batch, which means it would take many updates to grow the magnitude of every row.

**Gradient Clipping**: In single-epoch testing, the training and validation showed large and seemingly random loss spikes. These were assumed to be outlier batches causing gradients to grow too large, and gradient clipping was implemented to prevent this.

**Dataset**: WikiText-103, flattened and tokenized with GPT-2's BPE tokenizer, and then divided into non-overlapping chunks. A training, validation, and testing split were used. Training for gradient updates, validation for diagnostics throughout training, and testing for a final performance number, evaluated after training is complete.

**Ablation Design**: The ablation involved training four configurations, with different context lengths (64, 128, 256, and 512), but the same batch size, optimizer (AdamW), learning rate, and random seed. The configurations were trained for the same number of steps (20,000), as opposed to the same number of epochs. This is because non-overlapping chunks means that the number of chunks in an epoch is approximately `total_tokens/context_length`. This means that one epoch with a high context length gives the model less optimizer steps than a model with a lower context length.

## Results

| Context Length | Test Perplexity |
|---|---|
| 64  | 390.9 |
| 128 | 314.1 |
| 256 | 278.6 |
| 512 | 256.4 |

!["validation perplexity graph"](validation_perplexity_graph.jpg)

Perplexity decreased most when context length was doubled from 64 to 128, causing a ~77 point reduction. Further increasing context length led to diminishing returns. Doubling from 256 to 512 only caused a ~22 point reduction. This data would suggest that the model gets the most useful context from closer tokens, but longer-range context contributes smaller, but not insignificant, gains.

All four of the runs were still improving at the 20,000 step cutoff, which confirms that none of the models had finished learning (plateaued), making the ablation a fair comparison.

## Limitations

 - Results were not averaged across multiple seeds due to limited access to cloud GPUs.
 - No learning rate warmup or decay was implemented.
 - Due to non-overlapping chunks, the 256 and 512 token context length configurations completed multiple epochs in 20,000 steps, while the 64 and 128 runs did not. However, the validation curves show no sign of overfitting.

## Future Work
 - Implement learning rate warmup and decay
 - Conduct another ablation on the number of attention heads or the number of layers
 - Implement learned positional embeddings
 - Compare the performance of tied and untied embeddings
 - Find out if there is a point where larger context length becomes harmful to performance