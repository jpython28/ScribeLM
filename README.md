# ScribeLM

ScribeLM is a decoder-only transformer model, implemented from scratch in PyTorch. It was trained on WikiText-103.

## Model Architecture

Many architectural decisions were based on "Attention Is All you Need".
 - ~65M parameters
 - Decoder-only transformer
 - Does not use `nn.MultiheadAttention` or `nn.Transformer`, for the purpose of getting a better understanding of the transformer architecture
 - Post-normalization: layer norm is applied after residual sums, as opposed to pre-norm, where layer norm is applied before attention and feed-forward networks
 - Sinusoidal positional encodings
 - Causal masking: prevents later tokens from attending to earlier ones
 - Weight tying (using the same weight matrix) for input embedding and output unembedding
 - Pre-trained GPT-2 tokenizer (vocab size 50,257)

## Repository Structure

 - `model.py` - model architecture
 - `data.py` - `torch.utils.Dataset` to download, tokenize, and chunk WikiText-103
 - `train.py` - training, evaluation, and logging (to wandb)
 - `configs/` - config files for each run of the ablation
 - `colab.ipynb` - jupyter notebook to run training on Colab
 - `REPORT.md` - full report of experimental proccess and findings

## Setup

```
pip install -r requirements.txt
```
Training is intended for and tested on Google Colab with an A100 GPU. Colab offers an option to load a notebook from Github. Load `colab.ipynb` and run a config:
```
python train.py --config configs/context_512.yaml
```

## Results: Context Length Ablation

Four configurations were trained with varying context length. Architecture, optimization, batch_size, learning rate, and random seed were held constant. The configurations were trained for the same number of steps, as opposed to the same number of epochs. This is because non-overlapping chunks means that the number of chunks in an epoch is approximately `total_tokens/context_length`. This means that one epoch with a high context length gives the model less optimizer steps than with a lower context length.

| Context Length | Final Validation Perplexity |
|---|---|
| 64  | 391.8 |
| 128 | 315.1 |
| 256 | 278.0 |
| 512 | 256.9 |

Perplexity decreased most when context length was doubled from 64 to 128, causing a ~77 poiny reduction. Further increasing context length led to diminishing returns. Doubling from 128 to 256 only caused a ~21 point reduction. This data would suggest that the model gets the most useful context from closer tokens, but longer-range context contributes smaller, but not insignificant, gains.

Full loss and perplexity curves, along with run configs can be found at [https://wandb.ai/jschaller2028-personal/ScribeLM](https://wandb.ai/jschaller2028-personal/ScribeLM).

## Limitations & Future Work

 - No learning rate schedule was used
 - Due to non-overlapping chunks, the 256 and 512 context configurations saw the entire dataset more than once, which could have caused overfitting, but the validation curves showed none through all 20,000 steps
 - Future experiments: learning rate schedule, attention head count ablation, learned positional embeddings, untyed embedding weights

`REPORT.md` contains a full report of the experimental proccess, and describes a surprising effect of parameter initialization.
