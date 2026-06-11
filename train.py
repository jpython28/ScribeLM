"""Trains a model according to hyperparameters from a config file, and logs results to a wandb project"""

import torch
import math
import wandb
import yaml
import argparse
from torch import nn
from model import ScribeLM
from data import WikiText103
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

run_name = config["train"]["run_name"]
epochs = config["train"]["epochs"]
batch_size = config["train"]["batch_size"]
lr = config["train"]["lr"]
context_length = config["model"]["context_length"]
vocab_size = config["model"]["vocab_size"]
num_layers = config["model"]["num_layers"]
d_model = config["model"]["d_model"]
d_feedforward = config["model"]["d_feedforward"]
num_heads = config["model"]["num_heads"]

tokenizer = Tokenizer.from_pretrained("gpt2")

train_data = WikiText103(
                    split="train",
                    tokenizer=tokenizer,
                    context_length=context_length,
                    )
test_data = WikiText103(
                    split="test",
                    tokenizer=tokenizer,
                    context_length=context_length,
                    )

train_loader = DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        )
test_loader = DataLoader(dataset=test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        )
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ScribeLM(context_length=context_length,
                vocab_size=vocab_size,
                n=num_layers,
                d_model=d_model,
                d_ff=d_feedforward,
                h=num_heads,
                )
model.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

scaler = torch.amp.GradScaler(device=device)

if not torch.cuda.is_bf16_supported():
    print("Warning: bfloat16 not supported")

autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

run = wandb.init(
    entity="jschaller2028-personal",
    project="ScribeLM",
    name=run_name,
    config=config,
)

print(f"|{"Mode":^10}|{"Epoch":^10}|{"Batch":^10}|{"Loss":^10}|{"Accuracy":^10}|{"PPL":^20}|")

total_steps = 0

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device, dtype=autocast_dtype):
            preds = model(x)
            loss = loss_fn(preds.permute(0, 2, 1), y)

        optimizer.zero_grad()
        if torch.cuda.is_bf16_supported():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_steps += 1
        if (i+1)%500==0:
            train_loss = loss.item()
            train_accuracy = float(torch.sum(torch.argmax(preds, dim=-1)==y)/y.numel()*100.0)
            train_ppl = float(math.exp(loss.item()))
            print(f"|{"TRAIN":^10}|{epoch+1:^10}|{f"{i+1}/{len(train_loader)}":^10}|{round(train_loss, 5):^10}|{round(train_accuracy, 5):^10}|{round(train_ppl, 5):^20}|")
            with torch.inference_mode():
                test_loss = 0.0
                test_accuracy = 0.0
                test_ppl = 0.0
                for i, data in enumerate(test_loader):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast(device, dtype=autocast_dtype):
                        preds = model(x)
                        loss = loss_fn(preds.permute(0, 2, 1), y)
                    test_loss += loss.item()
                    accuracy = float(torch.sum(torch.argmax(preds, dim=-1)==y)/y.numel()*100.0)
                    test_accuracy += accuracy
                    ppl = float(math.exp(loss.item()))
                    test_ppl += ppl
            test_loss /= len(test_loader)
            test_accuracy /= len(test_loader)
            test_ppl /= len(test_loader)
            print(f"|{"TEST":^10}|{epoch+1:^10}|{"":^10}|{round(test_loss, 5):^10}|{round(test_accuracy, 5):^10}|{round(test_ppl, 5):^20}|")
            run.log(
            {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_perplexity": train_ppl,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_perplexity": test_ppl,
            }
            )
run.finish()