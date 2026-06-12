"""Trains a model according to hyperparameters from a config file, and logs results to a wandb project"""

import torch
import math
import wandb
import yaml
import argparse
import os
from torch import nn
from model import ScribeLM
from data import WikiText103
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str, autocast_dtype=torch.float16, use_amp=False) -> float:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        total_loss = 0.0
        for data in dataloader:
            x, y = data
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, dtype=autocast_dtype, enabled=use_amp):
                preds = model(x)
                loss = loss_fn(preds.permute(0, 2, 1), y)
            total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss/len(dataloader)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

wandb_config_path = os.path.abspath(os.path.dirname(__file__))+"/configs/wandb.yaml"
if os.path.exists(wandb_config_path):
    with open(wandb_config_path, "r") as f:
        wandb_config = yaml.safe_load(f)
    use_wandb = wandb_config["wandb"]["enabled"]
    wandb_entity = wandb_config["wandb"]["entity"]
    wandb_project = wandb_config["wandb"]["project"]
else:
    print(f"wandb config not found at {wandb_config_path}, not logging to wanb.")
    use_wandb = False
    wandb_entity = None
    wandb_project = None

run_name = config["train"]["run_name"]
epochs = config["train"]["epochs"]
batch_size = config["train"]["batch_size"]
lr = config["train"]["lr"]
seed = config["train"]["seed"]
context_length = config["model"]["context_length"]
vocab_size = config["model"]["vocab_size"]
num_layers = config["model"]["num_layers"]
d_model = config["model"]["d_model"]
d_feedforward = config["model"]["d_feedforward"]
num_heads = config["model"]["num_heads"]

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

tokenizer = Tokenizer.from_pretrained("gpt2")

train_data = WikiText103(
                    split="train",
                    tokenizer=tokenizer,
                    context_length=context_length,
                    )
validation_data = WikiText103(
                    split="validation",
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
validation_loader = DataLoader(dataset=validation_data,
                        batch_size=batch_size,
                        shuffle=False,
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

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

if not use_bf16:
    print("Warning: bfloat16 not supported")

use_amp = device == "cuda"
autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

run = wandb.init(
    entity=wandb_entity,
    project=wandb_project,
    name=run_name,
    config=config,
    mode = "online" if use_wandb else "disabled",
)

print(f"|{"Mode":^10}|{"Epoch":^10}|{"Batch":^10}|{"Loss":^10}|{"PPL":^20}|")

total_steps = 0
best_val_loss = float("inf")

model.train()
for epoch in range(epochs):
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type=device, dtype=autocast_dtype, enabled=use_amp):
            preds = model(x)
            loss = loss_fn(preds.permute(0, 2, 1), y)

        optimizer.zero_grad()
        if use_amp and not use_bf16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_steps += 1
        if (batch_idx+1)%500==0:
            train_loss = loss.item()
            train_ppl = float(math.exp(loss.item()))
            print(f"|{"TRAIN":^10}|{epoch+1:^10}|{f"{batch_idx+1}/{len(train_loader)}":^10}|{round(train_loss, 5):^10}|{round(train_ppl, 5):^20}|")
            validation_loss = evaluate(model=model,
                                       dataloader=validation_loader,
                                       loss_fn=loss_fn,
                                       device=device,
                                       autocast_dtype=autocast_dtype,
                                       use_amp=use_amp,
                                       )
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                torch.save(model.state_dict(), "best_model.pt")
            validation_ppl = math.exp(validation_loss)
            print(f"|{"VALID":^10}|{epoch+1:^10}|{"":^10}|{round(validation_loss, 5):^10}|{round(validation_ppl, 5):^20}|")
            run.log(
                {
                    "train/loss": train_loss,
                    "train/perplexity": train_ppl,
                    "validation/loss": validation_loss,
                    "validation/perplexity": validation_ppl,
                },
                step = total_steps
            )

model.load_state_dict(torch.load("best_model.pt", map_location=device))

test_loss = evaluate(model=model,
                     dataloader=test_loader,
                     loss_fn=loss_fn,
                     device=device,
                     autocast_dtype=autocast_dtype,
                     use_amp=use_amp,
                     )
test_ppl = math.exp(test_loss)

run.summary["test/loss"] = test_loss
run.summary["test/perplexity"] = test_ppl

artifact = wandb.Artifact(
    name=f"{run_name}-model",
    type="model",
    metadata={
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
    },
)
artifact.add_file("best_model.pt")
run.log_artifact(artifact)

run.finish()