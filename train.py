import torch
import math
from torch import nn
from tokenizers import Tokenizer
from model import ScribeLM
from datasets import WikiText2
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_pretrained("gpt2")
tokenizer.get_vocab_size()

model = ScribeLM(context_length=128,
                       vocab_size=tokenizer.get_vocab_size()
                       )
model.to(device)

train_data = WikiText2(
                       split="train",
                       tokenizer=tokenizer,
                       context_length=128,
                       )

test_data = WikiText2(
                      split="test",
                      tokenizer=tokenizer,
                      context_length=128,
                      )

train_loader = DataLoader(dataset=train_data,
                          batch_size=32,
                          shuffle=True,
                          )

test_loader = DataLoader(dataset=test_data,
                          batch_size=64,
                          shuffle=False,
                          )

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

scaler = torch.amp.GradScaler(device=device)

print(f"|{"Mode":^10}|{"Epoch":^10}|{"Batch":^10}|{"Loss":^10}|{"Accuracy":^10}|{"PPL":^20}|")

for epoch in range(10):
  with torch.inference_mode():
    total_loss = 0.0
    total_accuracy = 0.0
    total_ppl = 0.0
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device, dtype=torch.float16):
          preds = model(x)
          loss = loss_fn(preds.permute(0, 2, 1), y)
        total_loss += loss.item()
        accuracy = float(torch.sum(torch.argmax(preds, dim=-1)==y)/y.numel()*100.0)
        total_accuracy += accuracy
        ppl = float(math.exp(loss.item()))
        total_ppl += ppl
        if (i+1)%100==0:
          print(f"|{"TEST":^10}|{epoch+1:^10}|{f"{i+1}/{len(test_loader)}":^10}|{round(loss.item(), 5):^10}|{round(accuracy, 5):^10}|{round(ppl, 5):^20}|")
    print(f"|{"TEST FINAL":^10}|{epoch+1:^10}|{"":^10}|{round(total_loss/len(test_loader), 5):^10}|{round(total_accuracy/len(test_loader), 5):^10}|{round(total_ppl/len(test_loader), 5):^20}|")
  for i, data in enumerate(train_loader):
    x, y = data
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast(device, dtype=torch.float16):
      preds = model(x)
      loss = loss_fn(preds.permute(0, 2, 1), y)

    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if (i+1)%100==0:
      print(f"|{"TRAIN":^10}|{epoch+1:^10}|{f"{i+1}/{len(train_loader)}":^10}|{round(loss.item(), 5):^10}|{round(float(torch.sum(torch.argmax(preds, dim=-1)==y)/y.numel()*100.0), 5):^10}|{round(float(math.exp(loss.item())), 5):^20}|")
  torch.save(model.state_dict(), "checkpoints/model.pth")