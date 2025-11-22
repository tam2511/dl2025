import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from model import MultiBranchMLP
from data_module import DataModule

data_dir = '../data'
dm = DataModule(data_dir=data_dir, batch_size=128, num_workers=0)
dm.setup()

model = MultiBranchMLP(
    input_dim=dm.input_dim,
    hidden_dim=256,
    output_dim=dm.n_classes,
    num_blocks=4,
    dropout=0.1,
    combine_mode='concat'
)

train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Быстрое обучение baseline (5 эпох)...")
for epoch in range(5):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='macro')
    print(f"Epoch {epoch+1}/5 - Train Loss: {train_loss/len(train_loader):.4f}, Test Acc: {acc:.4f}, Test F1: {f1:.4f}")

print(f"\nФинальное качество baseline (5 эпох):")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1-macro: {f1:.4f}")


