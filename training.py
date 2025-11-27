import copy
import torch
import torch.nn as nn
import torch.optim as optim
from utils import device

def train_one_epoch(model, loader, optimizer, loss_fn, device=device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, loss_fn, device=device):
    model.eval()
    model = model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_model(model, train_loader, val_loader, epochs=5, lr=0.01):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
