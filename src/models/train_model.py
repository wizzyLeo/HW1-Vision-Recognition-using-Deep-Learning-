import torch
import os
import time
from tempfile import TemporaryDirectory

device = (
    torch.device("mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu")
)

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels)

    return running_loss, running_corrects

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

    return running_loss, running_corrects

def train_model(
    model, criterion, optimizer, scheduler,
    dataloaders,
    num_epochs=25, early_stop_patience=5,
    save_path="/best_model.pt"
):
    save_path = os.path.join("../checkpoints", save_path.lstrip("/"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists

    since = time.time()
    best_acc = 0.0
    no_improve_counter = 0

    dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

    model = model.to(device)
    torch.save(model.state_dict(), save_path)  # Save initial model

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }


    for epoch in range(num_epochs):
        print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        train_loss, train_corrects = train_one_epoch(model, dataloaders['train'], criterion, optimizer)
        train_loss /= dataset_sizes['train']
        train_acc = train_corrects.float()  / dataset_sizes['train']
        print(f"🟢 Train    | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_corrects = evaluate(model, dataloaders['val'], criterion)
        val_loss /= dataset_sizes['val']
        val_acc = val_corrects.float()  / dataset_sizes['val']
        print(f"🔵 Val      | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # scheduler.step(val_acc.item())
        scheduler.step(epoch + 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve_counter = 0
            print(f"✅ New best model saved at: {save_path}")
        else:
            no_improve_counter += 1
            print(f"⚠️ No improvement for {no_improve_counter} epoch(s)")

        if no_improve_counter >= early_stop_patience:
            print("⛔ Early stopping triggered.")
            break



    time_elapsed = time.time() - since
    print(f"\n🏁 Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(torch.load(save_path))

    return model, history
