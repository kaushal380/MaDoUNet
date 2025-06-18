import torch
from tqdm import tqdm
import os

def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    metrics,
    device,
    num_epochs: int = 30,
    save_path: str = "checkpoints/best_model.pth",
    log_file: str = "logs/training_log.csv"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    model.to(device)

    # Write CSV header
    with open(log_file, "w") as f:
        header = "Epoch,Train Loss,Val Loss," + ",".join(metrics.keys()) + ",LR\n"
        f.write(header)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            images = images.to(device)
            masks_f = masks.to(device).float()

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks_f)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        for metric in metrics.values():
            metric.reset()

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images = images.to(device)
                masks_f = masks.to(device).float()
                masks_int = masks.to(device).int()

                logits = model(images)
                loss = loss_fn(logits, masks_f)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                for name, metric in metrics.items():
                    metric.update(probs, masks_int)

        val_loss /= len(val_loader)
        scores = {n: m.compute().item() for n, m in metrics.items()}
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        log_line = (
            f"{epoch},{train_loss:.4f},{val_loss:.4f}," +
            ",".join(f"{scores[n]:.4f}" for n in metrics) +
            f",{current_lr:.6f}"
        )
        print(f"[Epoch {epoch:2d}] {log_line}")
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        # Update LR
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (epoch {epoch})")
