import torch
from Dataset.dataset import DataModule
from Models.MaDoUNet import MaDoUNet
from Train.Loss import get_loss_fn
from Train.Metrics import get_metrics
from Train.Optimizers import get_optimizer, get_scheduler
from Train.Trainer import train_model

def run_training(
    image_dir,
    mask_dir,
    input_size=(256, 256),
    batch_size=8,
    num_workers=2,
    val_split=0.2,
    lr=1e-4,
    num_epochs=30,
    save_path="checkpoints/madounet_best.pth",
    log_file="logs/madounet_train_log.csv",
    device = None 
):
    # Init dataset
    data_module = DataModule(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split
    )
    data_module.setup()
    train_loader, val_loader = data_module.get_loaders()

    # Device
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"Using device: {device}")

    # Model
    model = MaDoUNet().to(device)

    # Loss + Metrics
    loss_fn = get_loss_fn()
    metrics = get_metrics(device)

    # Optimizer + Scheduler
    optimizer = get_optimizer(model, lr=lr)
    scheduler = get_scheduler(optimizer)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        device=device,
        num_epochs=num_epochs,
        save_path=save_path,
        log_file=log_file
    )
