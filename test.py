import torch
from Models.MaDoUNet import MaDoUNet
from Dataset.dataset import DataModule
from Train.Metrics import get_metrics
import os
from tqdm import tqdm
import csv
from datetime import datetime

def evaluate_model(
    image_dir,
    mask_dir,
    checkpoint_path,
    input_size=(256, 256),
    batch_size=8,
    num_workers=2,
    val_split=0.2,
    device=None,
    output_csv="logs/test_metrics.csv"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MaDoUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Prepare data
    data_module = DataModule(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split
    )
    data_module.setup()
    _, val_loader = data_module.get_loaders()

    # Initialize metrics
    metrics = get_metrics(device)
    for metric in metrics.values():
        metric.reset()

    # Run evaluation
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks_int = masks.to(device).int()

            logits = model(images)
            probs = torch.sigmoid(logits)

            for name, metric in metrics.items():
                metric.update(probs, masks_int)

    # Compute results
    results = {name: metric.compute().item() for name, metric in metrics.items()}

    print("\n📊 Evaluation Metrics:")
    for name, value in results.items():
        print(f"{name:>10}: {value:.4f}")

    # Log to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp"] + list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Timestamp": datetime.now().isoformat(), **results})

    print(f"\n✅ Metrics saved to: {output_csv}")
