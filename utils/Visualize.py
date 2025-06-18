import torch
import matplotlib.pyplot as plt
import random
import os
from Models.MaDoUNet import MaDoUNet
from Dataset.dataset import DataModule

def visualize_predictions(
    image_dir: str,
    mask_dir: str,
    checkpoint_path: str,
    input_size=(256, 256),
    batch_size=8,
    num_workers=2,
    val_split=0.2,
    device=None,
    num_samples=3,
    save_path="outputs/predictions.png"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load model
    model = MaDoUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Setup dataset and dataloader
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
    val_dataset = val_loader.dataset

    if num_samples >= len(val_dataset):
        print(f"⚠️ num_samples ({num_samples}) >= dataset size ({len(val_dataset)}). Defaulting to 1 sample.")
        num_samples = 1


    # Select samples
    indices = random.sample(range(len(val_dataset)), num_samples)
    plt.figure(figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        image, mask = val_dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image_tensor)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

        # Convert tensors to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        gt_mask = mask.squeeze().cpu().numpy()
        pred_mask = pred.squeeze().cpu().numpy()

        # Plot
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image_np)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to: {save_path}")
