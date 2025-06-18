import os
import torch
from PIL import Image
from torchvision import transforms
from Models.MaDoUNet import MaDoUNet
from tqdm import tqdm


def predict_directory(
    input_dir,
    output_dir,
    checkpoint_path,
    input_size=(256, 256),
    threshold=0.5,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = MaDoUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

    # List image files
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

    print(f"🔍 Found {len(image_files)} image(s) in '{input_dir}'")

    for filename in tqdm(image_files, desc="Running Inference"):
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            pred_mask = (probs > threshold).float()

        # Convert to PIL Image
        mask_img = transforms.ToPILImage()(pred_mask.squeeze().cpu())
        mask_img.save(os.path.join(output_dir, filename))

    print(f"Saved predictions to '{output_dir}'")
