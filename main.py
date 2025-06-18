import argparse
import os
# import torch
from datetime import datetime

from train import run_training
from utils.Visualize import visualize_predictions
from test import evaluate_model
from inference import predict_directory

def make_run_folder(base_dir, action, run_instance_name=None):
    if not run_instance_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_instance_name = f"{timestamp}_{action}"

    run_dir = os.path.join(base_dir, run_instance_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_instance_name

def main():
    parser = argparse.ArgumentParser(description="Train/Test/Visualize MaDoUNet on Kvasir-SEG")

    # Dataset
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--input_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_split', type=float, default=0.2)

    # Run config
    parser.add_argument('--run_instance_name', type=str, default=None, help="Optional run name. Defaults to timestamp.")

    # Actions
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')

    # Training args
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=30)

    # Visualization
    parser.add_argument('--num_samples', type=int, default=1)

    # Inference
    parser.add_argument('--predictions_dir', type=str, help="(Optional) Will be overridden if run_instance_name is used.")
    parser.add_argument('--threshold', type=float, default=0.5)

    # Testing
    parser.add_argument('--test_metrics_path', type=str, help="(Optional) Will be overridden if run_instance_name is used.")

    # Visualize and Inference and Test checkpoint
    parser.add_argument('--checkpoint_path', type=str, help="Path to trained model (.pth)")

    args = parser.parse_args()

    # Training
    if args.train:
        train_dir, name = make_run_folder(r"results\train_results", "train", args.run_instance_name)
        args.save_path = os.path.join(train_dir, "madounet_best.pth")
        args.log_file = os.path.join(train_dir, "madounet_train_log.csv")

        run_training(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            lr=args.lr,
            num_epochs=args.num_epochs,
            save_path=args.save_path,
            log_file=args.log_file
        )

    # Visualization
    if args.visualize:
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for visualization (--checkpoint_path)")

        vis_dir, name = make_run_folder("results/visualization_results", "visualize", args.run_instance_name)
        args.visualize_path = os.path.join(vis_dir, "visualization.png")

        visualize_predictions(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            checkpoint_path=args.checkpoint_path,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            num_samples=args.num_samples,
            save_path=args.visualize_path
        )

    # Testing
    if args.test:
        test_dir, name = make_run_folder("results/test_results", "test", args.run_instance_name)
        args.test_metrics_path = os.path.join(test_dir, "test_metrics.csv")

        evaluate_model(
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            checkpoint_path=args.checkpoint_path,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            output_csv=args.test_metrics_path
        )

    # Inference
    if args.inference:
        infer_dir, name = make_run_folder("results/inference_results", "inference", args.run_instance_name)
        args.predictions_dir = infer_dir

        predict_directory(
            input_dir=args.image_dir,
            output_dir=args.predictions_dir,
            checkpoint_path=args.checkpoint_path,
            input_size=tuple(args.input_size),
            threshold=args.threshold,
        )

if __name__ == "__main__":
    main()
