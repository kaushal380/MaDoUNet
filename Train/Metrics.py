import torch
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryAccuracy,
    BinaryRecall,
    BinaryPrecision,
    BinaryAUROC
)

def get_metrics(device: torch.device):
    return {
        "IoU":       BinaryJaccardIndex(threshold=0.5).to(device),
        "Dice":      BinaryF1Score(threshold=0.5).to(device),
        "Accuracy":  BinaryAccuracy(threshold=0.5).to(device),
        "Recall":    BinaryRecall(threshold=0.5).to(device),
        "Precision": BinaryPrecision(threshold=0.5).to(device),
        "AUROC":     BinaryAUROC().to(device)  # works on logits or probs
    }
