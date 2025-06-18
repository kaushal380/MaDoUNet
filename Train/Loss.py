import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous()
        y_true = y_true.contiguous()

        intersection = (y_pred * y_true).sum(dim=(2, 3))
        union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()


def get_loss_fn():
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    def combined_loss(logits, masks):
        """
        Combined BCE + Dice loss.
        Expects:
          - logits: raw model outputs
          - masks: ground truth masks
        """
        probs = torch.sigmoid(logits)
        return 0.5 * bce(logits, masks) + 0.5 * dice(probs, masks)

    return combined_loss
