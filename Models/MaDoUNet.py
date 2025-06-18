import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# -------------------------
# Basic Blocks
# -------------------------
class UpBlock(nn.Module):
    """Simple upsample by 2 then two 3×3 convs."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.up(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

class SkipUpBlock(nn.Module):
    """Upsample + concat skip + two 3×3 convs."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch,           out_ch, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x

# -------------------------
# Encoders
# -------------------------
class EfficientNetB4Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # features_only returns 5 feature maps:
        self.encoder = timm.create_model("efficientnet_b4", pretrained=pretrained, features_only=True)
        self.out_ch  = [24, 32, 56, 160, 448]
    def forward(self, x):
        feats = self.encoder(x)
        return feats

class DenseNet121Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model("densenet121", pretrained=pretrained, features_only=True)
    def forward(self, x):
        feats = self.encoder(x)
        return feats

# -------------------------
# Transformer Block
# -------------------------
class DepthwiseConvTransformer(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4):
        super().__init__()
        hidden = in_ch * expansion
        self.proj1 = nn.Conv2d(in_ch, hidden, 1)
        self.dw    = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.bn    = nn.BatchNorm2d(hidden)
        self.act   = nn.GELU()
        self.proj2 = nn.Conv2d(hidden, out_ch, 1)
    def forward(self, x):
        x = self.proj1(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.proj2(x)

# -------------------------
# The Dual‑Encoder Model (logits only)
# -------------------------
class MaDoUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Encoder 1 ---
        self.enc1  = EfficientNetB4Encoder()
        self.trans = DepthwiseConvTransformer(in_ch=448, out_ch=512)

        # Build decoder1 with skips from enc1
        rev_ch = self.enc1.out_ch[::-1]   # [448,160,56,32,24]
        skip_ch = rev_ch[1:]              # [160,56,32,24]
        dec_chs  = [256, 128, 64, 32]
        self.dec1 = nn.ModuleList([
            SkipUpBlock(512, skip_ch[0], dec_chs[0]),
            SkipUpBlock(dec_chs[0], skip_ch[1], dec_chs[1]),
            SkipUpBlock(dec_chs[1], skip_ch[2], dec_chs[2]),
            SkipUpBlock(dec_chs[2], skip_ch[3], dec_chs[3]),
            UpBlock(dec_chs[3], 16)
        ])
        # raw logits head
        self.head1 = nn.Conv2d(16, 1, kernel_size=1)

        # --- Encoder 2 ---
        self.enc2 = DenseNet121Encoder()
        self.vss  = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,512*2,1),
            nn.Conv2d(512*2,512*2,3,padding=1,groups=512*2),
            nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(512*2,512,1), nn.BatchNorm2d(512),
        )
        # Decoder2 (no skips)
        dec2_chs = [256,128,64,32]
        ch = 512
        self.dec2 = nn.ModuleList()
        for out_ch in dec2_chs:
            self.dec2.append(UpBlock(ch, out_ch))
            ch = out_ch
        self.head2 = nn.Conv2d(ch, 1, kernel_size=1)

        # --- Learnable fusion (logits) ---
        self.fuse  = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        f1 = self.enc1(x)
        x1 = self.trans(f1[-1])
        for i, block in enumerate(self.dec1):
            if i < 4:
                x1 = block(x1, f1[-2 - i])
            else:
                x1 = block(x1)
        logit1 = self.head1(x1)

        # Stage 2
        out1_prob = torch.sigmoid(logit1)
        x2 = x * out1_prob
        f2 = self.enc2(x2)
        x2 = self.vss(f2[-1])
        for block in self.dec2:
            x2 = block(x2)
        logit2 = self.head2(x2)
        logit2 = F.interpolate(logit2, size=logit1.shape[2:], mode='bilinear', align_corners=False)

        # Fuse logits
        fused = self.fuse(torch.cat([logit1, logit2], dim=1))
        return fused
