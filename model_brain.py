# model_brain.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# ================== DATASET (OPTIONAL - for future use) ==================
class BrainDataset:
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32)

        mask = mask.float().unsqueeze(0) / 255.0
        return img, mask


# ================== MODEL ==================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        out = self.pool4(x4)
        return x1, x2, x3, x4, out


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(512, 512)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        return self.dropout(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # size alignment (important)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up3 = UpBlock(128, 128, 64)
        self.up4 = UpBlock(64, 64, 32)

    def forward(self, x, x1, x2, x3, x4):
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class BrainUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4, out = self.encoder(x)
        out = self.bottleneck(out)
        out = self.decoder(out, x1, x2, x3, x4)
        return self.final(out)


# ================== LOAD MODEL ==================
def load_brain_model(model_path, device="cpu"):
    model = BrainUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ================== INFERENCE FUNCTION ==================
def predict_brain(image, model, device="cpu"):
    """
    image: numpy array (H, W, 3)
    """

    img_resized = cv2.resize(image, (256, 256)) / 255.0
    
    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_resized - mean) / std
    
    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(img_tensor)

    pred_probs = torch.sigmoid(pred[0]).cpu().numpy().squeeze()
    # Increased threshold to 0.6 to avoid false positives with normalization
    brain_threshold = 0.6
    pred_mask = (pred_probs > brain_threshold).astype(np.uint8)

    # Noise Filtering: Remove tiny speckles
    kernel = np.ones((3,3), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    # Relative to brain area
    gray_small = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, brain_mask_small = cv2.threshold(gray_small, 30, 255, cv2.THRESH_BINARY)
    brain_pixel_count = np.sum(brain_mask_small > 0)
    
    tumor_percent = (pred_mask.sum() / (brain_pixel_count + 1e-6)) * 100

    return pred_mask, tumor_percent