import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- CBAM Block ----
class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) with optional residual connection
    and configurable pooling type for spatial attention.
    """
    def __init__(self, channel, reduction=16, kernel_size=7, use_residual=False, spatial_pool='both'):
        super(CBAMBlock, self).__init__()
        assert spatial_pool in ['avg', 'max', 'both'], "spatial_pool must be one of 'avg', 'max', or 'both'"
        self.use_residual = use_residual
        self.spatial_pool = spatial_pool

        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2 if spatial_pool == 'both' else 1, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x if self.use_residual else None

        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        if self.spatial_pool == 'avg':
            sa_input = torch.mean(x, dim=1, keepdim=True)
        elif self.spatial_pool == 'max':
            sa_input, _ = torch.max(x, dim=1, keepdim=True)
        else:  # both
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            sa_input = torch.cat([avg_out, max_out], dim=1)

        sa = self.spatial_attention(sa_input)
        x = x * sa

        return x + residual if residual is not None else x


# ---- Inception Block ----
class InceptionBlock(nn.Module):
    """
    Inception-style block with 1x1, 3x3, 5x5, and pooling branches.
    BatchNorm is applied after each Conv2d layer.
    """
    def __init__(self, in_channels, out_channels=256):
        super(InceptionBlock, self).__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        branch_channels = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.pool_proj(x)
        return torch.cat([b1, b3, b5, bp], dim=1)


# ---- SE Block ----
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise feature recalibration.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
