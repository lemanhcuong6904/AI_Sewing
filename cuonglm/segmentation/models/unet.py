"""UNet architecture used by the lung-tumor segmentation notebook."""

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise convolution followed by a 1x1 pointwise convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def _conv3x3(in_channels: int, out_channels: int, separable: bool) -> nn.Module:
    if separable:
        return DepthwiseSeparableConv(in_channels, out_channels)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class DoubleConv(nn.Module):
    """Two normalized 3x3 convolution--ReLU operations preserving spatial size."""

    def __init__(
        self, in_channels: int, out_channels: int, separable: bool = False
    ) -> None:
        super().__init__()
        self.step = nn.Sequential(
            _conv3x3(in_channels, out_channels, separable),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            _conv3x3(out_channels, out_channels, separable),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x)


class UNet(nn.Module):
    """Four-level UNet producing a one-channel mask the size of the input.

    Height and width must be divisible by 8 (three 2x poolings), e.g. 480x640.
    """

    def __init__(self, in_channels: int = 3, separable: bool = False) -> None:
        super().__init__()

        self.layer1 = DoubleConv(in_channels, 64, separable)
        self.layer2 = DoubleConv(64, 128, separable)
        self.layer3 = DoubleConv(128, 256, separable)
        self.layer4 = DoubleConv(256, 512, separable)

        self.layer5 = DoubleConv(512 + 256, 256, separable)
        self.layer6 = DoubleConv(256 + 128, 128, separable)
        self.layer7 = DoubleConv(128 + 64, 64, separable)
        self.layer8 = nn.Conv2d(64, 1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(self.maxpool(x1))
        x3 = self.layer3(self.maxpool(x2))
        x4 = self.layer4(self.maxpool(x3))

        x5 = self.layer5(torch.cat([self.upsample(x4), x3], dim=1))
        x6 = self.layer6(torch.cat([self.upsample(x5), x2], dim=1))
        x7 = self.layer7(torch.cat([self.upsample(x6), x1], dim=1))
        return self.layer8(x7)


if __name__ == "__main__":
    from torchsummary import summary
    summary(UNet(in_channels=3), input_size=(3, 480, 640), device="cpu")
    # Total params (standard conv): 7,788,545; separable: 904,991