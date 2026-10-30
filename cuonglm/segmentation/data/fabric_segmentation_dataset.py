"""Fabric segmentation dataset built from the labeled ``square_*`` folders."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SUBSETS = ("square_01", "square_02")
# UNet pools three times, so H and W must be divisible by 8.
IMAGE_SIZE = (480, 640)


class FabricSegmentationDataset(Dataset):
    """RGB images z-score normalized per image/channel, with binary masks."""

    def __init__(
        self,
        root: str | Path,
        subsets: Sequence[str] = SUBSETS,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ) -> None:
        self.image_size = tuple(image_size)
        self.samples: list[tuple[Path, Path]] = []
        for subset in subsets:
            image_dir, label_dir = (Path(root) / subset / name for name in ("images", "labels"))
            for image_path in sorted(image_dir.glob("*.png")):
                label_path = label_dir / image_path.name
                if not label_path.is_file():
                    raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
                self.samples.append((image_path, label_path))
        if not self.samples:
            raise ValueError(f"no images found under {root} for {list(subsets)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(label_path).convert("L"))
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        mask = torch.from_numpy(mask > 0).float().unsqueeze(0)  # (1, H, W)

        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        image = (image - mean) / std

        if tuple(image.shape[-2:]) != self.image_size:
            image = F.interpolate(
                image[None], size=self.image_size, mode="bilinear", align_corners=False
            )[0]
            mask = F.interpolate(mask[None], size=self.image_size, mode="nearest")[0]
        return image, mask


def create_dataloader(
    root: str | Path,
    *,
    subsets: Sequence[str] = SUBSETS,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    dataset = FabricSegmentationDataset(root, subsets, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
