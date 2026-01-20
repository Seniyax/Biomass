import os
from typing import Callable,Optional
from pathlib import Path
from __future__ import annotations
from dataclasses import dataclass, field    

@dataclass
class CONFIG:
    train_path: Path = Path("/kaggle/input/csiro-biomass/train.csv")
    train_img: Path = Path("/kaggle/input/csiro-biomass/train")
    state = 42
    model = ""
    folds = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    epoches = 20
    pretrained: bool = True
    img_size: int = 518
    train_lr: float = 1e-4
    train_targets: list[str] = field(default_factory=lambda: ['Dry_Total_g', 'GDM_g', 'Dry_Green_g'])
    all_target_cols: list[str] = field(default_factory=lambda: ['Dry_Total_g', 'GDM_g', 'Dry_Green_g','Dry_Clover_g','Dry_Dead_g'])
    r2_weights:list[float] = field(default_factory=lambda: [
        0.1, 0.1, 0.1, 0.2, 0.5
    ])
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        'total_loss': 0.5,
        'gdm_loss': 0.2,
        'green_loss': 0.1
    })
    def display_info(self):
        print(f"{'='*70}")
        print(f"Configuration")
        print(f"{'='*70}")
        print(f"Backbone:{self.model}")
        print(f"Epoches:{self.epoches}")
        print(f"Device:{self.device}")
        print(f"Image size:{self.img_size} x {self.img_size}")
        print(f"Output Dir:{self.output_dir}")
        print(f"{'='*70}\n")

