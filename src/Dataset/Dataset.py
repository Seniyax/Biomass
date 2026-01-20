import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class BiomassDataset(Dataset):
    def __init__(self, df: pd.DataFrame,img_dir,transform,train_target_cols,all_target_cols):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.train_target_cols = list[str]
        self.all_target_cols = list[str]
        self.train_targets = df[train_target_cols].values
        self.all_targets = df[all_target_cols].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        image_paths = self.df['image_path'].values
        img_idx = image_paths[idx]
        train_target = self.train_targets[idx]
        all_target = self.all_targets[idx]

        full_path = self.img_dir /Path(img_idx).name
        image = cv2.imread(str(full_path))

        if image is None:
            raise FileNotFoundError(f"Image not found at path: {full_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        mid_point = width // 2
        img_left = image[:, :mid_point]
        img_right = image[:, mid_point:]

        trans_left = self.transform(image=img_left)['image']
        trans_right = self.transform(image=img_right)['image']

        train_target_tensor = torch.tensor(train_target, dtype=torch.float32)
        all_target_tensor = torch.tensor(all_target, dtype=torch.float32)

        return trans_left,trans_right, train_target_tensor, all_target_tensor