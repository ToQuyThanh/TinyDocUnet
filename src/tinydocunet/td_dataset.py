import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
from pathlib import Path

class DisplacementDataset(Dataset):
    """
    Dataset cho bài toán dự đoán displacement map từ ảnh grayscale input
    
    Cấu trúc dữ liệu:
    dataset/
    ├── images/
    │   ├── 000000.jpg (ảnh grayscale)
    │   ├── 000001.jpg
    │   └── ...
    └── labels/
        ├── 000000.npy (displacement map)
        ├── 000001.npy
        └── ...
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
            transform (callable, optional): Transform áp dụng lên dữ liệu
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.transform = transform
        
        # Lấy danh sách tất cả file ảnh
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
        
        # Kiểm tra file displacement tương ứng
        self.valid_pairs = []
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.npy"
            if label_path.exists():
                self.valid_pairs.append((img_path, label_path))
        
        print(f"Tìm thấy {len(self.valid_pairs)} cặp dữ liệu hợp lệ")
        
        if len(self.valid_pairs) == 0:
            raise ValueError(f"Không tìm thấy dữ liệu hợp lệ trong {data_dir}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]
        
        # Load ảnh đa cấp xám
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Load displacement map
        displacement = np.load(str(label_path))
        
        # Convert sang tensor
        # Ảnh grayscale: (H, W) -> (1, H, W), normalize về [0, 1]
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        # Displacement: (H, W, 2) -> (2, H, W)
        displacement = torch.from_numpy(displacement).permute(2, 0, 1).float()
        
        # Apply transform nếu có
        if self.transform:
            image, displacement = self.transform(image, displacement)
        
        return image, displacement

