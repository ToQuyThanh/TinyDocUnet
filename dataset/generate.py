import cv2
import numpy as np
import os
import random
from pathlib import Path

class DocDatasetGenerator:
    def __init__(self, input_dir, background_dir, output_dir, img_size=512, num_samples=10, 
                 distortion_level=0.1, max_displacement=50.0):
        self.input_dir = Path(input_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.num_samples = num_samples
        self.distortion_level = distortion_level
        self.max_displacement = max_displacement
        
        # Tạo thư mục output
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Load danh sách file
        self.scan_files = list(self.input_dir.glob("*.jpg")) + list(self.input_dir.glob("*.png"))
        self.bg_files = list(self.background_dir.glob("*.jpg")) + list(self.background_dir.glob("*.png"))
    
    def generate_perspective_points(self, img_size, distortion_level=None):
        """Tạo perspective points với kiểm soát distortion"""
        if distortion_level is None:
            distortion_level = self.distortion_level
            
        img_size = int(img_size)
        src = np.array([
            [0, 0],
            [img_size, 0], 
            [img_size, img_size],
            [0, img_size]
        ], dtype=np.float32)
        
        # Distorted corners với random offset được kiểm soát
        max_offset = int(img_size * distortion_level)
        dst = src.copy()
        
        for i in range(4):
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)
            dst[i] += [offset_x, offset_y]
        
        return src, dst
    
    def compute_displacement_map(self, src, dst, img_size):
        """Tính displacement map từ perspective transform với giới hạn"""
        # Forward transform matrix
        H = cv2.getPerspectiveTransform(src, dst)
        
        # Tạo grid coordinates
        x, y = np.meshgrid(np.arange(img_size), np.arange(img_size))
        coords = np.stack([x, y], axis=2).reshape(-1, 2).astype(np.float32)
        
        # Transform coordinates
        coords_homogeneous = np.column_stack([coords, np.ones(len(coords))])
        transformed = H @ coords_homogeneous.T
        transformed = transformed[:2] / transformed[2]  # Normalize
        transformed = transformed.T.reshape(img_size, img_size, 2)
        
        # Displacement = transformed - original
        original_coords = np.stack([x, y], axis=2)
        displacement = transformed - original_coords
        
        # Giới hạn displacement trong khoảng [-max_displacement, max_displacement]
        displacement = np.clip(displacement, -self.max_displacement, self.max_displacement)
        
        return displacement
    
    def create_sample(self, scan_path, bg_path, sample_id):
        """Tạo một sample training với ảnh grayscale"""
        # Load images
        scan = cv2.imread(str(scan_path))
        bg = cv2.imread(str(bg_path))
        
        # Chuyển về grayscale ngay từ đầu
        if len(scan.shape) == 3:
            scan = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
        if len(bg.shape) == 3:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        
        # Resize
        scan = cv2.resize(scan, (self.img_size, self.img_size))
        bg = cv2.resize(bg, (self.img_size, self.img_size))
        
        # Generate perspective points
        src, dst = self.generate_perspective_points(self.img_size)
        
        # Perspective transform
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(scan, H, (self.img_size, self.img_size),
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Create mask
        mask = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
        mask_warped = cv2.warpPerspective(mask, H, (self.img_size, self.img_size))
        
        # Composite với background (grayscale)
        result = bg.copy()
        valid_area = mask_warped > 128
        result[valid_area] = warped[valid_area]
        
        # Compute displacement map (inverse transform)
        displacement = self.compute_displacement_map(dst, src, self.img_size)
        
        # Save ảnh grayscale (single channel)
        cv2.imwrite(str(self.output_dir / "images" / f"{sample_id:06d}.jpg"), result)
        
        # Save displacement map
        np.save(str(self.output_dir / "labels" / f"{sample_id:06d}.npy"), displacement)
        
        return result, displacement
    
    def generate_dataset(self):
        """Tạo toàn bộ dataset"""
        print(f"Generating {self.num_samples} grayscale samples...")
        
        for i in range(self.num_samples):
            # Random chọn scan và background
            scan_file = random.choice(self.scan_files)
            bg_file = random.choice(self.bg_files)
            
            try:
                img, disp = self.create_sample(scan_file, bg_file, i)
                print(f"Generated grayscale sample {i+1}/{self.num_samples}")
                print(f"  Image shape: {img.shape}")
                print(f"  Displacement shape: {disp.shape}")
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
        
        print("Grayscale dataset generation completed!")
    
    def visualize_sample(self, sample_id):
        """Hiển thị một sample để kiểm tra"""
        img_path = self.output_dir / "images" / f"{sample_id:06d}.jpg"
        label_path = self.output_dir / "labels" / f"{sample_id:06d}.npy"
        
        if img_path.exists() and label_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            displacement = np.load(str(label_path))
            
            print(f"Sample {sample_id}:")
            print(f"  Image shape: {img.shape}")
            print(f"  Image dtype: {img.dtype}")
            print(f"  Image range: [{img.min()}, {img.max()}]")
            print(f"  Displacement shape: {displacement.shape}")
            print(f"  Displacement range: [{displacement.min():.2f}, {displacement.max():.2f}]")
            
            return img, displacement
        else:
            print(f"Sample {sample_id} not found!")
            return None, None

# Sử dụng
if __name__ == "__main__":
    INPUT_IMG = r"dataset\scan_images"
    BACKGROUND_DIR = r"dataset\background_images"
    OUTPUT_DIR = r"dataset\generated_grayscale"
    IMG_SIZE = 256
    NUM_SAMPLES = 10
    DISTORTION_LEVEL = 0.05  # Giảm distortion để giới hạn displacement
    MAX_DISPLACEMENT = 30.0  # Giới hạn displacement tối đa

    generator = DocDatasetGenerator(
        INPUT_IMG, BACKGROUND_DIR, OUTPUT_DIR, 
        IMG_SIZE, NUM_SAMPLES, 
        distortion_level=DISTORTION_LEVEL,
        max_displacement=MAX_DISPLACEMENT
    )
    generator.generate_dataset()
    
    # Kiểm tra một sample
    print("\n" + "="*50)
    print("CHECKING SAMPLE:")
    generator.visualize_sample(0)