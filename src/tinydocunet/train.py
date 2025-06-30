import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import logging
import json
from datetime import datetime
import cv2
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings("ignore")

from tinydocunet.model import Doc_UNet  # Import model class
from tinydocunet.td_dataset import DisplacementDataset  # Import dataset class

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better text preservation"""
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])  # Up to relu3_3
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
        
    def forward(self, input_img, target_img):
        # Convert single channel to 3 channels for VGG
        if input_img.size(1) == 1:
            input_img = input_img.repeat(1, 3, 1, 1)
        if target_img.size(1) == 1:
            target_img = target_img.repeat(1, 3, 1, 1)
            
        # Extract features
        input_features = self.feature_extractor(input_img)
        target_features = self.feature_extractor(target_img)
        
        return self.mse_loss(input_features, target_features)

class SSIMMetric:
    """SSIM metric calculator"""
    @staticmethod
    def calculate_ssim(img1, img2):
        """Calculate SSIM between two images"""
        # Convert to numpy and squeeze
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
            
        # Handle batch dimension
        if len(img1.shape) == 4:
            img1 = img1[0, 0]  # Take first item, first channel
            img2 = img2[0, 0]
        elif len(img1.shape) == 3:
            img1 = img1[0]
            img2 = img2[0]
            
        return ssim(img1, img2, data_range=1.0)

class DocUNetTrainer:
    def __init__(self, model, device='cuda', lr=1e-4, accumulation_steps=4):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Better learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(device)
        self.ssim_metric = SSIMMetric()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ssim_scores = []
        self.val_ssim_scores = []
        self.learning_rates = []
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/training_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        log_file = os.path.join(log_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir
        
        # Save config
        config = {
            "learning_rate": self.lr,
            "accumulation_steps": self.accumulation_steps,
            "device": str(self.device),
            "timestamp": timestamp
        }
        with open(os.path.join(log_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    def calculate_combined_loss(self, pred, target, original_img):
        """Calculate combined loss: L1 + Perceptual + Gradient"""
        # L1 Loss
        l1 = self.l1_loss(pred, target)
        
        # Apply displacement to create warped image (simplified)
        try:
            # Create a simple warped version for perceptual loss
            warped_img = self.apply_displacement_simple(original_img, pred)
            perceptual = self.perceptual_loss(warped_img, original_img) * 0.1
        except:
            perceptual = torch.tensor(0.0, device=self.device)
        
        # Gradient loss for smoothness
        grad_loss = self.gradient_loss(pred) * 0.01
        
        total_loss = l1 + perceptual + grad_loss
        
        return total_loss, {
            'l1': l1.item(),
            'perceptual': perceptual.item() if isinstance(perceptual, torch.Tensor) else 0.0,
            'gradient': grad_loss.item()
        }
    
    def apply_displacement_simple(self, img, displacement):
        """Simple displacement application for perceptual loss"""
        # This is a simplified version - in practice you'd use proper grid sampling
        return img  # Placeholder - return original for now
    
    def gradient_loss(self, displacement):
        """Gradient loss for smooth displacement field"""
        dx = displacement[:, :, :, 1:] - displacement[:, :, :, :-1]
        dy = displacement[:, :, 1:, :] - displacement[:, :, :-1, :]
        return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    
    def train_epoch(self, train_loader):
        """Train one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        total_ssim = 0.0
        num_batches = len(train_loader)
        loss_components = {'l1': 0.0, 'perceptual': 0.0, 'gradient': 0.0}
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            y1, y2 = self.model(images)
            
            # Calculate losses for both stages
            loss1, components1 = self.calculate_combined_loss(y1, targets, images)
            loss2, components2 = self.calculate_combined_loss(y2, targets, images)
            
            # Combined loss with stage weighting
            batch_loss = (0.3 * loss1 + 0.7 * loss2) / self.accumulation_steps
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Calculate SSIM (using stage 2 output)
            try:
                batch_ssim = self.ssim_metric.calculate_ssim(y2, targets)
                total_ssim += batch_ssim
            except:
                batch_ssim = 0.0
            
            total_loss += batch_loss.item() * self.accumulation_steps
            
            # Accumulate loss components
            for key in loss_components:
                loss_components[key] += (components1[key] + components2[key]) / 2
            
            # Progress logging
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{num_batches}, '
                    f'Loss: {batch_loss.item() * self.accumulation_steps:.6f}, '
                    f'SSIM: {batch_ssim:.4f}'
                )
        
        # Final gradient step if needed
        if len(train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / num_batches
        avg_ssim = total_ssim / num_batches
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return avg_loss, avg_ssim, loss_components
    
    def validate(self, val_loader):
        """Enhanced validation with metrics"""
        self.model.eval()
        total_loss = 0.0
        total_ssim = 0.0
        loss_components = {'l1': 0.0, 'perceptual': 0.0, 'gradient': 0.0}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                y1, y2 = self.model(images)
                
                # Calculate losses
                loss1, components1 = self.calculate_combined_loss(y1, targets, images)
                loss2, components2 = self.calculate_combined_loss(y2, targets, images)
                
                batch_loss = 0.3 * loss1 + 0.7 * loss2
                total_loss += batch_loss.item()
                
                # SSIM
                try:
                    batch_ssim = self.ssim_metric.calculate_ssim(y2, targets)
                    total_ssim += batch_ssim
                except:
                    pass
                
                # Loss components
                for key in loss_components:
                    loss_components[key] += (components1[key] + components2[key]) / 2
        
        avg_loss = total_loss / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        
        for key in loss_components:
            loss_components[key] /= len(val_loader)
        
        return avg_loss, avg_ssim, loss_components
    
    def save_checkpoint(self, epoch, train_loss, val_loss, train_ssim, val_ssim, 
                       save_dir, is_best=False, is_periodic=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_ssim': train_ssim,
            'val_ssim': val_ssim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ssim_scores': self.train_ssim_scores,
            'val_ssim_scores': self.val_ssim_scores,
            'learning_rates': self.learning_rates
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            self.logger.info(f'Best model saved! Val Loss: {val_loss:.6f}, Val SSIM: {val_ssim:.4f}')
        
        if is_periodic:
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            self.logger.info(f'Periodic checkpoint saved for epoch {epoch}')
    
    def visualize_predictions(self, val_loader, save_dir, epoch):
        """Visualize predictions for monitoring"""
        self.model.eval()
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= 3:  # Only save first 3 batches
                    break
                    
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                y1, y2 = self.model(images)
                
                # Save first image in batch
                img = images[0, 0].cpu().numpy()
                target = targets[0].cpu().numpy()  # 2 channels
                pred = y2[0].cpu().numpy()  # 2 channels
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original image
                axes[0, 0].imshow(img, cmap='gray')
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                # Target displacement X
                axes[0, 1].imshow(target[0], cmap='viridis')
                axes[0, 1].set_title('Target Displacement X')
                axes[0, 1].axis('off')
                
                # Predicted displacement X
                axes[1, 0].imshow(pred[0], cmap='viridis')
                axes[1, 0].set_title('Predicted Displacement X')
                axes[1, 0].axis('off')
                
                # Difference
                diff = np.abs(target[0] - pred[0])
                axes[1, 1].imshow(diff, cmap='hot')
                axes[1, 1].set_title('Absolute Difference')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch}_batch_{i}.png'))
                plt.close()
    
    def train(self, train_loader, val_loader, epochs=50, save_dir='checkpoints'):
        """Enhanced training loop"""
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        viz_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        self.logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_ssim, train_components = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_ssim, val_components = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_ssim_scores.append(train_ssim)
            self.val_ssim_scores.append(val_ssim)
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - start_time
            
            # Comprehensive logging
            self.logger.info(f'Epoch {epoch+1}/{epochs}:')
            self.logger.info(f'  Train Loss: {train_loss:.6f} (L1: {train_components["l1"]:.6f}, '
                           f'Perceptual: {train_components["perceptual"]:.6f}, '
                           f'Gradient: {train_components["gradient"]:.6f})')
            self.logger.info(f'  Val Loss: {val_loss:.6f} (L1: {val_components["l1"]:.6f}, '
                           f'Perceptual: {val_components["perceptual"]:.6f}, '
                           f'Gradient: {val_components["gradient"]:.6f})')
            self.logger.info(f'  Train SSIM: {train_ssim:.4f}')
            self.logger.info(f'  Val SSIM: {val_ssim:.4f}')
            self.logger.info(f'  Time: {epoch_time:.2f}s')
            self.logger.info(f'  LR: {current_lr:.8f}')
            self.logger.info('-' * 60)
            
            # Save best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss, train_ssim, val_ssim, 
                                   save_dir, is_best=True)
            else:
                patience_counter += 1
            
            # Periodic checkpoints (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss, train_ssim, val_ssim,
                                   save_dir, is_periodic=True)
            
            # Visualizations (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.visualize_predictions(val_loader, viz_dir, epoch + 1)
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        self.logger.info('Training completed!')
        
        # Save final training curves
        self.plot_training_curves(save_dir)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ssim_scores': self.train_ssim_scores,
            'val_ssim_scores': self.val_ssim_scores,
            'learning_rates': self.learning_rates
        }
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self, save_dir):
        """Enhanced training curves with multiple metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # SSIM curves
        axes[0, 1].plot(self.train_ssim_scores, label='Train SSIM', color='green')
        axes[0, 1].plot(self.val_ssim_scores, label='Validation SSIM', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM Scores')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Loss comparison
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            axes[1, 1].plot(np.array(self.val_losses) - np.array(self.train_losses), 
                           color='red', label='Val - Train Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].set_title('Overfitting Monitor')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path):
        """Enhanced checkpoint loading"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_ssim_scores = checkpoint.get('train_ssim_scores', [])
        self.val_ssim_scores = checkpoint.get('val_ssim_scores', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {checkpoint.get('epoch', 0) + 1}")


def main():
    """Enhanced main training function"""
    
    # Hyperparameters
    BATCH_SIZE = 4  # Reduced for gradient accumulation
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
    DATA_DIR = r"dataset\generated_grayscale"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*60)
    print("ENHANCED DOCUNET GRAYSCALE DOCUMENT TRAINING")
    print("="*60)

    # Create datasets
    dataset = DisplacementDataset(DATA_DIR)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Effective batch size: {BATCH_SIZE * ACCUMULATION_STEPS}")
    
    # Create model
    model = Doc_UNet(input_channels=1, n_classes=2)
    
    print(f"Model created: {model.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-"*60)
    
    # Create enhanced trainer
    trainer = DocUNetTrainer(
        model, 
        device=device, 
        lr=LEARNING_RATE, 
        accumulation_steps=ACCUMULATION_STEPS
    )
    
    # Start training
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    print("Enhanced training completed!")


def test_inference_pipeline(model_path="checkpoints/best_model.pth", test_image=None):
    """Complete inference pipeline with post-processing"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Doc_UNet(input_channels=1, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print("Model loaded successfully!")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"Best validation SSIM: {checkpoint.get('val_ssim', 'N/A'):.4f}")
    
    # Test inference
    with torch.no_grad():
        if test_image is None:
            # Dummy input
            test_input = torch.randn(1, 1, 256, 256).to(device)
        else:
            test_input = test_image.to(device)
        
        y1, y2 = model(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Stage 1 output: {y1.shape}")
        print(f"Stage 2 output: {y2.shape}")
        print(f"Displacement range: [{y2.min():.4f}, {y2.max():.4f}]")
        
        # Here you would add the actual unwarping logic
        # unwarped_image = apply_displacement_field(test_input, y2)
        
        print("Inference pipeline test successful!")
        
        return y2  # Return displacement field


if __name__ == "__main__":
    main()
    
    # Test enhanced inference
    # test_inference_pipeline()