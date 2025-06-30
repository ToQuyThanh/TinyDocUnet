# DocUNet Training Pipeline

A comprehensive deep learning pipeline for **Document Unwarping** using enhanced U-Net architecture. This system corrects geometric distortions in scanned or photographed documents by learning displacement fields.

## 🎯 Overview

DocUNet learns to predict 2D displacement fields (dx, dy) that can "unwarp" distorted document images back to their flat, readable form. This is particularly useful for:

- Mobile document scanning applications
- OCR preprocessing for curved/bent documents  
- Historical document digitization
- Real-time document correction

## 🏗️ Architecture

```
Input: Grayscale Document (1 channel, 256x256)
    ↓
DocUNet (Two-Stage U-Net)
    ↓
Output: Displacement Field (2 channels: dx, dy)
```

**Two-Stage Design:**
- **Stage 1**: Coarse displacement prediction
- **Stage 2**: Refined displacement (weighted 70% in loss)

## 🚀 Key Features

### 📊 **Advanced Loss Functions**
- **L1 Loss**: Primary reconstruction loss
- **Perceptual Loss**: VGG16-based feature preservation for text quality
- **Gradient Loss**: Ensures smooth displacement fields
- **Weighted Combination**: `Total = L1 + 0.1×Perceptual + 0.01×Gradient`

### 📈 **Comprehensive Metrics**
- **SSIM (Structural Similarity)**: Real image quality assessment
- **Multi-component Loss Tracking**: Individual loss component monitoring
- **Training/Validation Curves**: Full metric visualization

### 🎓 **Advanced Training Techniques**

#### **Gradient Accumulation**
```python
# Effective batch size = batch_size × accumulation_steps
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4  # Effective: 16
```
- Enables larger effective batch sizes on limited GPU memory
- Maintains gradient stability
- Reduces training variance

#### **Smart Learning Rate Scheduling**
```python
CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-7)
```
- **Warm Restarts**: Periodic LR resets to escape local minima
- **Cosine Annealing**: Smooth LR decay between restarts
- **Adaptive**: T_mult=2 increases restart periods exponentially

#### **Enhanced Regularization**
- **Gradient Clipping**: `max_norm=1.0` prevents exploding gradients
- **Weight Decay**: `1e-4` for AdamW optimizer
- **Early Stopping**: Patience=15 with validation monitoring

### 🔍 **Monitoring & Visualization**

#### **Real-time Logging**
```
logs/training_20241229_143052/
├── training.log          # Detailed training logs
├── config.json          # Hyperparameter tracking
└── training_history.json # Exportable metrics
```

#### **Visual Monitoring**
```
checkpoints/visualizations/
├── predictions_epoch_5_batch_0.png
├── predictions_epoch_10_batch_0.png
└── ...
```

**Visualization includes:**
- Original distorted document
- Ground truth displacement (X & Y components)
- Predicted displacement fields
- Absolute difference heatmaps

#### **Training Curves**
- **Loss Evolution**: Training vs Validation
- **SSIM Progress**: Image quality metrics
- **Learning Rate Schedule**: CosineAnnealing visualization
- **Overfitting Monitor**: Val-Train loss difference

### 💾 **Smart Checkpointing**

#### **Multiple Checkpoint Types**
```
checkpoints/
├── best_model.pth           # Best validation performance
├── checkpoint_epoch_10.pth  # Periodic saves (every 10 epochs)
├── checkpoint_epoch_20.pth
├── training_curves.png      # Final visualization
└── training_history.json    # Complete metrics export
```

#### **Complete State Saving**
- Model weights + optimizer state
- Scheduler state for resume training
- Full training history and metrics
- Hyperparameter configuration

## 📦 Installation

### Requirements
```bash
pip install torch torchvision
pip install numpy matplotlib opencv-python
pip install scikit-image logging pathlib
```

### File Structure
```
project/
├── improved_trainer.py     # Main training pipeline
├── unet.py                 # DocUNet model definition
├── cccd.py                 # DisplacementDataset class
├── dataset/
│   └── generated_grayscale/    # Training data
├── checkpoints/            # Model saves (auto-created)
└── logs/                  # Training logs (auto-created)
```

## 🎮 Usage

### Basic Training
```python
python improved_trainer.py
```

### Advanced Configuration
```python
# Hyperparameters
BATCH_SIZE = 4              # GPU memory dependent
LEARNING_RATE = 1e-4        # Adaptive with scheduler
EPOCHS = 50                 # Early stopping enabled
ACCUMULATION_STEPS = 4      # Effective batch size = 16
```

### Resume Training
```python
trainer = DocUNetTrainer(model, device=device, lr=LEARNING_RATE)
trainer.load_checkpoint("checkpoints/checkpoint_epoch_20.pth")
trainer.train(train_loader, val_loader, epochs=50)
```

### Inference Pipeline
```python
# Complete inference with metrics
displacement_field = test_inference_pipeline(
    model_path="checkpoints/best_model.pth",
    test_image=your_distorted_document
)
```

## 📊 Training Output

### Console Logging
```
Epoch 15/50:
  Train Loss: 0.004521 (L1: 0.004102, Perceptual: 0.000341, Gradient: 0.000078)
  Val Loss: 0.004836 (L1: 0.004394, Perceptual: 0.000362, Gradient: 0.000080)
  Train SSIM: 0.8734
  Val SSIM: 0.8612
  Time: 45.23s
  LR: 0.00006234
```

### Metric Tracking
- **Loss Components**: Individual contribution monitoring
- **SSIM Evolution**: Structural similarity improvement
- **Learning Rate**: Adaptive scheduling visualization
- **Time Tracking**: Epoch duration monitoring

## 🔧 Technical Details

### Loss Function Design
```python
def calculate_combined_loss(pred, target, original_img):
    l1 = L1Loss(pred, target)                    # Reconstruction
    perceptual = VGGLoss(warped_img, original)   # Text preservation  
    gradient = GradientLoss(pred)                # Smoothness
    
    return l1 + 0.1*perceptual + 0.01*gradient
```

### Two-Stage Training
```python
# Stage weighting in final loss
total_loss = 0.3 * stage1_loss + 0.7 * stage2_loss
```

### Memory Optimization
- **Gradient Accumulation**: Large effective batch size
- **Pin Memory**: Faster GPU transfers
- **Mixed Precision**: Optional for further speedup

## 📈 Performance Optimization

### GPU Memory Usage
- **Batch Size**: Start with 4, adjust based on GPU memory
- **Accumulation**: Increase steps if memory allows
- **Workers**: Set `num_workers=2` for optimal I/O

### Training Speed
- **Pin Memory**: Enable for GPU training
- **Gradient Accumulation**: Reduce backward pass frequency
- **Efficient Logging**: Periodic rather than every batch

### Convergence Tips
- **Warm Restarts**: Helps escape local minima
- **Gradient Clipping**: Prevents training instability  
- **Early Stopping**: Prevents overfitting

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8  # Maintain effective batch size
```

**Slow Convergence**
```python
# Increase learning rate or adjust scheduler
LEARNING_RATE = 2e-4
T_0 = 5  # Shorter restart cycles
```

**Poor SSIM Scores**
- Check displacement field magnitude
- Verify ground truth quality
- Increase perceptual loss weight

**Training Instability**
- Lower learning rate
- Increase gradient clipping
- Check for NaN values in loss

### Monitoring Guidelines
- **SSIM should increase**: >0.7 for good quality
- **Loss components balanced**: No single component dominates
- **Learning rate**: Should decrease smoothly with restarts

## 📚 Advanced Features

### Custom Dataset Integration
```python
# Implement your own dataset
class CustomDisplacementDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Load your distorted images + displacement ground truth
        pass
```

### Multi-Scale Training
```python
# Train on multiple resolutions
for scale in [128, 256, 512]:
    model = Doc_UNet(input_channels=1, n_classes=2)
    # Scale-specific training
```

### Ensemble Methods
```python
# Combine multiple models for robust prediction
models = [load_model(path) for path in model_paths]
ensemble_pred = torch.mean(torch.stack([m(x) for m in models]), dim=0)
```

## 📄 Citation

If you use this pipeline in your research, please cite:
```bibtex
@misc{docunet_pipeline,
  title={Enhanced DocUNet Training Pipeline for Document Unwarping},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/docunet-pipeline}
}
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For questions and issues:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Email: your.email@example.com
- Documentation: [Wiki](https://github.com/your-repo/wiki)

---

**Happy Training! 🚀**