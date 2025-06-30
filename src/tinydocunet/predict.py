import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Import the Doc_UNet model class (assuming unet.py contains it)
from tinydocunet.model import Doc_UNet 

class DocUNetPredictor:
    def __init__(self, model_path, input_channels=1, n_classes=2, img_size=(256, 256), device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model architecture
        self.model = Doc_UNet(input_channels=input_channels, n_classes=n_classes)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        
        print(f"Model loaded from {model_path} successfully!")
        print(f"Model will run on: {self.device}")
        
        self.img_size = img_size
        
        # Define the same transformations used during training (crucial!)
        # Assuming the training data was normalized to [0, 1] or similar
        # If your training data was specifically normalized to (-1, 1) using
        # transforms.Normalize((0.5,), (0.5,)), then you must use that here too.
        self.transform = T.Compose([
            T.ToPILImage(), # Ensure input is PIL Image for Resize
            T.Resize(self.img_size),
            T.Grayscale(num_output_channels=1), # Ensure 1 channel if not already
            T.ToTensor(), # Converts to [0, 1] range automatically
            # Add normalization if applied during training
            # T.Normalize((mean,), (std,)) 
            # For simplicity, assuming data is scaled to [0,1] or no special normalization for prediction
        ])

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image for inference.
        Assumes image is grayscale or will be converted to grayscale.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Convert to float32 and scale to [0, 1] if not already
        img = img.astype(np.float32) / 255.0
        
        # Convert numpy array to PyTorch tensor using the defined transform
        # Unsqueeze adds batch and channel dimensions (1, 1, H, W)
        img_tensor = self.transform(img).unsqueeze(0) 
        
        return img_tensor, img.shape # Return original shape for unwarping

    def predict_displacement(self, image_tensor):
        """
        Predicts the displacement field for a given image tensor.
        """
        with torch.no_grad():
            # Model returns y1, y2. We need y2 (the refined displacement field)
            _, displacement_field = self.model(image_tensor.to(self.device))
            
            # Move to CPU and convert to numpy
            # Output is (B, C, H, W) -> (1, 2, H, W)
            displacement_field_np = displacement_field.squeeze(0).cpu().numpy() # Shape (2, H, W)
            
        return displacement_field_np

    def unwarp_image(self, original_image_path, displacement_field_np):
        """
        Applies the predicted displacement field to unwarp the original image.
        
        Args:
            original_image_path (str): Path to the original warped image.
            displacement_field_np (np.ndarray): Predicted displacement field (2, H, W).
                                                  Contains dx and dy for each pixel.
        Returns:
            np.ndarray: The unwarped image.
        """
        # Load the original image again, without resizing, for unwarping
        img_original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise FileNotFoundError(f"Original image for unwarping not found at {original_image_path}")

        # Ensure original image is float32 for interpolation
        img_original = img_original.astype(np.float32) / 255.0 # Scale to [0, 1]
        H, W = img_original.shape
        
        # Resize the displacement field to the original image dimensions
        # The model outputs displacement for 256x256, but we need it for original HxW
        # OpenCV resize uses (W, H) for dsize
        dx_resized = cv2.resize(displacement_field_np[0], (W, H), interpolation=cv2.INTER_LINEAR)
        dy_resized = cv2.resize(displacement_field_np[1], (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Create a grid of pixel coordinates
        # Grid should be (W, H) and values normalized to [-1, 1] for F.grid_sample
        # However, for OpenCV's remap, we need absolute pixel coordinates.
        
        # Create meshgrid of coordinates (X, Y)
        # X coordinates go from 0 to W-1, Y from 0 to H-1
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        
        # Apply the displacement field to get the source coordinates
        # Source_X = Current_X + dx
        # Source_Y = Current_Y + dy
        # Note: Your displacement might be from target to source or source to target.
        # This implementation assumes the model predicts the displacement *from* the target
        # (unwarped) image *to* the source (warped) image for each pixel.
        # If it's the other way around, you might need to subtract dx/dy.
        # The typical unwarping formulation: target_coords = original_coords + displacement
        # So to get the pixel from original_coords, we need to lookup original_coords - displacement
        
        # Let's assume displacement_field_np contains (target_x - source_x, target_y - source_y)
        # To get the source coordinates from which to sample, we subtract:
        map_x = (X - dx_resized).astype(np.float32)
        map_y = (Y - dy_resized).astype(np.float32)

        # Use OpenCV's remap function for unwarping
        # INTER_LINEAR is a good general choice, INTER_CUBIC is smoother but slower.
        unwarped_img = cv2.remap(img_original, map_x, map_y, cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        return unwarped_img

    def process_and_unwarp(self, image_path):
        """
        Full pipeline from image path to unwarped image.
        """
        # 1. Preprocess image for model input
        input_tensor, original_shape = self.preprocess_image(image_path)
        
        # 2. Predict displacement field
        displacement_field = self.predict_displacement(input_tensor)
        
        # 3. Unwarp the original image
        unwarped_img = self.unwarp_image(image_path, displacement_field)
        
        return unwarped_img, displacement_field

# Example usage for prediction:
if __name__ == "__main__":
    # Ensure you have a trained model checkpoint
    MODEL_CHECKPOINT_PATH = r"checkpoints\best_model.pth" # Adjust to your actual path
    
    # Path to a sample warped image you want to unwarp
    SAMPLE_WARPED_IMAGE_PATH = r"dataset\generated_grayscale\images\000068.jpg" 
    # ^^^ Replace with an actual path to a warped image you want to test ^^^

    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}.")
        print("Please run the training script first or provide the correct path.")
    elif not os.path.exists(SAMPLE_WARPED_IMAGE_PATH):
        print(f"Error: Sample warped image not found at {SAMPLE_WARPED_IMAGE_PATH}.")
        print("Please provide a valid path to a warped image for testing.")
    else:
        # Initialize the predictor
        predictor = DocUNetPredictor(model_path=MODEL_CHECKPOINT_PATH)
        
        # Process and unwarp the image
        unwarped_image, displacement_field = predictor.process_and_unwarp(SAMPLE_WARPED_IMAGE_PATH)
        
        # Visualize results
        original_img = cv2.imread(SAMPLE_WARPED_IMAGE_PATH, cv2.IMREAD_GRAYSCALE) / 255.0

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Warped Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        # Display the X displacement field (dx)
        plt.imshow(displacement_field[0], cmap='viridis') 
        plt.title('Predicted Displacement X')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(unwarped_image, cmap='gray')
        plt.title('Unwarped Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # You can save the unwarped image
        # cv2.imwrite("unwarped_result.png", (unwarped_image * 255).astype(np.uint8))
        print("Unwarping complete. Check the displayed images.")