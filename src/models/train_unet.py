"""U-Net Training Script for Image Segmentation.

This module implements a U-Net convolutional neural network for image segmentation
tasks. It loads preprocessed data, trains the model, and saves checkpoints after
each epoch.

The U-Net architecture consists of an encoder (contracting path), bottleneck, and
decoder (expanding path) with skip connections between corresponding encoder and
decoder layers.

Typical usage example:
    python train_unet.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm   # Progress bar for training iterations


# -------------------------------------------------------------
# UNET MODEL
# -------------------------------------------------------------
class UNet(nn.Module):
    """U-Net architecture for image segmentation.
    
    U-Net is a fully convolutional network consisting of a contracting path
    (encoder) to capture context and an expansive path (decoder) to enable
    precise localization. Skip connections between encoder and decoder help
    preserve spatial information.
    
    Attributes:
        enc1: First encoder block (64 channels).
        pool1: First max pooling layer (2x2).
        enc2: Second encoder block (128 channels).
        pool2: Second max pooling layer (2x2).
        enc3: Third encoder block (256 channels).
        pool3: Third max pooling layer (2x2).
        bottleneck: Bottleneck layer (512 channels).
        up3: Third upsampling layer (transpose convolution).
        dec3: Third decoder block (256 channels).
        up2: Second upsampling layer (transpose convolution).
        dec2: Second decoder block (128 channels).
        up1: First upsampling layer (transpose convolution).
        dec1: First decoder block (64 channels).
        out: Final 1x1 convolution to produce output channels.
    """
    
    def __init__(self, in_channels=7, out_channels=1):
        """Initializes the U-Net model.
        
        Args:
            in_channels: Number of input channels (default: 7).
            out_channels: Number of output channels (default: 1).
        """
        super().__init__()

        def conv_block(in_c, out_c):
            """Creates a convolutional block with two conv layers and ReLU activations.
            
            Each block consists of:
                - 3x3 convolution with padding
                - ReLU activation (in-place)
                - 3x3 convolution with padding
                - ReLU activation (in-place)
            
            Args:
                in_c: Number of input channels.
                out_c: Number of output channels.
                
            Returns:
                nn.Sequential: A sequential container of the conv block layers.
            """
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder path (contracting)
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder path (expansive)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)  # 512 due to skip connection concatenation

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)  # 256 due to skip connection concatenation

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)   # 128 due to skip connection concatenation

        # Final output layer
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder path with skip connection storage
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder path with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # Concatenate along channel dimension

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
EPOCHS = 5          # Number of training epochs
BATCH_SIZE = 1      # Batch size (full 1024×1024 images per batch)
LR = 1e-4           # Learning rate for Adam optimizer

# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
# Construct path to processed data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
X = np.load(os.path.join(DATA_DIR, "X.npy"))   # Input features: (N, 7, H, W)
y = np.load(os.path.join(DATA_DIR, "y.npy"))   # Target labels: (N, H, W)

print("Loaded dataset:")
print("X:", X.shape, " y:", y.shape)

# Split data into training and validation sets (80/20 split)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Create PyTorch datasets and data loaders
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)


# -------------------------------------------------------------
# MODEL, OPTIMIZER, LOSS
# -------------------------------------------------------------
# Initialize model and move to device
model = UNet().to(DEVICE)

# Adam optimizer with specified learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Mean Squared Error loss for regression task
criterion = nn.MSELoss()

# Create checkpoint directory if it doesn't exist
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------------------------------------------
# TRAINING LOOP
# -------------------------------------------------------------
print("\nStarting training...\n")

for epoch in range(1, EPOCHS + 1):
    # Set model to training mode (enables dropout, batch norm updates, etc.)
    model.train()
    train_losses = []

    # Training phase with progress bar
    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", ncols=90):
        # Move batch to device
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Forward pass
        out = model(Xb)
        
        # Compute loss
        loss = criterion(out, yb)
        
        # Backward pass (compute gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()

        # Store loss for averaging
        train_losses.append(loss.item())

    # Validation phase
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    val_losses = []

    # Disable gradient computation for validation
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            val_loss = criterion(out, yb)
            val_losses.append(val_loss.item())

    # Calculate average losses
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)

    # Print epoch summary
    print(f"\nEpoch {epoch}/{EPOCHS}  |  Train Loss: {avg_train:.6f}  |  Val Loss: {avg_val:.6f}\n")

    # Save model checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"unet_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}\n")


print("Training complete.")
print("Checkpoints stored in:", CHECKPOINT_DIR)
