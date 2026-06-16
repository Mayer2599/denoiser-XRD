"""
Model Architectures untuk XRD Denoising
- SimpleCNN1D: Simple but fast
- UNet1D: Better performance but slower
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1D(nn.Module):
    """
    Simple CNN architecture untuk denoising
    Fast inference, moderate performance
    """
    
    def __init__(self, base_channels=32, input_length=8500):
        super(SimpleCNN1D, self).__init__()
        
        self.input_length = input_length
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        
        return x


class UNet1D(nn.Module):
    """
    UNet architecture untuk denoising
    Better performance dengan skip connections
    """
    
    def __init__(self, base_channels=32, input_length=8500):
        super(UNet1D, self).__init__()
        
        self.input_length = input_length
        
        # Encoder
        self.enc1 = self._conv_block(1, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = self._conv_block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = self._conv_block(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool1d(2)
        
        self.enc4 = self._conv_block(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_channels*16, base_channels*8)
        
        self.upconv3 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels*8, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels*4, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels*2, base_channels)
        
        # Output
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        """Convolutional block dengan BatchNorm dan ReLU"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = self._pad_to_match(x, enc4)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = self._pad_to_match(x, enc3)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = self._pad_to_match(x, enc2)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = self._pad_to_match(x, enc1)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        
        return x
    
    def _pad_to_match(self, x, target):
        """Pad x to match target size"""
        diff = target.size(2) - x.size(2)
        if diff > 0:
            x = F.pad(x, (diff // 2, diff - diff // 2))
        elif diff < 0:
            x = x[:, :, :target.size(2)]
        return x


def get_model(model_type='unet', base_channels=32, input_length=8500):
    """
    Factory function untuk create model
    
    Parameters:
    -----------
    model_type : str
        'unet' atau 'simple_cnn'
    base_channels : int
        Number of base channels
    input_length : int
        Input sequence length
    
    Returns:
    --------
    model : nn.Module
        Model instance
    """
    
    if model_type.lower() == 'unet':
        model = UNet1D(base_channels=base_channels, input_length=input_length)
        print(f"Created UNet1D model with {base_channels} base channels")
    elif model_type.lower() == 'simple_cnn':
        model = SimpleCNN1D(base_channels=base_channels, input_length=input_length)
        print(f"Created SimpleCNN1D model with {base_channels} base channels")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'unet' or 'simple_cnn'")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test models
    print("="*80)
    print("TESTING MODELS")
    print("="*80)
    
    # Test input
    batch_size = 4
    input_length = 8500
    x = torch.randn(batch_size, 1, input_length)
    
    print(f"\nTest input shape: {x.shape}")
    
    # Test SimpleCNN1D
    print("\n" + "-"*80)
    print("Testing SimpleCNN1D...")
    print("-"*80)
    model_simple = SimpleCNN1D(base_channels=32, input_length=input_length)
    
    with torch.no_grad():
        output_simple = model_simple(x)
    
    print(f"Output shape: {output_simple.shape}")
    print(f"✓ SimpleCNN1D works!")
    
    # Test UNet1D
    print("\n" + "-"*80)
    print("Testing UNet1D...")
    print("-"*80)
    model_unet = UNet1D(base_channels=32, input_length=input_length)
    
    with torch.no_grad():
        output_unet = model_unet(x)
    
    print(f"Output shape: {output_unet.shape}")
    print(f"✓ UNet1D works!")
    
    # Compare model sizes
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    simple_params = sum(p.numel() for p in model_simple.parameters())
    unet_params = sum(p.numel() for p in model_unet.parameters())
    
    print(f"\nSimpleCNN1D parameters: {simple_params:,}")
    print(f"UNet1D parameters: {unet_params:,}")
    print(f"Ratio: UNet1D is {unet_params/simple_params:.1f}x larger")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
