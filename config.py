"""
Configuration file untuk XRD Denoising Training
Semua hyperparameters dan paths
"""

import os
from pathlib import Path

class Config:
    """Configuration class untuk training"""
    
    # ============================================================================
    # DATA PATHS
    # ============================================================================
    # Base directory (sesuaikan dengan struktur folder Anda)
    BASE_DIR = Path("C:/Users/Lenovo/Documents/xrdAI_withoutmatch3_v2")
    DATA_DIR = BASE_DIR / "data"
    
    # Processed data directories
    TRAIN_CLEAN_DIR = DATA_DIR / "processed" / "train" / "clean"
    TRAIN_NOISY_DIR = DATA_DIR / "processed" / "train" / "noisy"
    VAL_CLEAN_DIR = DATA_DIR / "processed" / "val" / "clean"
    VAL_NOISY_DIR = DATA_DIR / "processed" / "val" / "noisy"
    
    # Output directories
    SAVE_DIR = BASE_DIR / "models" / "saved"
    LOG_DIR = BASE_DIR / "logs"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    MODEL_TYPE = "unet"  # Options: "unet", "simple_cnn"
    BASE_CHANNELS = 32   # Base number of channels (32, 64, 128)
    INPUT_LENGTH = 8500  # Length of XRD data
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    BATCH_SIZE = 16          # Batch size (8, 16, 32)
    LEARNING_RATE = 0.001    # Learning rate (0.001, 0.0005, 0.0001)
    WEIGHT_DECAY = 1e-5      # L2 regularization
    EPOCHS = 100             # Total training epochs
    
    # Scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "step"  # Options: "step", "cosine", "plateau"
    SCHEDULER_STEP_SIZE = 20  # For StepLR
    SCHEDULER_GAMMA = 0.5     # For StepLR
    
    # Early stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for N epochs
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================
    NUM_WORKERS = 4          # Number of data loading workers
    PIN_MEMORY = True        # Pin memory for GPU
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    DEVICE = "cuda"  # Options: "cuda", "cpu"
    
    # ============================================================================
    # LOGGING & CHECKPOINTING
    # ============================================================================
    LOG_INTERVAL = 10        # Print every N batches
    SAVE_INTERVAL = 5        # Save checkpoint every N epochs
    SAVE_BEST_ONLY = True    # Only save best model
    
    # ============================================================================
    # LOSS FUNCTION
    # ============================================================================
    LOSS_TYPE = "mse"        # Options: "mse", "mae", "huber"
    
    # ============================================================================
    # TEST MODE (untuk debugging)
    # ============================================================================
    TEST_MODE = False        # Use subset of data for testing
    TEST_SAMPLES = 1000      # Number of samples in test mode
    
    # ============================================================================
    # METHODS
    # ============================================================================
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.SAVE_DIR,
            cls.LOG_DIR,
            cls.CHECKPOINT_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ All directories created")
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        print("="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        
        print("\n[DATA PATHS]")
        print(f"  Base dir: {cls.BASE_DIR}")
        print(f"  Train clean: {cls.TRAIN_CLEAN_DIR}")
        print(f"  Train noisy: {cls.TRAIN_NOISY_DIR}")
        print(f"  Val clean: {cls.VAL_CLEAN_DIR}")
        print(f"  Val noisy: {cls.VAL_NOISY_DIR}")
        
        print("\n[MODEL]")
        print(f"  Type: {cls.MODEL_TYPE}")
        print(f"  Base channels: {cls.BASE_CHANNELS}")
        print(f"  Input length: {cls.INPUT_LENGTH}")
        
        print("\n[TRAINING]")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Weight decay: {cls.WEIGHT_DECAY}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Early stopping: {cls.EARLY_STOPPING} (patience: {cls.EARLY_STOPPING_PATIENCE})")
        
        print("\n[SCHEDULER]")
        print(f"  Use scheduler: {cls.USE_SCHEDULER}")
        if cls.USE_SCHEDULER:
            print(f"  Type: {cls.SCHEDULER_TYPE}")
            if cls.SCHEDULER_TYPE == "step":
                print(f"  Step size: {cls.SCHEDULER_STEP_SIZE}")
                print(f"  Gamma: {cls.SCHEDULER_GAMMA}")
        
        print("\n[DATA LOADING]")
        print(f"  Num workers: {cls.NUM_WORKERS}")
        print(f"  Pin memory: {cls.PIN_MEMORY}")
        
        print("\n[DEVICE]")
        print(f"  Device: {cls.DEVICE}")
        
        print("\n[OUTPUT]")
        print(f"  Save dir: {cls.SAVE_DIR}")
        print(f"  Log dir: {cls.LOG_DIR}")
        print(f"  Checkpoint dir: {cls.CHECKPOINT_DIR}")
        
        if cls.TEST_MODE:
            print("\n[WARNING] TEST MODE ENABLED!")
            print(f"  Using only {cls.TEST_SAMPLES} samples")
        
        print("="*80)
    
    @classmethod
    def get_dict(cls):
        """Get configuration as dictionary"""
        config_dict = {
            'model_type': cls.MODEL_TYPE,
            'base_channels': cls.BASE_CHANNELS,
            'input_length': cls.INPUT_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'epochs': cls.EPOCHS,
            'early_stopping': cls.EARLY_STOPPING,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'scheduler': cls.USE_SCHEDULER,
            'scheduler_type': cls.SCHEDULER_TYPE,
            'loss_type': cls.LOSS_TYPE,
        }
        return config_dict


# Alternative: YAML-based config (jika prefer YAML)
def save_config_yaml(config, filepath="config.yaml"):
    """Save configuration to YAML file"""
    import yaml
    
    config_dict = config.get_dict()
    
    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"✓ Configuration saved to {filepath}")


def load_config_yaml(filepath="config.yaml"):
    """Load configuration from YAML file"""
    import yaml
    
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update Config class
    for key, value in config_dict.items():
        key_upper = key.upper()
        if hasattr(Config, key_upper):
            setattr(Config, key_upper, value)
    
    print(f"✓ Configuration loaded from {filepath}")
    return config_dict


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    
    # Display config
    Config.display()
    
    # Create directories
    print("\nCreating directories...")
    Config.create_directories()
    
    # Get config dict
    config_dict = Config.get_dict()
    print(f"\nConfiguration dictionary: {config_dict}")
    
    print("\n✓ Configuration test completed!")
