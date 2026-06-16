"""
Training Monitor - Real-time monitoring dengan plotting
Optional TensorBoard integration
"""

import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import numpy as np


class TrainingMonitor:
    """Monitor training progress dengan live plotting"""
    
    def __init__(self, log_dir, use_tensorboard=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
                print("✓ TensorBoard logging enabled")
                print(f"  View with: tensorboard --logdir {self.log_dir / 'tensorboard'}")
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
    
    def log_epoch(self, epoch, train_loss, val_loss, learning_rate):
        """Log epoch metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        
        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
    
    def log_batch(self, epoch, batch_idx, batch_loss, num_batches):
        """Log batch metrics"""
        if self.writer is not None:
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Loss/batch', batch_loss, global_step)
    
    def plot_live(self, save_path=None):
        """Create live training plot"""
        if len(self.epochs) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / 'training_progress.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_history(self, filepath=None):
        """Save training history to JSON"""
        if filepath is None:
            filepath = self.log_dir / 'training_history.json'
        
        history = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_history(self, filepath):
        """Load training history from JSON"""
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        self.epochs = history['epochs']
        self.train_losses = history['train_losses']
        self.val_losses = history['val_losses']
        self.learning_rates = history.get('learning_rates', [])
    
    def check_anomaly(self):
        """Check for training anomalies"""
        anomalies = []
        
        if len(self.train_losses) < 2:
            return anomalies
        
        # Check for NaN
        if np.isnan(self.train_losses[-1]) or np.isnan(self.val_losses[-1]):
            anomalies.append("WARNING: NaN detected in losses!")
        
        # Check for exploding loss
        if len(self.train_losses) >= 5:
            recent_avg = np.mean(self.train_losses[-5:])
            if recent_avg > self.train_losses[0] * 10:
                anomalies.append("WARNING: Loss is exploding!")
        
        # Check for no improvement
        if len(self.val_losses) >= 10:
            recent_best = min(self.val_losses[-10:])
            overall_best = min(self.val_losses)
            if recent_best > overall_best * 1.5:
                anomalies.append("WARNING: Validation loss not improving!")
        
        # Check for overfitting
        if len(self.train_losses) >= 5 and len(self.val_losses) >= 5:
            train_avg = np.mean(self.train_losses[-5:])
            val_avg = np.mean(self.val_losses[-5:])
            if val_avg > train_avg * 1.5:
                anomalies.append("WARNING: Possible overfitting detected!")
        
        return anomalies
    
    def print_status(self):
        """Print current training status"""
        if len(self.epochs) == 0:
            print("No training data yet")
            return
        
        print("\n" + "="*80)
        print("TRAINING STATUS")
        print("="*80)
        print(f"Current epoch: {self.epochs[-1]}")
        print(f"Training loss: {self.train_losses[-1]:.6f}")
        print(f"Validation loss: {self.val_losses[-1]:.6f}")
        print(f"Learning rate: {self.learning_rates[-1]:.6f}")
        
        if len(self.val_losses) > 1:
            best_val = min(self.val_losses)
            print(f"Best validation loss: {best_val:.6f} (epoch {self.val_losses.index(best_val) + 1})")
        
        # Check for anomalies
        anomalies = self.check_anomaly()
        if anomalies:
            print("\nANOMALIES DETECTED:")
            for anomaly in anomalies:
                print(f"  {anomaly}")
        
        print("="*80)
    
    def close(self):
        """Close resources"""
        if self.writer is not None:
            self.writer.close()


# Example usage
if __name__ == "__main__":
    # Test monitor
    monitor = TrainingMonitor(log_dir='logs/test', use_tensorboard=False)
    
    # Simulate training
    print("Simulating training...")
    for epoch in range(1, 51):
        # Simulate losses (decreasing trend with noise)
        train_loss = 0.1 * np.exp(-epoch/20) + np.random.normal(0, 0.01)
        val_loss = 0.12 * np.exp(-epoch/20) + np.random.normal(0, 0.015)
        lr = 0.001 * (0.95 ** (epoch // 10))
        
        monitor.log_epoch(epoch, train_loss, val_loss, lr)
        
        if epoch % 10 == 0:
            monitor.plot_live()
            monitor.print_status()
            time.sleep(0.5)
    
    # Save final results
    monitor.save_history()
    monitor.plot_live()
    
    print("\n✓ Monitor test completed!")
    print(f"Check logs in: logs/test/")
