"""
Learning rate schedule and early stopping utilities for LT-Gate training.
"""

class LRScheduler:
    """Learning rate scheduler for multi-phase training."""
    
    def __init__(self, cfg):
        """
        Initialize scheduler with config.
        
        Args:
            cfg (dict): Configuration with lr_steps and lrs fields
        """
        self.steps = cfg.get('lr_steps', [20])  # Default: drop LR at epoch 20
        self.lrs = cfg.get('lrs', [0.005, 0.002])  # Default LRs
        
        if len(self.lrs) != len(self.steps) + 1:
            raise ValueError("Number of learning rates must be steps + 1")
    
    def get_lr(self, epoch):
        """
        Get learning rate for current epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Learning rate for this epoch
        """
        # Find which LR step we're in
        idx = sum(epoch >= step for step in self.steps)
        return self.lrs[idx]


class EarlyStopping:
    """Early stopping based on validation accuracy."""
    
    def __init__(self, cfg):
        """
        Initialize early stopping.
        
        Args:
            cfg (dict): Configuration with early_stop_delta and patience fields
        """
        self.delta = cfg.get('early_stop_delta', 0.002)  # Min improvement
        self.patience = cfg.get('patience', 5)  # Epochs to wait
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False
    
    def __call__(self, val_acc):
        """
        Check if training should stop.
        
        Args:
            val_acc (float): Current validation accuracy
            
        Returns:
            bool: True if training should stop
        """
        if val_acc - self.best_acc < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0
        
        return self.early_stop
