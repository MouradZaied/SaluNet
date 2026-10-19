# utils/ema.py
import copy
import torch


class ModelEma:
    """
    Exponential Moving Average for model parameters
    Matches the original implementation that works with decay=0.9997
    """
    def __init__(self, model, decay=0.9999):
        # Get device from model parameters
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
        
        # Create deep copy - this is what the original code does
        self.model = copy.deepcopy(model).to(device)
        self.decay = decay
        
        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.model.eval()
        
    def update(self, model):
        """
        Vectorized update - matches original implementation
        """
        with torch.no_grad():
            ema_params = list(self.model.parameters())
            model_params = list(model.parameters())
            
            # Vectorized update
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.mul_(self.decay).add_(model_p, alpha=1.0 - self.decay)
            
            # Update buffers (running stats)
            for ema_b, model_b in zip(self.model.buffers(), model.buffers()):
                ema_b.copy_(model_b)