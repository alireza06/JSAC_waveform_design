import numpy as np
import torch

class BasePulseGenerator:
    def __init__(self, t, Tb=None, T=None, device='cpu'):
        """
        Base class for pulse generators.

        Parameters:
        - t (np.ndarray or torch.Tensor): Time index.
        - Tb (float): Bit interval for communication signals.
        - T (float): Signal period for radar signals.
        - device (str): Device to use ('cpu' or 'cuda').
        """
        self.device = device
        self.t = self._to_tensor(t)
        self.Tb = Tb
        self.T = T

    def _to_tensor(self, t):
        """Convert NumPy array to PyTorch tensor if needed."""
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(self.device)
        return t
