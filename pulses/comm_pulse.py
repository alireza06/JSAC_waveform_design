from .base_pulse import BasePulseGenerator
import torch
import numpy as np

class CommPulseGenerator(BasePulseGenerator):
    def TDMA_pulse(self, l, k, K):
        """
        Generate Time Division Multiple Access (TDMA) pulse.

        Parameters:
        - l (int): l-th bit of the user.
        - k (int): k-th user.
        - K (int): Total number of users.

        Returns:
        - torch.Tensor: Generated TDMA pulse.
        """
        pulse = (torch.heaviside(self.t - (l + (k - 1) / K) * self.Tb, torch.tensor([1.0], device=self.device)) 
                - torch.heaviside(self.t - (l + k / K) * self.Tb, torch.tensor([0.0], device=self.device)))
        return pulse.to(torch.complex64)

    def CDMA_pulse(self, l, d):
        """
        Generate Code Division Multiple Access (CDMA) pulse.

        Parameters:
        - l (int): l-th bit of the user.
        - d (torch.Tensor or np.ndarray): CDMA code for the k-th user.

        Returns:
        - torch.Tensor: Generated CDMA pulse.
        """
        
        d = self._to_tensor(d)
        pulse = torch.zeros_like(self.t).to(self.device)
        I = len(d)
        for i in range(1, I+1):
            pulse += d[i-1] * (torch.heaviside(self.t - (l + (i - 1) / I) * self.Tb, torch.tensor([1.0], device=self.device)) 
                             - torch.heaviside(self.t - (l + i / I) * self.Tb, torch.tensor([0.0], device=self.device)))
        return pulse.to(torch.complex64)

    def OFDMA_pulse(self, l, k):
        """
        Generate Orthogonal Frequency Division Multiple Access (OFDMA) pulse.

        Parameters:
        - l (int): l-th bit of the user.
        - k (int): k-th user.

        Returns:
        - torch.Tensor: Generated OFDMA pulse.
        """
        return (torch.exp(1j * 2 * torch.pi * k / self.Tb * (self.t - l * self.Tb)) 
                * (torch.heaviside(self.t - l * self.Tb, torch.tensor([1.0], device=self.device)) 
                - torch.heaviside(self.t - (l + 1) * self.Tb, torch.tensor([0.0], device=self.device))))
