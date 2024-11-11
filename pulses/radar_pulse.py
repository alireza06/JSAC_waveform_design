from .base_pulse import BasePulseGenerator
import torch

class RadarPulseGenerator(BasePulseGenerator):
    def LFM_pulse(self, B):
        """
        Generate Linear Frequency Modulation (LFM) pulse.

        Parameters:
        - B (float): Radar signal bandwidth.

        Returns:
        - torch.Tensor: Generated LFM pulse.
        """
        return torch.exp(-1j * torch.pi * B / self.T * (self.t ** 2))

    def Gaussian_pulse(self, B):
        """
        Generate Gaussian pulse.

        Parameters:
        - B (float): Radar signal bandwidth.

        Returns:
        - torch.Tensor: Generated Gaussian pulse.
        """
        a = torch.exp(-1 * B / self.T * ((self.t - self.T / 2) ** 2))
        return a.to(torch.complex64)

    def Barker_pulse(self, b):
        """
        Generate Barker-coded pulse.

        Parameters:
        - b (torch.Tensor or np.ndarray): Barker code.
        
        Returns:
        - torch.Tensor: Generated Barker-coded pulse.
        """
        b = self._to_tensor(b)
        pulse = torch.zeros_like(self.t).to(self.device)
        M = len(b)
        for m in range(M):
            pulse += b[m] * (torch.heaviside(self.t - m * self.T / M, torch.tensor([0.5], device=self.device)) 
                             - torch.heaviside(self.t - m * self.T / M - self.T / M, torch.tensor([0.5], device=self.device)))
        return pulse.to(torch.complex64)
