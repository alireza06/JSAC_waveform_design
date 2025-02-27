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
        self.dt = t[1] - t[0]

    def _to_tensor(self, t):
        """Convert NumPy array to PyTorch tensor if needed."""
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(self.device)
        return t
    
    def make_delay(self, signal, delay):
        return torch.roll(signal, shifts=int(delay / self.dt))
    
    def cross_correlation(self, base_signal, rx_signal):
        X = torch.fft.fft(base_signal)
        Y = torch.fft.fft(rx_signal)
        corr = torch.fft.ifft(Y * torch.conj(X))
        return corr[:int(len(corr)/2)]*self.dt
    
    def cross_correlation_tau(self, base_signal, rx_signal , tau):
        shifted_signal = torch.roll(base_signal, shifts=int(tau / self.dt))
        return torch.real(torch.sum(rx_signal*torch.conj(shifted_signal))*self.dt)
        # xcorr_rx = lambda tau : myRadarPulses.cross_correlation2(x, y, tau)

        # data = torch.zeros_like(delays)
        # for idx, d in tqdm(enumerate(delays)):
        #     data[idx] = xcorr_rx1(d)
