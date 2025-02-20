from .base_pulse import BasePulseGenerator
import torch
import numpy as np
from tqdm import tqdm


class RadarPulseGenerator(BasePulseGenerator):
    def LFM_pulse(self, B):
        """
        Generate Linear Frequency Modulation (LFM) pulse.

        Parameters:
        - B (float): Radar signal bandwidth.

        Returns:
        - torch.Tensor: Generated LFM pulse.
        """
        return torch.exp(1j * torch.pi * B / self.T * (self.t ** 2))

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

    def compute_ambiguity_function(self, s_t, delays, doppler_shifts):
            """
            Compute the ambiguity function for a given signal.

            Parameters:
            - s_t (torch.Tensor): Input signal s(t).
            - delays (torch.Tensor): Array of time delays (tau).
            - doppler_shifts (torch.Tensor): Array of Doppler shifts (f_d).

            Returns:
            - torch.Tensor: 2D ambiguity function values.
            """
            ambiguity = torch.zeros((len(doppler_shifts), len(delays)), dtype=torch.complex64, device=self.device)
            s_t_star = torch.conj(s_t)

            for i, f_d in enumerate(doppler_shifts):
                # Multiply by the Doppler shift term
                doppler_term = torch.exp(1j * 2 * torch.pi * f_d * self.t)
                for j, tau in enumerate(delays):
                    # Shift the signal by tau
                    shifted_signal = torch.roll(s_t_star, shifts=int(tau / self.dt))
                    # Compute the integral
                    product = s_t * shifted_signal * doppler_term
                    ambiguity[i, j] = torch.sum(product)*self.dt

            return torch.abs(ambiguity)
    
    def translate_distance(self, xcorr, wave_speed):
         return self.t[torch.argmax(xcorr)] * wave_speed / 2
    
    def montecarlo_estimation(self, radar_signal, sigma2, distance, channel_gain, wave_speed, montecarlo_number):
        tau = 2 * distance / wave_speed
        pure_rx_signal = np.abs(channel_gain)**2 * self.make_delay(radar_signal, tau)
        data = torch.zeros(montecarlo_number)
        for i in tqdm(range(montecarlo_number)):
            xcorr = self.cross_correlation(radar_signal, pure_rx_signal + np.sqrt(sigma2)*torch.randn(len(radar_signal), device=self.device))
            data[i] = self.translate_distance(xcorr, wave_speed)
        return data
