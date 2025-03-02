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
    
    def montecarlo_estimation_with_abs(self, radar_signal, sigma2, distance, channel_gain, wave_speed, montecarlo_number):
        pure_rx_signal = np.abs(channel_gain)**2 * self.make_delay(radar_signal, 2 * distance / wave_speed)
        dis = torch.zeros(montecarlo_number)
        tau = torch.zeros(montecarlo_number)
        for i in tqdm(range(montecarlo_number)):
            noise = torch.sqrt(torch.tensor(sigma2)/2) * (torch.randn_like(radar_signal) + 1j * torch.randn_like(radar_signal))
            xcorr = torch.abs(self.cross_correlation(radar_signal, pure_rx_signal + noise))
            dis[i] = self.translate_distance(xcorr, wave_speed)
            tau[i] = self.t[torch.argmax(xcorr)]
        return dis, tau
    
    def montecarlo_estimation_with_real(self, radar_signal, sigma2, distance, channel_gain, wave_speed, montecarlo_number):
        pure_rx_signal = np.abs(channel_gain)**2 * self.make_delay(radar_signal, 2 * distance / wave_speed)
        dis = torch.zeros(montecarlo_number)
        tau = torch.zeros(montecarlo_number)
        for i in tqdm(range(montecarlo_number)):
            noise = torch.sqrt(torch.tensor(sigma2)/2) * (torch.randn_like(radar_signal) + 1j * torch.randn_like(radar_signal))
            xcorr = torch.real(self.cross_correlation(radar_signal, pure_rx_signal + noise))
            dis[i] = self.translate_distance(xcorr, wave_speed)
            tau[i] = self.t[torch.argmax(xcorr)]
        return dis, tau
    
    def LFM_delayCRLB_with_real(self, B, T, sigma2, received_signal_amp):
        return 3 / 2 / np.pi**2 / B**2 / T * sigma2 * self.dt / received_signal_amp**2 / 4
    
    def LFM_delayCRLB_with_abs(self, B, T, sigma2, received_signal_amp):
        return 3 / 2 / np.pi**2 / B**2 / T * sigma2 * self.dt / received_signal_amp**2
    
    def crlb_delay(self, snr, B_eff):
        """
        Compute the CRLB for time delay estimation.
        
        Parameters:
        snr   : Signal-to-noise ratio (linear scale)
        B_eff : Effective bandwidth in Hz.
        
        Returns:
        CRLB for delay (variance in seconds^2)
        """
        return 1 / (8 * np.pi**2 * snr * B_eff**2)

    def crlb_doppler(self, snr, T_eff):
        """
        Compute the CRLB for Doppler (frequency) estimation.
        
        Parameters:
        snr   : Signal-to-noise ratio (linear scale)
        T_eff : Effective time duration in seconds.
        
        Returns:
        CRLB for Doppler (variance in (Hz)^2)
        """
        return 1 / (8 * np.pi**2 * snr * T_eff**2)

    def compute_I_tautau(self, s: torch.Tensor, 
                        A: complex or float, 
                        sigma_squared: float) -> float:
        """
        Compute I_tautau (Fisher Information for time delay τ) for a discrete signal s[n].

        Args:
            s (torch.Tensor): 1D tensor of shape [N], representing the discrete signal.
            A (complex/float): Complex amplitude of the received signal.
            sigma_squared (float): Noise variance (σ²).
            dt (float): Sampling interval (seconds).

        Returns:
            float: I_tautau value.
        """
        # Compute gradient (derivative) of s with respect to time
        s_prime = torch.gradient(s, spacing=self.dt)[0]  # s'[n] ≈ ds/dτ
        
        # Sum of squared magnitudes of the derivative, scaled by Δt
        sum_term = torch.sum(torch.abs(s_prime)**2) * self.dt
        
        # Fisher Information for τ
        I_tautau = (2 * torch.abs(torch.tensor(A))**2 / sigma_squared) * sum_term
        return I_tautau.item()