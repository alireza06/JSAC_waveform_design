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
        # shifts=int(delay / self.dt)
        # ret_signal = torch.zeros_like(signal)
        # if shifts < len(signal):
        #     ret_signal[shifts:] = signal[:len(signal) - shifts]
        # return ret_signal
        return torch.roll(signal, shifts=int(delay / self.dt))
    
    def make_delay_doppler(self, signal, delay, doppler):
        # shifts=int(delay / self.dt)
        # ret_signal = torch.zeros_like(signal)
        # if shifts < len(signal):
        #     ret_signal[shifts:] = signal[:len(signal) - shifts]
        # return ret_signal
        return torch.roll(signal * torch.exp(1j * 2 * np.pi * doppler * self.t), shifts=int(delay / self.dt))
    
    def cross_correlation(self, base_signal, rx_signal):
        X = torch.fft.fft(base_signal)
        Y = torch.fft.fft(rx_signal)
        corr = torch.fft.ifft(Y * torch.conj(X))
        return torch.fft.fftshift(corr)*self.dt
    
    def cross_correlation_tau(self, base_signal, rx_signal , tau):
        shifted_signal = torch.roll(base_signal, shifts=int(tau / self.dt))
        return torch.real(torch.sum(rx_signal*torch.conj(shifted_signal))*self.dt)
        # xcorr_rx = lambda tau : myRadarPulses.cross_correlation2(x, y, tau)

        # data = torch.zeros_like(delays)
        # for idx, d in tqdm(enumerate(delays)):
        #     data[idx] = xcorr_rx1(d)

    # Fast Ambiguity Function via vectorization
    def fast_ambiguity(self, rx_signal, base_signal):
        N = rx_signal.shape[0]
        max_delay_samples = N // 2  # Maximum delay in samples
        delays = torch.arange(-max_delay_samples, max_delay_samples + 1, device=self.device)
        # Create shifted versions of s for all delays (vectorized)
        idx = (torch.arange(N, device=self.device)[None, :] - delays[:, None]) % N
        base_signal_shifted = base_signal[idx]  # Shape: (num_delays, N)
        
        # Compute r(t) * s^*(t - τ) for all τ
        product = rx_signal[None, :] * base_signal_shifted.conj()  # Element-wise multiply
        
        # FFT along time axis to get Doppler dimension
        ambiguity = torch.fft.fftshift(torch.fft.fft(product, dim=1), dim=1)
        self.doppler_bins = torch.fft.fftshift(torch.fft.fftfreq(N, d=self.dt)).to(self.device)
        return ambiguity


    def compute_Teff(self, signal):
        """
        Compute the effective time duration T_eff for a signal s(t).
        
        Parameters:
        signal : 1D numpy array containing s(t).
        
        Returns:
        T_eff : Effective time duration in seconds.
        """
        energy = torch.sum(torch.abs(signal)**2) * self.dt
        T2 = torch.sum(self.t**2 * torch.abs(signal)**2) * self.dt / energy
        return torch.sqrt(T2)
    
    def compute_Beff(self, signal):
        """
        Compute the effective bandwidth B_eff for a signal s(t).
        The Fourier transform S(ω) is computed numerically.
        
        Parameters:
        signal : 1D numpy array containing s(t).
        
        Returns:
        B_eff : Effective bandwidth in Hertz.
        """
        
        # Compute Fourier Transform. Using fftshift to center the zero frequency.
        S = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(signal)))
        
        # Frequency axis in Hz:
        freq = torch.fft.fftshift(torch.fft.fftfreq(len(self.t), d=self.dt)).to(self.device)
        # Convert to angular frequency (rad/s)
        omega = 2 * np.pi * freq
        domega = omega[1] - omega[0]
        
        # Energy in frequency domain (using Parseval's theorem)
        energy_freq = torch.sum(torch.abs(S)**2) * domega
        # Second moment of the frequency distribution
        omega2 = torch.sum(omega**2 * torch.abs(S)**2) * domega / energy_freq
        Omega_eff = torch.sqrt(omega2)
        
        # Convert angular frequency spread to effective bandwidth in Hz
        B_eff = Omega_eff / (2 * np.pi)
        return B_eff
    
    def LOS_pathloss_db(self, distance, carrier_frequency, wave_speed):
        FSPL_db = 20*np.log10(distance) + 20*np.log10(carrier_frequency) + 20*np.log10(4*np.pi/wave_speed)
        return -FSPL_db
    
    def LOS_pathloss(self, distance, carrier_frequency, wave_speed):
        return 10**(self.LOS_pathloss_db(distance, carrier_frequency, wave_speed)/10)
    
    def power_calc(self, signal):
        return torch.sum(torch.abs(signal)**2)*self.dt