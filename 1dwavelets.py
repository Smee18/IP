import numpy as np
import matplotlib.pyplot as plt

def get_1d_wavelets(L, k_scale=1, sigma_scale=0.8):
    psi_freq_list = []
    for k in range(k_scale):
        qs = np.fft.fftfreq(L) * L 
        # Your corrected logic: Frequency decreases as k increases
        xi = 2.5 * (2**-k)   
        sigma = sigma_scale * (2**-k)
        
        gaussian_peak = np.exp(- (qs - xi)**2 / (2 * sigma**2))
        gaussian_correction = np.exp(- (qs**2) / (2 * sigma**2)) 
        kappa = np.exp(- 0.5 * xi**2 / sigma**2)
        
        psi_freq = gaussian_peak - kappa * gaussian_correction
        psi_freq /= np.max(np.abs(psi_freq))
        psi_freq_list.append(psi_freq)
    
    return psi_freq_list

# Parameters for visualization
L_demo = 256  # High resolution for smooth plots
K_demo = 2    # 3 scales

# Generate wavelets
wavelets_freq = get_1d_wavelets(L_demo, k_scale=K_demo)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 1. Frequency Domain Plot
colors = ['r', 'g', 'b']
for k, wf in enumerate(wavelets_freq):
    # fftshift puts 0 frequency in the center for easier viewing
    wf_shifted = np.fft.fftshift(wf)
    freqs = np.fft.fftshift(np.fft.fftfreq(L_demo) * L_demo)
    
    axs[0].plot(freqs, np.abs(wf_shifted), 
                label=f'Scale k={k} (center={2.5 * 2**-k})', color=colors[k])

axs[0].set_title("Frequency Domain (Magnitude)")
axs[0].set_xlabel("Frequency (Cycles per 2π)")
axs[0].set_xlim(-5, 10) # Zoom in on the positive frequencies
axs[0].grid(True, alpha=0.3)
axs[0].legend()

# 2. Spatial/Angular Domain Plot
for k, wf in enumerate(wavelets_freq):
    # IFFT to get time domain
    w_spatial = np.fft.ifft(wf)
    # Shift so the wavelet is centered in the plot
    w_spatial = np.fft.fftshift(w_spatial)
    angles = np.linspace(-np.pi, np.pi, L_demo)
    
    axs[1].plot(angles, np.real(w_spatial), 
                label=f'Scale k={k}', color=colors[k])

axs[1].set_title("Angular Domain (Real Part)")
axs[1].set_xlabel("Angle (Radians)")
axs[1].grid(True, alpha=0.3)
axs[1].legend()

plt.tight_layout()
plt.show()