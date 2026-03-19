import numpy as np
import matplotlib.pyplot as plt
from rigid_motion_embedding import get_1d_wavelets, get_angular_phi


def plot_angular_filters(L, phi, wavelets):
    qs = np.fft.fftfreq(L) * L
    # Sort for continuous plotting
    idx = np.argsort(qs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(qs[idx], phi[idx], 'k--', label=r'Low-Pass $\hat{\phi}_{ang}$ (DC=1)', linewidth=2)
    
    for i, psi in enumerate(wavelets):
        plt.plot(qs[idx], psi[idx], label=r'High-Pass $\hat{\psi}_{k=' + str(i+1) + '}$ (DC=0)')
    
    plt.title("Angular Filter Bank in Frequency Domain ($q$)")
    plt.xlabel("Angular Frequency ($q$)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Usage:
wavelets_1d = get_1d_wavelets(L=8, K=3)
phi_ang_filter = get_angular_phi(L=8)
plot_angular_filters(8, phi_ang_filter, wavelets_1d)