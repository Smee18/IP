from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from scipy.fft import fft2

# FIX 1: Use a power of 2 for image size.
# J=3 means we downsample by 2^3 = 8. M must be divisible by 8.

M = 32
J = 2
L = 8

filters_set = filter_bank(M, M, J, L=L)


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c

# FIX 2: Disable LaTeX requirement
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

fig, axs = plt.subplots(J, L, sharex=True, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(6)

i = 0
# Loop through the 'psi' (wavelet) filters
for filter in filters_set['psi']:
    f = filter["levels"][0]
    filter_c = fft2(f)
    filter_c = np.fft.fftshift(filter_c)
    
    # Handle the subplot indexing
    ax = axs[i // L, i % L]
    ax.imshow(colorize(filter_c))
    ax.axis('off')
    ax.set_title(f"j={i // L}, theta={i % L}", fontsize=8) # Simplified title
    i = i+1

fig.suptitle("Wavelets (Psi) in Frequency Domain\n(Hue=Phase, Brightness=Magnitude)", fontsize=13)

# FIX 3: Use plt.show() to ensure the window opens
plt.tight_layout()
plt.show()

# --- Optional: Visualize the Low Pass Filter (Phi) ---
plt.figure()
plt.axis('off')

# Grab the low pass filter
f = filters_set['phi']["levels"][0]

filter_c = fft2(f)
filter_c = np.fft.fftshift(filter_c)
plt.title("Low-pass Filter (Phi)", fontsize=13)

# We usually just view the magnitude for the low pass
plt.imshow(np.abs(filter_c), cmap='inferno')
plt.show()