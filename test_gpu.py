import cupy as cp

try:
    x = cp.array([1, 2, 3])
    print("Success! Pascal (1050 Ti) is initialized.")
    y = cp.fft.fft(x)
    print("FFT Test: Passed.")
except Exception as e:
    print(f"Error: {e}")