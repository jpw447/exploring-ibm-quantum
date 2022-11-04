import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def gaussian(t, mean, amp, std):
    return amp * np.exp(-(t-mean)**2/(2*std**2))

omega = 4.96216 * 1e9           # transition frequency in Hz
V = 0.05                        # qiskit amplitude
sigma = 0.015 * 1e-6 * omega    # dimensionless standard deviation 
t0 = 4*sigma                    # mean of gaussian

t_array = np.linspace(0, 8*sigma)

area = quad(gaussian, np.min(t_array), np.max(t_array), args=(t0, V, sigma))[0]

f = gaussian(t_array, t0, V, sigma)

plt.plot(t_array, f)

print("Area was determined to be "+str(area))


