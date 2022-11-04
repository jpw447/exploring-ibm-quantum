import numpy as np
import matplotlib.pyplot as plt
from gaussian_pulse_funcs import gauss_probability_calculator

'''
CELL 1
This cell reads in the data from a specified frequency and data file. Copying the
code from Qiskit chapter 6.1, it creates a function to fit a Lorentzian to the data
and provide an evaluation of the transition frequenyc, in GHz (rough_qubit_frequency).

The data and Lorentzian are then plotted.
'''

frequencies = np.loadtxt("Data//v=0.1, 2048 shots, 0.5 step gaussian frequencies.txt", delimiter=",")
qiskit_gauss_data = np.loadtxt("Data//v=0.1, 2048 shots, 0.5 step gaussian data.txt", delimiter=",")

'''Qiskit code to find qubit frequency'''
from scipy.optimize import curve_fit

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

fit_params, y_fit = fit_function(frequencies,
                                 qiskit_gauss_data, 
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [1, 4.975, 1, -2] # initial parameters for curve_fit
                                )

a, q, b, c = fit_params
rough_qubit_frequency = q*1e9 
'''End of Qiskit code'''

# converting frequencies to be dimensionless
dimensionless_frequencies = frequencies/(rough_qubit_frequency/1e9)

# scaling the data  and curve fit to make the maximum occur at 1 and minimum at 0.
# first translating and then scaling
diff = 0 - np.min(qiskit_gauss_data)
qiskit_gauss_data += diff
y_fit += diff
scale_factor = 0.05/np.max(qiskit_gauss_data)
qiskit_gauss_data *= scale_factor
y_fit *= scale_factor

# plots
fig_qiskit = plt.figure(figsize=(8,6))
ax_qiskit = fig_qiskit.gca()
ax_qiskit.plot(dimensionless_frequencies, qiskit_gauss_data, "ko")
ax_qiskit.plot(dimensionless_frequencies, y_fit, color='red')
ax_qiskit.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
ax_qiskit.set_ylabel("Signal Strength", fontsize=16)



#%%
'''
CELL 2
Performs a frequency sweep of given parameters V_prime, sigma_prime and maps
over Qiskit data. Also calculates the area for the pulse for each frequency.
'''
from scipy.integrate import quad
plt.close()

# defining the gaussian envelope function here
def envelope_function(t, v, t0, sigma):
    return v*np.exp(-((t)-(t0))**2/(2*(sigma)**2))

# variables
sigma_prime = 74.4324
V_prime = 0.05 * np.pi/9.328127015269782 # scales area to pi for pi-pulse
t0_prime = 10*sigma_prime
initial_conditions = [1,0,0,0]
num_points = 10000
tmax = t0_prime + 4*sigma_prime
tmin = t0_prime - 4*sigma_prime

# time array and when to measure probability (after cutoff)
t_prime_array = np.linspace(0, 20*sigma_prime, num_points)
cutoff = int(np.argwhere(t_prime_array >= tmax)[0])

# creating arrays for probability, area and frequency
N = 500
freq_max = np.zeros(N)
area = np.zeros(N)
frequency_array = np.linspace(0.9959918258322906, 1.0040528328693592, N)

# looping through each frequency to get post-pulse probability and area
for i, omega_prime in enumerate(frequency_array):        
    p1 = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)[1]
    freq_max[i] = np.mean(p1[cutoff:])
    area[i] = quad(envelope_function, tmin, tmax, args=(V_prime, t0_prime, sigma_prime))[0]

# we're reloading the data and scaling it to make the maximum coincide with the
# frequency sweep maximum
# qiskit_gauss_data = np.loadtxt("Data//v=0.1, 2048 shots, 0.5 step gaussian data.txt", delimiter=",")
scale_factor = np.max(freq_max)/np.max(qiskit_gauss_data)
rescaled_data = qiskit_gauss_data *  scale_factor
rescaled_fit = y_fit * scale_factor

# plot to compare probabilities
fig_compare = plt.figure(figsize=(8,6))
ax_compare = fig_compare.gca()
ax_compare.plot(dimensionless_frequencies, rescaled_data, "ko", label="Qiskit Data")
ax_compare.plot(frequency_array, freq_max, "b--", label="Simulation")
ax_compare.plot(dimensionless_frequencies, rescaled_fit, color='red', label="SciPy Curve Fit")
ax_compare.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
ax_compare.set_ylabel("Signal/Post-Pulse Probability", fontsize=16)
ax_compare.legend(fontsize=12)

# plots area vs frequency (which is obviously constant)
fig_area = plt.figure(figsize=(8,6))
ax_area = fig_area.gca()
ax_area.plot(frequency_array, area, "k")
ax_area.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
ax_area.set_ylabel("Pulse Area", fontsize=16)

#%%
'''
CELL 3
Fitting the perturbation theory prediction to Qiskit using SciPy curve_fit.
Uses the values for V and tmax (T) used by Qiskit as initial guesses
Also compares areas
'''
from scipy.integrate import quad

plt.close()

'''
Curve fitting function to full perturbation theory equation. This is because
SciPy might throw a few errors or improperly fit the Taylor series to the data
if you try to use the Taylor series. I didn't try this out.
'''
def func(x, amp, time):
    return amp**2 * np.sin(0.5*time*(1-x))**2/(1-x)**2

# variables
sigma_prime = 74.4324
V_prime = 0.05 * np.pi/9.328127015269782
duration = 8*sigma_prime

# curve fitting guessing based on Qiskit parameters
param_q, _ = curve_fit(func, dimensionless_frequencies, qiskit_gauss_data, [0.05, duration])
V, T = param_q
f = func(dimensionless_frequencies, V, T)

# truncating frequencies because Taylor series goes very negative otherwise
f_min = 0.998
f_max = 1.002
argmin = np.min(np.argwhere(dimensionless_frequencies >= f_min))
argmax = np.min(np.argwhere(dimensionless_frequencies >= f_max))
truncated_frequencies = dimensionless_frequencies[argmin:argmax] 
taylor = V**2 * T**2 / 4 * (1-((0.5*T*(1-truncated_frequencies))**2)/3)

def RWA_fit(x, amp, time):
    return amp**2/((1-x)**2 + amp**2) * np.sin(0.5*time*np.sqrt((1-x)**2+amp**2))**2

param_RWA, _ = curve_fit(RWA_fit, dimensionless_frequencies, qiskit_gauss_data, [0.05, duration])
V_RWA, T_RWA = param_q

RWA = RWA_fit(dimensionless_frequencies, V_RWA, T_RWA)

# plots
fig_pt = plt.figure(figsize=(8,6))
ax_pt = fig_pt.gca()    
ax_pt.plot(dimensionless_frequencies, qiskit_gauss_data, "ko", label="Qiskit Data")
ax_pt.plot(dimensionless_frequencies, f, "b", label="Perturbation Theory")
ax_pt.plot(truncated_frequencies, taylor, "r--", label="Taylor Series")
# ax_pt.plot(dimensionless_frequencies, RWA, "g--", label="RWA")
ax_pt.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
ax_pt.set_ylabel("Signal/Post-Pulse Probability", fontsize=16)
ax_pt.legend(fontsize=12)