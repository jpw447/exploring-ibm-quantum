import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import curve_fit
from gaussian_pulse_funcs import gauss_envelope_function, gauss_probability_calculator

'''
This solves the perturbation ODEs for a Gaussian-modulated cosine perturbation
for a given frequency and plots the transition probabilities and the state
at time t, given these probabilities. It also plots the perturbation signal.

The simulation runs from t = μ - 5σ to t = μ + 5σ.
'''

#%%
'''
Performs a gaussian pulse of varying frequencies to provide a frequency sweep.
The area is fixed to new_area = pi
'''
from gaussian_pulse_funcs import area_probability_calculator
if __name__ == "__main__":
    plt.close()
    # variables
    omega_prime = 1
    sigma_prime = 2
    t0_prime = 10
    initial_conditions = [1,0,0,0]
    num_points = 10000
    V_prime = 0.01
    tmax = t0_prime + 6*sigma_prime
    
    # running simulation for t0 + 6 sigma, and then measuring post-pulse probability
    # 4 sigma after mean position of pulse
    t_prime_array = np.linspace(0, tmax, num_points)
    cutoff = int(np.argwhere(t_prime_array >= t0_prime+4*sigma_prime)[0])
    
    # fixing the area of the Gaussian pules to be pi
    new_area = np.pi/10
    
    # creating arrays for probability and area
    N = 500
    freq_max = np.zeros(N)
    area = np.zeros(N)
    frequency_array = np.linspace(0, 2, N)
    for i, omega_prime in enumerate(frequency_array):        
        p1 = area_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions, new_area)[1]
        
        # post-pulse probability
        freq_max[i] = np.mean(p1[cutoff:])
       
    # plots
    fig_single = plt.figure(figsize=(8,6))
    ax_single = fig_single.gca()
    ax_single.plot(frequency_array, freq_max, "k")
    ax_single.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_single.set_ylabel("Post-Pulse Probability", fontsize=16)
    
    # comparing the result to perturbation theory prediction
    tmax = new_area/V_prime
    p_pt = V_prime**2  * (np.sin(0.5*tmax*(1-frequency_array))**2)/(1-frequency_array)**2
    ax_single.plot(frequency_array, p_pt, "r--")
    
#%%
'''Identical to the cell above, but it sweeps over a smaller frequency range to
test perturbation theory. The same parameters are used, and then SciPy is used
to fit the Taylor series to the curve. '''
from gaussian_pulse_funcs import area_probability_calculator
from scipy.optimize import curve_fit
if __name__ == "__main__":
    plt.close()
    # variables
    omega_prime = 1
    sigma_prime = 2
    t0_prime = 10
    initial_conditions = [1,0,0,0]
    num_points = 10000
    V_prime = 0.01
    tmax = t0_prime + 6*sigma_prime
    
    t_prime_array = np.linspace(0, tmax, num_points)
    cutoff = int(np.argwhere(t_prime_array >= t0_prime+4*sigma_prime)[0])
    
    # fixing the area of the Gaussian pules to be pi
    new_area = np.pi/100
    
    # creating arrays for probability and area
    N = 500
    freq_max = np.zeros(N)
    area = np.zeros(N)
    frequency_array = np.linspace(0.9, 1.1, N)
    for i, omega_prime in enumerate(frequency_array):        
        p1 = area_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions, new_area)[1]
        
        freq_max[i] = np.mean(p1[cutoff:])
       

    fig_single = plt.figure(figsize=(8,6))
    ax_single = fig_single.gca()
    ax_single.plot(frequency_array, freq_max, "k", label="Gaussian Sweep")
    ax_single.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_single.set_ylabel("Post-Pulse Probability", fontsize=16)
    
    # comparing the result to perturbation theory prediction
    tmax = new_area/V_prime
    p_pt = V_prime**2  * (np.sin(0.5*tmax*(1-frequency_array))**2)/(1-frequency_array)**2
    taylor = (V_prime**2 * tmax**2)/4 * (1- (0.5*tmax*(1-frequency_array))**2/3)
    interval = int(N/20)
    ax_single.plot(frequency_array[::interval], p_pt[::interval], "ro", label="Perturbation Theory")
    ax_single.plot(frequency_array, taylor, "b--", label="Taylor Series")
    ax_single.legend(fontsize=12)
    
    ''' Using curve fitting to find the right parameters that we can use to
    use perturbation theory to predict the curve'''
    def func(x, amp, time):
        return amp**2 * np.sin(0.5*time*(1-x))**2/(1-x)**2
    
    param, _ = curve_fit(func, frequency_array, freq_max)
    V, T = param
    new_taylor = (V**2 * T**2)/4 * (1- (0.5*T*(1-frequency_array))**2/3)
    
    fig_taylor = plt.figure(figsize=(8,6))
    ax_taylor = fig_taylor.gca()
    ax_taylor.plot(frequency_array, freq_max, "k", label="Gaussian Sweep")
    ax_taylor.plot(frequency_array, new_taylor, "b--", label="Taylor Series")
    ax_taylor.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_taylor.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_taylor.legend(fontsize=12)

#%%
'''
Amplitude sweeping. This was never ultimately used in the report
'''

if __name__ == "__main__":
    plt.close()
    # variables
    sigma_prime = 2
    t0_prime = 10
    initial_conditions = [1,0,0,0]
    num_points = 10000
    omega_prime = frequency_array[np.argmax(freq_max)]
    t_prime_array = np.linspace(0, 25, num_points)
    
    # post-pulse probability index
    cutoff = int(np.argwhere(t_prime_array >= t0_prime+4*sigma_prime)[0])
    
    # creating arrays for probability and area
    N = 200
    amp_max = np.zeros(N)
    area = np.zeros(N)
    amp_array = np.linspace(0, 1, N)
    for i, V_prime in enumerate(amp_array):
        p1 = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)[1]
        amp_max[i] = np.mean(p1[cutoff:])
        
    # plots
    fig_single = plt.figure(figsize=(8,6))
    ax_single = fig_single.gca()
    ax_single.plot(amp_array, amp_max, "k")
    ax_single.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_single.set_ylabel("Post-Pulse Probability", fontsize=16)
    
#%%
'''
Investigating sigma and amplitude simultaneously to create a surface map. This
was never used in the report, but it is quite cool.
'''
import matplotlib.cm as cm
if __name__ == "__main__":
    plt.close()
    # variables
    omega_prime = 1
    t0_prime = 10
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 50, num_points)
    sigma_prime = 2
    cutoff = int(np.argwhere(t_prime_array >= t0_prime+4*sigma_prime)[0])
    
    N = 50
    amp_array = np.linspace(0, 1, N)
    std_array = np.linspace(0, 5, N)
    ppp = np.zeros((N,N))
    
    for i, V_prime in enumerate(amp_array):
        for j, sigma_prime in enumerate(std_array):
            cutoff = int(np.argwhere(t_prime_array >= t0_prime+4*sigma_prime)[0])
            p1 = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)[1]
            ppp[i,j] = np.mean(p1[cutoff:])
    
    amp_grid, std_grid = np.meshgrid(amp_array, std_array)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    ax.set_xlabel("$\\tilde{V}$", fontsize=16)
    ax.set_ylabel("$\\tilde{\\sigma}$", fontsize=16)
    ax.set_zlabel("Post-Pulse Probability", fontsize=16)
    contourplot = ax.plot_surface(amp_grid, std_grid, ppp, cmap = cm.hot)
    fig.colorbar(contourplot, shrink=0.4, aspect=5, pad=0.1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    