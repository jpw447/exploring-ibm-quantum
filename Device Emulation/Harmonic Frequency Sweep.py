import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from harmonic_pulse_funcs import harm_envelope_function, harm_probability_calculator

'''
This solves the perturbation ODEs for a Gaussian-modulated cosine perturbation
for a given frequency and plots the transition probabilities and the state
at time t, given these probabilities. It also plots the perturbation signal.

The simulation runs from t = μ - 5σ to t = μ + 5σ.
'''
    
#%%
'''
Frequency sweep for a harmonic perturbation by finding maximum probability
'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.01
    initial_conditions = [1,0,0,0]
    num_points = 10000
    tmax = np.pi/V_prime
    t_prime_array = np.linspace(0, tmax, num_points) 
    
    N = 500
    frequency_array = np.linspace(0.5, 1.5, N)
    ppp = np.zeros(N) # post-pulse probability
    
    for i, omega_prime in enumerate(frequency_array):
        p1 = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        ppp[i] = p1[-1]
    
    fig_sweep = plt.figure(figsize=(8,6))
    ax_sweep = fig_sweep.gca()
    ax_sweep.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_sweep.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_sweep.plot(frequency_array, ppp, "k", label="Numerical")
    
    # adding perturbation theory for comparison
    p_pt = V_prime**2  * (np.sin(0.5*tmax/20*(1-frequency_array))**2)/(1-frequency_array)**2
    ax_sweep.plot(frequency_array, p_pt, "r--", label="Perturbation Theory")
    ax_sweep.legend(fontsize=12)

#%%
'''
Near resonance frequency sweep, with perturbation theory for comparison
'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.01
    initial_conditions = [1,0,0,0]
    num_points = 10000
    tmax = 0.1*np.pi/V_prime
    t_prime_array = np.linspace(0, tmax, num_points) 
    
    N = 500
    frequency_array = np.linspace(0, 2, N)
    
    ppp = np.zeros(N)
    for i, omega_prime in enumerate(frequency_array):
        p1 = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        ppp[i] = p1[-1]
    
    fig_sweep = plt.figure(figsize=(8,6))
    ax_sweep = fig_sweep.gca()
    ax_sweep.set_xlabel("Frequency $\\tilde{\\omega}$", fontsize=16)
    ax_sweep.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_sweep.plot(frequency_array, ppp, "k", label="Numerical")
    
    # adding perturbation theory for comparison
    ''' Finds the ratio between solution maxima, and then scales V_prime so that
    the perturbation theory maximum is at P=1'''
    p_pt = V_prime**2  * (np.sin(0.5*tmax*(1-frequency_array))**2)/(1-frequency_array)**2
    # scale_factor = np.max(prob_max)/np.max(p_pt)
    # p_pt *= scale_factor
    ax_sweep.plot(frequency_array, p_pt, "r--", label="Perturbation Theory")
    ax_sweep.legend(fontsize=12)
        
#%%
'''
Time sweep for harmonic perturbation, by scoping for the probability
achieved after the pulse has been applied, for multiple pulse durations
'''
if __name__ == "__main__":
    plt.close()
    omega_prime = 1
    V_prime = 0.01
    initial_conditions = [1,0,0,0]
    num_points = 10000
    
    N = 1000
    tmax_array = np.linspace(200, 400, N)
    
    ppp = np.zeros(N)
    
    '''Changing tmax changes the magnitude of probability'''
    
    for i, tmax in enumerate(tmax_array):
        t_prime_array = np.linspace(0, tmax, 10000)
        p1 = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        ppp[i] = p1[-1]
    
    fig_prob = plt.figure(figsize=(8,6))
    ax_prob = fig_prob.gca()
    ax_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_prob.plot(t_prime_array, p1, "b", label="Numerical Solution")
    ax_prob.legend(fontsize=12)
    
    fig_sweep = plt.figure(figsize=(8,6))
    ax_sweep = fig_sweep.gca()
    ax_sweep.set_xlabel("Pulse Duration $\\tilde{t}$", fontsize=16)
    ax_sweep.set_ylabel("Post-Pulse Probability", fontsize=16)
    # ax_sweep.set_title("$\\tilde{V}="+str(V_prime)+"$", fontsize=24)
    ax_sweep.plot(tmax_array, ppp, "k")
    
    pulse_duration = tmax_array[np.argmax(ppp)]
    
#%%
'''
Amplitude sweep for harmonic perturbation, by scoping for the probability
achieved after the pulse has been applied, for multiple amplitudes V_prime
'''
if __name__ == "__main__":
    plt.close()
    omega_prime = 1
    initial_conditions = [1,0,0,0]
    num_points = 10000
    pulse_duration = 10
    t_prime_array = np.linspace(0, pulse_duration, num_points) 
    
    N = 500
    amp_array = np.linspace(0, 1, N)
    
    ppp = np.zeros(N)
    for i, V_prime in enumerate(amp_array): 
        p1 = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        ppp[i] = p1[-1]
    
    fig_sweep = plt.figure(figsize=(8,6))
    ax_sweep = fig_sweep.gca()
    ax_sweep.set_xlabel("Amplitude $\\tilde{V}$", fontsize=16)
    ax_sweep.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_sweep.plot(amp_array, ppp, "k", label="Numerical")
    ax_sweep.plot(amp_array, np.sin(0.5*pulse_duration*amp_array)**2,"r--", label="RWA")
    ax_sweep.legend(fontsize=12)
    print(amp_array[np.argmax(ppp)])