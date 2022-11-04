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
'''

#%%
'''Plots probability vs time
Pulse applied for exactly enough time to achieve  P=1, worked out using
pulse area theorem
'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.01
    omega_prime = 1
    initial_conditions = [1,0,0,0]
    num_points = 10000
    tmax = np.pi/V_prime # area/amplitude
    t_prime = np.linspace(0, tmax, num_points)
    
    # probability-time plot
    p = harm_probability_calculator(t_prime, omega_prime, V_prime, initial_conditions)[1]
    
    # rotating wave approximation for comparison
    RWA = np.sin(0.5*t_prime * V_prime)**2
    
    fig_pulse = plt.figure(figsize=(8,6))
    ax_pulse = fig_pulse.gca()
    ax_pulse.plot(t_prime, harm_envelope_function(t_prime, omega_prime, V_prime))
    
    fig_prob = plt.figure(figsize=(8,6))
    ax_prob = fig_prob.gca()
    ax_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_prob.plot(t_prime, p, "k", label="Numerical Solution")
    ax_prob.plot(t_prime, RWA, "b--", label="RWA Solution")
    ax_prob.legend(fontsize=12)
    

#%%
'''Sweeping over pulse durations'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.3
    omega_prime = 1
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t1 = np.linspace(0, 10, num_points) 
    t2 = np.linspace(0, 18, num_points)
    
    # iterating over applying the pulse for various times tmax
    tmax_array = np.arange(1, 21, 1)
    prob = np.zeros(len(tmax_array))
    
    for i, tmax in enumerate(tmax_array):
        t_prime = np.linspace(0, tmax, num_points)
        p = harm_probability_calculator(t_prime, omega_prime, V_prime, initial_conditions)[1]
        prob[i] = p[-1]
    
    # plotting probability vs tmax
    fig_prob = plt.figure(figsize=(8,6))
    ax_prob = fig_prob.gca()
    ax_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_prob.plot(tmax_array, prob, "r", label="Numerical Solution")
    ax_prob.legend(fontsize=12)
    
#%%
'''
Plots numeric and perturbation theory probabilities for a specified area,
in this case. The area here is pi.
'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.01
    omega_prime = 0.999999
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, np.pi/V_prime, num_points) 
    
    # numerical solution
    p1 = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
    
    # plotting numerical solution
    fig_prob = plt.figure(figsize=(8,6))
    ax_prob = fig_prob.gca()
    ax_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_prob.plot(t_prime_array, p1, "b", label="Numerical Solution")
    
    # checking to see if perturbation theory applies (ratio should be << 1)
    print("V prime/(1-omega) ratio is "+str(np.round(V_prime/(1-omega_prime), 5)))
    
    # rotating wave approximation and perturbation theory for a much smaller
    # amplitude for perturbation theory
    RWA = (V_prime**2)/((1-omega_prime)**2 + V_prime**2) * np.sin(0.5*t_prime_array * np.sqrt((1-omega_prime)**2 + V_prime**2))**2
    V_prime = 0.006366268160757986
    pert_theory = V_prime**2 * (np.sin(0.5*(1-omega_prime)*t_prime_array)**2)/(1-omega_prime)**2
    
    # plots of theory
    ax_prob.plot(t_prime_array, RWA, "k--", label="RWA Solution")
    ax_prob.plot(t_prime_array, pert_theory, "r--", label="Perturbation Theory")
    ax_prob.legend(fontsize=12)

#%%
'''
Increasing amplitudes to find PERTURBATION THEORY deviation
t is run from 0 to pi/V_prime so that we get as close to a pi-pulse as possible,
so that it's a fair comparison.
'''
if __name__ == "__main__":
    plt.close()
    omega_prime = 0.99
    initial_conditions = [1,0,0,0]
    num_points = 10000
    ratio_array = np.arange(0.01, 1, 0.01)
    amp_array = ratio_array * (1-omega_prime)
    
    numerical = np.zeros(len(amp_array))
    PT = np.zeros(len(amp_array))
    
    # iterating over amplitudes for numerical results and perturbation theory
    for i, V_prime in enumerate(amp_array):        
        t_prime_array = np.linspace(0, np.pi/V_prime, num_points) 
        p1_num = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        p1_pt = V_prime**2 * (np.sin(0.5*(1-omega_prime)*t_prime_array)**2)/(1-omega_prime)**2
        
        numerical[i] = p1_num[-1]
        PT[i] = p1_pt[-1]
    
    
    fig_compare = plt.figure(figsize=(8,6))
    ax_compare = fig_compare.gca()
    ax_compare.set_xlabel("$\\tilde{V}/(1-\\tilde{\\omega})$", fontsize=16)
    ax_compare.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_compare.plot(ratio_array, numerical, "b", label="Numerical Solution")
    ax_compare.plot(ratio_array, PT, "r", label="Perturbation Theory")
    ax_compare.legend(fontsize=12)
#%%
'''
Increasing amplitudes to find ROTATING WAVE APPROXIMATION deviation
t is run from 0 to pi/V_prime so that we get as close to a pi-pulse as possible,
so that it's a fair comparison.
'''
if __name__ == "__main__":
    # RWA
    omega_prime = 0.9999
    amp_array = np.linspace(0.01, 1, 1000)
    numerical = np.zeros(len(amp_array))
    RWA = np.zeros(len(amp_array))
    
    # creating figure to plot an example probability versus time plot
    fig_example = plt.figure(figsize=(8,6))
    ax_example = fig_example.gca()
    ax_example.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_example.set_ylabel("Probability", fontsize=16)
    
    # iterating over amplitudes
    for i, V_prime in enumerate(amp_array):        
        t_prime_array = np.linspace(0, np.pi/V_prime, num_points) 
        p1_num = harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions)[1]
        p1_rwa = (V_prime**2)/((1-omega_prime)**2 + V_prime**2) * np.sin(0.5*t_prime_array * np.sqrt((1-omega_prime)**2 + V_prime**2))**2
        
        numerical[i] = p1_num[-1]
        RWA[i] = p1_rwa[-1]
        
        # plotting probability for V = 0.7 (i=697)
        if i == 697:
            ax_example.plot(t_prime_array, p1_num, "b", label="Numerical")
            ax_example.plot(t_prime_array, p1_rwa, "k", label="RWA")
            ax_example.legend(fontsize=12)
            
    # creating figure to compare the numerical solution to RWA
    fig_RWA = plt.figure(figsize=(8,6))
    ax_RWA = fig_RWA.gca()
    ax_RWA.set_xlabel("$\\tilde{V}$", fontsize=16)
    ax_RWA.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_RWA.plot(amp_array, numerical, "b", label="Numerical Solution")
    ax_RWA.plot(amp_array, RWA, "k", label="RWA Solution")
    ax_RWA.legend(fontsize=12)