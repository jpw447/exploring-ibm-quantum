 import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from constant_pulse_funcs import const_envelope_function, const_probability_calculator

'''
This solves the perturbation ODEs for a Gaussian-modulated cosine perturbation
for a given frequency and plots the transition probabilities and the state
at time t, given these probabilities. It also plots the perturbation signal.
'''
    
if __name__ == "__main__":
    plt.close()
    # variables
    V_prime = 0.02
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 25, num_points)

    tmin = 0 # switch on perturbation
    
    # creating arrays for probability and area
    N = 500                             # 500 points to look at
    signal = np.zeros(N)
    areas = np.zeros(N)
    
    # iterating over various pulse lengths and, therefore, areas
    tmax_array = np.linspace(0, 20, N)
    for i, tmax in enumerate(tmax_array):
        p0, p1 = const_probability_calculator(t_prime_array, V_prime, tmin, tmax, initial_conditions)
        
        # post pulse probability "signal"
        cutoff = int(np.argwhere(t_prime_array >= tmax)[0])
        p1_cutoff = p1[cutoff:]
        signal[i] = np.mean(p1_cutoff)
        
        # area of perturbation
        areas[i] = (tmax - tmin) * V_prime
    
    # frequency sweep plot
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("$\\tilde{t}_{max}$", fontsize=16)
    ax_signal.set_ylabel("Signal Strength", fontsize=16)
    ax_signal.plot(tmax_array, signal, "k")

    # pertubation plot
    fig_envelope = plt.figure(figsize=(8,6))
    ax_envelope = fig_envelope.gca()
    ax_envelope.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_envelope.set_ylabel("Perturbation ($\\tilde{V}$)", fontsize=16)
    ax_envelope.plot(t_prime_array, const_envelope_function(t_prime_array, tmin, tmax, V_prime), "k")
    
    # area plot
    fig_area = plt.figure(figsize=(8,6))
    ax_area = fig_area.gca()
    ax_area.set_xlabel("$\\tilde{t}_{max}$", fontsize=16)
    ax_area.set_ylabel("Area of Perturbation Pulse", fontsize=16)
    ax_area.plot(tmax_array, areas, "k")
    
    ## Comparing to Perturbation Theory ###
    # tmax where resonance occurs
    peak_index = find_peaks(signal)[0][0]
    peak_time = tmax_array[peak_index]
    comparison_tmax = 25
    # numerical result
    p0_num, p1_num = const_probability_calculator(t_prime_array, V_prime, tmin, comparison_tmax, initial_conditions)
    
    # perturbation theory
    '''
    Little bit hacked together. Grabs the nearest t value to the time where the
    pulse is switched off (switchoff_index)
    Then defines the perturbation theory equation for the duration of the pulse,
    and equal to its last value after the pulse is switched off.
    '''
    switchoff_index = np.absolute(t_prime_array - comparison_tmax).argmin()
    p1_PT = np.zeros(num_points)
    p1_PT[:switchoff_index] = 4*V_prime**2 * np.sin((t_prime_array[:switchoff_index]-tmin)/2)**2
    p1_PT[switchoff_index:] = p1_PT[:switchoff_index][-1]
    
    # probability time plots
    fig_probs = plt.figure(figsize=(8,6))
    ax_probs = fig_probs.gca()
    ax_probs.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_probs.set_ylabel("Probability", fontsize=16)
    # ax_probs.plot(t_prime_array, p0_num, "r", label="$P_{0}$")
    ax_probs.plot(t_prime_array, p1_num, "k", label="$P_{01}$")
    ax_probs.plot(t_prime_array, p1_PT, "b--", label="Perturbation Theory")
    ax_probs.legend()
    
    
#%%

if __name__ == "__main__":
    '''
    Now performing an amplitude sweep so that we can compare the numerical results
    to perturbation theory, and where the two deviate.
    They are compared by the maximum probability
    '''    
    plt.close()
    # variables
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 25, num_points)

    tmin = 0 # switch on perturbation
    tmax = 25
    
    # creating arrays for probability and area
    N = 200                             # 500 points to look at
    signal = np.zeros(N)
    pert_theory = np.zeros(N)
    frequency = np.zeros(N)
    
    # iterating over various amplitudes
    amp_array = np.linspace(0, 1, N)
    for i, V_prime in enumerate(amp_array):
        p1_num = const_probability_calculator(t_prime_array, V_prime, tmin, tmax, initial_conditions)[1]
        
        # maximum probability
        signal[i] = p1_num[-1]
        pt = 4*V_prime**2 * np.sin((t_prime_array[:switchoff_index]-tmin)/2)**2
        pert_theory[i] = pt[-1]
    
    difference = abs(signal-pert_theory)/signal # percentage difference
    amp_threshold = amp_array[np.min(np.where(difference>=0.05))]
    
    # amplitude sweep plot
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("$\\tilde{V}$", fontsize=16)
    ax_signal.set_ylabel("Post-Pulse Probability", fontsize=16)
    ax_signal.plot(amp_array, signal, "k", label="Numerical Integration")
    ax_signal.plot(amp_array, pert_theory, "r", label="Perturbation Theory")
    ax_signal.legend(fontsize=14, loc="best")
    
    # different between theory and
    fig_diff = plt.figure(figsize=(8,6))
    ax_diff = fig_diff.gca()
    ax_diff.set_xlabel("$\\tilde{V}$", fontsize=16)
    ax_diff.set_ylabel("Percentage Maximum Probability Difference", fontsize=15)
    ax_diff.plot(amp_array, difference, "k")
    ax_diff.hlines(0.05, 0, amp_threshold, linestyle="dashed", color="blue")
    ax_diff.vlines(amp_threshold, 0, 0.05, linestyle="dashed", color="blue")
    # ax_diff.text(0.03, 0.09, "$("+str(np.round(amp_threshold,3))+",0.05)$", fontsize=14)
    
#%%
if __name__ == "__main__":
    '''
    Performing a single perturbation, so that we can get an area of the pulse
    and construct a Gaussian signal later with the same area
    '''
    plt.close()
    # variables
    initial_conditions = [1,0,0,0]
    num_points = 10000
    V_prime = 0.02
    t_prime_array = np.linspace(0, 25, num_points)

    tmin = 0 # switch on perturbation
    tmax = 3.12625250501002 # resonance at this pulse length (arbitrary)
    
    # creating arrays for probability and area
    N = 200                             # 500 points to look at
    
    pulse = const_envelope_function(t_prime_array, tmin, tmax, V_prime)
    area = integrate.quad(const_envelope_function, tmin, tmax, args=(tmin, tmax, V_prime))
    print("Area of pulse = "+str(np.round(area[0],5)))
    
    fig_single = plt.figure(figsize=(8,6))
    ax_single = fig_single.gca()
    ax_single.plot(t_prime_array, const_envelope_function(t_prime_array, tmin, tmax, V_prime), "k")

    
    

        
    
