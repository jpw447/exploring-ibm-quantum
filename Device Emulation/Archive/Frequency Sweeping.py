import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from harmonic_analysis import envelope_function, frequency_sweep

'''
Solving the differential equations describing a Gaussian perturbation to a
two-level system. By varying the perturbation frequency and measuring the state at time
'''

#%%
'''
This cell performs a frequency sweep for only one standard deviation and mean.
Small amplitude recommended.
'''

if __name__ == "__main__":
    plt.close()
    omega_fi = 1            # frequency between states
    amplitude = 0.1             # V_0/hbar
    std = 2.5 
    t0 = 12.8              # defined to make gaussian start at same height every time at t=0
    num_points = 10000      # number of time array points
    t_array = np.linspace(0, 4*t0, num_points)
    initial_conditions = [1,0,0,0]
    
    # establishing frequency array
    min_freq = 0
    max_freq = 3
    freq_step = 0.005
    frequency_array = np.arange(min_freq, max_freq+freq_step, freq_step)
    
    linestyles = ["#6edbfa","#82a6bd","#957080","#a93b42","#bc0505"]
    labels = []

    
    # figure for plotting the signal strength
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("Frequency $\\omega$", fontsize=16)
    ax_signal.set_ylabel("Signal Strength", fontsize=16)
    ax_signal.set_title("Average Signal Strength", fontsize=20)
    ax_signal.set_ylim(0, 1)
    
    # figure for plotting perturbation
    fig_envelope = plt.figure(figsize=(8,6))
    ax_envelope = fig_envelope.gca()
    ax_envelope.set_xlabel("Time $t$", fontsize=16)
    ax_envelope.set_ylabel("Perturbation Magnitude", fontsize=16)
    ax_envelope.set_title("Pertubation Signal", fontsize=20)
    
    frequency_sweep()
    probability_average = frequency_sweep(constants, std, t0, t_array, frequency_array, initial_conditions)
    perturbations = envelope_function(t_array, t0, std, 1)
    maxima = np.max(probability_average)
        
    # plotting frequency sweeps
    ax_signal.plot(frequency_array, probability_average, "k")
    
    # plotting perturbations
    ax_envelope.plot(t_array, perturbations, "k")
    
    resonant_freq = frequency_array[np.argmax(probability_average)]
    
    print("Resonant frequency is "+str(resonant_freq))

#%%
'''
This cell scans for various standard deviations, amplitudes or means to find
the largest peak for each. Scans one at a time, depending on what is and isn't
commented.
'''
if __name__ == "__main__":
    plt.close()
    omega_fi = 1            # frequency between states
    amplitude = 0.5             # V_0/hbar
    std = 0.5 
    t0 = 5*std              # defined to make gaussian start at same height every time at t=0
    num_points = 10000      # number of time array points
    initial_conditions = [1, 0, 0, 0]
    
    # establishing frequency array
    min_freq = 0
    max_freq = 3
    freq_step = 0.005
    frequency_array = np.arange(min_freq, max_freq+freq_step, freq_step)
    
    linestyles = ["#6edbfa","#82a6bd","#957080","#a93b42","#bc0505"]
    labels = []

    constants = [omega_fi, amplitude]
    
    # figure for plotting the signal strength
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("Frequency $\\omega$", fontsize=16)
    ax_signal.set_ylabel("Signal Strength", fontsize=16)
    ax_signal.set_title("Average Signal Strength", fontsize=20)
    ax_signal.set_ylim(0, 1)
    
    # figure for plotting perturbation
    fig_envelope = plt.figure(figsize=(8,6))
    ax_envelope = fig_envelope.gca()
    ax_envelope.set_xlabel("Time $t$", fontsize=16)
    ax_envelope.set_ylabel("Perturbation", fontsize=16)
    ax_envelope.set_title("Pertubation Signal", fontsize=20)
    
    # figure for plotting maxima
    fig_maxima = plt.figure(figsize=(8,6))
    ax_maxima = fig_maxima.gca()
    # ax_maxima.set_xlabel("Mean", fontsize=16)
    # ax_maxima.set_title("Maximum Signal Recorded VS Mean", fontsize=20)
    # ax_maxima.set_xlabel("Standard Deviation", fontsize=16)
    # ax_maxima.set_title("Maximum Signal Recorded VS Standard Deviation", fontsize=20)
    ax_maxima.set_xlabel("Amplitude", fontsize=16)
    ax_maxima.set_title("Maximum Signal Recorded VS Amplitude", fontsize=20)
    ax_maxima.set_ylabel("Maximum Signal", fontsize=16)
    
    # standard deviation
    N = 3
    std_array = np.linspace(0.5, 4, N)
    t0_array = np.linspace(5, 15, N)
    amp_array = np.linspace(0.1, 0.8, N)
    probability_average = np.zeros(len(std_array), dtype=object)
    perturbations = np.zeros(len(std_array), dtype=object)
    maxima = np.zeros(len(std_array), dtype=float)
    
    # for i, std in enumerate(std_array): 
    #     t0 = 5*std
    # for i, t0 in enumerate(t0_array):
    #     std = 2.5
    for i, amp in enumerate(amp_array):
        std = 2.5
        t0 = 12.8
        constants = [omega_fi, amp]
        t_array = np.linspace(0, 4*t0, num_points)
        labels.append("$\\sigma="+str(np.round(std,2))+", t_{0}="+str(np.round(t0,2))+"$")
        
        probability_average[i] = frequency_sweep(constants, std, t0, t_array, frequency_array, initial_conditions)
        perturbations[i] = envelope_function(t_array, t0, std, 1)
        maxima[i] = np.max(probability_average[i])
        
    # plotting frequency sweeps
    for average, style, label in zip(probability_average[:5], linestyles, labels):
        ax_signal.plot(frequency_array, average, color=style, label=label)
    ax_signal.legend()
    
    # plotting perturbations
    for perturbation, style, label in zip(perturbations[:5], linestyles, labels):
        ax_envelope.plot(t_array, perturbation, color=style, label=label)
    ax_envelope.legend()
    
    ax_maxima.plot(amp_array, maxima, "kx")
    print("Maximum signal occurs at "+str(std_array[np.argmax(maxima)])+" amplitude")
    
    
    

        
    
