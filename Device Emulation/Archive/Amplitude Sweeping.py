import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from harmonic_analysis import amplitude_sweep, fourier_func

'''
Solving the differential equations describing a Gaussian perturbation to a
two-level system. By varying the perturbation frequency and measuring the state at time
'''


#%%
'''
This cell scans for various standard deviations, amplitudes or means to find
the largest peak for each. Scans one at a time, depending on what is and isn't
commented.
'''
if __name__ == "__main__":
    plt.close()
    
    
    omega_fi = 0.2           # frequency between states
    std = 2.5
    t0 = 12.8
    num_points = 10000      # number of time array points
    num_measurements = 1024
    frequency = omega_fi    # resonance   
    initial_conditions = [1,0,0,0]
    
    # arrays
    N = 1000
    amp_array = np.linspace(0.1, 5, N)
    t_array = np.linspace(0, 4*t0, num_points)
    state_array = np.zeros(N)
    
    probability_averages = amplitude_sweep(omega_fi, std, t0, frequency, t_array, amp_array, initial_conditions)
    
    ### vectorised?
    # rand_nums = np.random.rand(N)
    # state_array = np.where(rand_nums >= probability_averages, 1, 0)
    # state = np.sum(state_array)
    
    for i, probability in enumerate(probability_averages):
        state = 0
        for n in range(num_measurements):
            rand_num = np.random.rand()
            if rand_num <= probability:
                state += 1
            else:
                pass
        state = state/num_measurements
        state_array[i] = state
    state_array -= np.mean(state_array)
    
    # fourier transform
    power, freq = fourier_func(state_array, np.max(t_array))
    sine_freq = freq[np.argmax(power)]
    print("Signal frequency is "+str(sine_freq))
    
    linestyles = ["#6edbfa","#82a6bd","#957080","#a93b42","#bc0505"]
    labels = []
    
    # figure for plotting the signal strength
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("Amplitude", fontsize=16)
    ax_signal.set_ylabel("Signal Strength", fontsize=16)
    ax_signal.set_title("Average Signal Strength over "+str(num_measurements)+" measurements", fontsize=20)
    # ax_signal.set_ylim(0, 1)
    
    ax_signal.plot(amp_array, state_array, "b")
    # ax_signal.plot(amp_array, np.sin(amp_array + 0.1))
    
    fig_fourier = plt.figure(figsize=(8,6))
    ax_fourier = fig_fourier.gca()
    ax_fourier.set_xlabel("Signal Frequency $t$", fontsize=16)
    ax_fourier.set_ylabel("Power", fontsize=16)
    ax_fourier.set_title("Frequency Spectrum of Signal", fontsize=20)
    ax_fourier.plot(freq, power, "ko")
# 