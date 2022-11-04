import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from harmonic_analysis import envelope_function, probability_calculator

'''
This solves the perturbation ODEs for a Gaussian-modulated cosine perturbation
for a given frequency and plots the transition probabilities and the state
at time t, given these probabilities. It also plots the perturbation signal.

The simulation runs from t = μ - 5σ to t = μ + 5σ.
'''
    
if __name__ == "__main__":
    plt.close()
    omega_prime = 1
    V_prime = 0.2
    t0_prime = 10
    sigma_prime = 2.5
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 25, num_points)
    
    
    linestyles = ["#6edbfa","#82a6bd","#957080","#a93b42","#bc0505"]

    
    N = 500
    signal = np.zeros(N)
    frequency_array = np.linspace(0, 2, N)
    for i, omega_prime in enumerate(frequency_array):
        p0, p1 = probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)
        
        cutoff = int(np.argwhere(t_prime_array >= 2*t0_prime)[0])
        p1_cutoff = p1[cutoff:]
        signal[i] = np.mean(p1_cutoff)
    
    # probability time plots
    fig_probs = plt.figure(figsize=(8,6))
    ax_probs = fig_probs.gca()
    ax_probs.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_probs.set_ylabel("Probability", fontsize=16)
    ax_probs.plot(t_prime_array, p0, "r", label="$P_{0}$")
    ax_probs.plot(t_prime_array, p1, "b", label="$P_{1}$")
    ax_probs.set_ylim(-0.1,1.1)
    ax_probs.legend(fontsize=14)
    
    # frequency sweep plot
    fig_signal = plt.figure(figsize=(8,6))
    ax_signal = fig_signal.gca()
    ax_signal.set_xlabel("Frequency $\\tilde{\\omega} = \\frac{\\omega}{\\omega_{10}}$", fontsize=16)
    ax_signal.set_ylabel("Signal Strength", fontsize=16)
    ax_signal.plot(frequency_array, signal, "k")
    
    # curve fit
    def func(x, a, b, c, d):
        # return a*np.exp((x + d)**2/b)
        return (a / np.pi) * (c / ((x - b)**2 + c**2)) + d
    
    param, _ = curve_fit(func, frequency_array, signal)
    a,b,c,d = param
    curve_label = "$y=\\frac{"+str(np.round(a,2))+"}{\\pi}\\cdot\\frac{"\
        +str(np.round(c,2))+"}{(x-"+str(np.round(b,2))+")^{2}+"+str(np.round(c,2))+"^{2})}"+str(np.round(d,2))+"$"
    ax_signal.plot(frequency_array, func(frequency_array,a,b,c,d), "r--", label=curve_label)
    ax_signal.legend(fontsize=12)
    

    # pertubation plot
    fig_envelope = plt.figure(figsize=(8,6))
    ax_envelope = fig_envelope.gca()
    ax_envelope.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_envelope.set_ylabel("Perturbation ($\\tilde{V}$)", fontsize=16)
    ax_envelope.plot(t_prime_array, envelope_function(t_prime_array, t0_prime, sigma_prime, omega_prime), "k")
    
    # comparing different frequencies
    fig_comparison = plt.figure(figsize=(8,6))
    ax_comparison = fig_comparison.gca()
    ax_comparison.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_comparison.set_ylabel("Probability", fontsize=16)
    
    for omega_prime, colour in zip([0.7, 0.8, 0.9], ["k", "r", "b"]):
        p0, p1 = probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)
        ax_comparison.plot(t_prime_array, p0, colour, label="$\\tilde{\\omega}="+str(omega_prime)+"$")
        ax_comparison.plot(t_prime_array, p1, colour)
    ax_comparison.legend(fontsize=14)
    
    
    


    
    

        
    
