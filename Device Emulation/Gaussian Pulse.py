import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from gaussian_pulse_funcs import gauss_envelope_function, gauss_probability_calculator

'''
This solves the perturbation ODEs for a Gaussian-modulated cosine perturbation
for a given frequency and plots the transition probabilities and the state
at time t, given these probabilities.

RUN THIS CELL BEFORE RUNNING ANY OTHER CELL
'''
   

#%%
'''Basic probability plot of probability-versus time for Gaussian pulse'''
if __name__ == "__main__":
    plt.close()
    omega_prime = 1
    V_prime = 0.02
    t0_prime = 10
    sigma_prime = 100
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 25, num_points) 
    
    p = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)[1]
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    ax.plot(t_prime_array, p)
    
#%%
'''
Using area theorem to find out how long to apply the pulse for for a given
amplitude V_prime, so that we get maximum probability.
'''
from gaussian_pulse_funcs import area_probability_calculator, area_specified_gauss
if __name__ == "__main__":
    plt.close()
    # variables
    V_prime = 0.01
    omgea_prime = 1
    sigma_prime = 3
    t0_prime = 10
    num_points = 10000
    
    t_prime = np.linspace(0, 25, num_points)
    new_area = np.pi
    
    # calculates probability-versus-time for given area, then plots the pulse
    # envelope for the specified area (pulse)
    p1 = area_probability_calculator(t_prime, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions, new_area)[1]
    area = quad(area_specified_gauss, np.min(t_prime), np.max(t_prime), args=(t0_prime, sigma_prime))[0]
    pulse = new_area/area * area_specified_gauss(t_prime, t0_prime, sigma_prime)
    
    # plotting the pulse envelope
    fig_pulse = plt.figure(figsize=(8,6))
    ax_pulse = fig_pulse.gca()
    ax_pulse.plot(t_prime, pulse, "k")
    
    # probability-versus-time plot    
    fig_area_prob = plt.figure(figsize=(8,6))
    ax_area_prob = fig_area_prob.gca()
    ax_area_prob.plot(t_prime, p1, "k")
    ax_area_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_area_prob.set_ylabel("Probability", fontsize=16)
    
#%%
'''
Varying sigma_prime for a fixed pulse area, pi, to see how it affects the
post-pulse probability compared to the rotating wave approximation.
In extreme of large standard deviations, they should be very similar
(small diff)
'''
from gaussian_pulse_funcs import area_probability_calculator, area_specified_gauss
if __name__ == "__main__":
    plt.close()
    # variables
    V_prime = 0.01
    omgea_prime = 1
    sigma_prime = 3
    t0_prime = 10
    num_points = 10000
    
    new_area = np.pi
    
    N = 50
    std_array = np.linspace(1, 10, N)
    prob = np.zeros(N)
    diff = np.zeros(N)
    
    # iterating through various standard deviations and calculating
    # post-pulse probability (p1[-1])
    for i, sigma_prime in enumerate(std_array):
        
        # t goes from 0 to t0+6sigma
        t_prime = np.linspace(0, t0_prime + 6*sigma_prime, num_points)
        p1 = area_probability_calculator(t_prime, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions, new_area)[1]
        area = quad(area_specified_gauss, np.min(t_prime), np.max(t_prime), args=(t0_prime, sigma_prime))[0]
        pulse = new_area/area * area_specified_gauss(t_prime, t0_prime, sigma_prime)
        
        prob[i] = p1[-1] # post-pulse probability
        
        # comparing post-pulse probability to rotating wave approximation's PPP
        RWA = np.sin(0.5*np.linspace(0, np.pi/V_prime, num_points)*V_prime)**2
        diff[i] = abs(p1[-1] - RWA[-1])/p1[-1]

    # plotting the difference between RWA and numerical solution
    fig_std = plt.figure(figsize=(8,6))
    ax_std = fig_std.gca()
    ax_std.plot(std_array, diff, "k")
    ax_std.set_xlabel("$\\tilde{\\sigma}$", fontsize=16)
    ax_std.set_ylabel("Percentage Difference", fontsize=16)
#%%
'''
Plotting probabilities for a select few perturbation frequencies.
Plots used in the report.
'''
if __name__ == "__main__":
    plt.close()
    V_prime = 0.02
    t0_prime = 10
    sigma_prime = 2
    initial_conditions = [1,0,0,0]
    num_points = 10000
    t_prime_array = np.linspace(0, 25, num_points) 
    
    # calculating probability-versus-time plots for four frequencies
    omega_array = [0.7, 0.8, 0.9, 1.0]
    plots = np.zeros(len(omega_array), dtype=object)
    for i, omega_prime in enumerate(omega_array):
        plots[i] = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions)[1]
    
    # plotting each probability for each frequency, excluding resonant perturbation
    fig_compare = plt.figure(figsize=(8,6))
    ax_compare = fig_compare.gca()
    ax_compare.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_compare.set_ylabel("Probability", fontsize=16)
    for plot, colour, omega in zip(plots[:3], ["b", "r", "k"], omega_array):
        ax_compare.plot(t_prime_array, plot, colour, label="$\\tilde{\\omega}="+str(omega)+"$")
    ax_compare.set_ylim(-0.0001, 0.0026)
    ax_compare.legend(fontsize=12)
    
    # plotting resonant probability transition
    fig_prob = plt.figure(figsize=(8,6))
    ax_prob = fig_prob.gca()
    ax_prob.set_xlabel("Time $\\tilde{t}$", fontsize=16)
    ax_prob.set_ylabel("Probability", fontsize=16)
    ax_prob.plot(t_prime_array, plots[3], "b", label="$\\tilde{\\omega}=1.0$")
    ax_prob.set_ylim(-0.0001, 0.0026)
    ax_prob.legend(fontsize=12)

#%%
'''
Demonstrating the effect of increasing sigma on the similarity between a 
Gaussian and constant harmonic pulse. Larger sigma means they become very similar.
'''
from harmonic_pulse_funcs import harm_probability_calculator
if __name__ == "__main__":
    plt.close()
    V_prime = 0.01
    omega_prime = 1
    t0_prime = 10
    sigma_1 = 2
    sigma_2 = 100
    num_points = 10000
    interval = int(num_points/20)
    
    t_prime_array = np.linspace(0, 25, num_points)
    
    # calculating perturbation envelopes, probabilities and rotating wave approx.
    envelope_1 = gauss_envelope_function(t_prime_array, omega_prime, V_prime, t0_prime, sigma_1)
    envelope_2 = gauss_envelope_function(t_prime_array, omega_prime, V_prime, t0_prime, sigma_2)
    p1 = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_1, [1,0,0,0])[1]
    p2 = gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_2, [1,0,0,0])[1]
    RWA = harm_probability_calculator(t_prime_array, omega_prime, V_prime, [1,0,0,0])[1]
    
    # making plots
    fig, ax = plt.subplots(2, 2, figsize=(8,6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6, top=0.85)
    for i in range(2):
        ax[i,0].set_ylabel("Perturbation ($\\tilde{V}$)", fontsize=14)
        ax[i,1].set_ylabel("Probability", fontsize=14)
        for j in range(2):
            ax[i,j].set_xlabel("Time $\\tilde{t}$", fontsize=14)
            
    ax[0,0].plot(t_prime_array, envelope_1, "k")
    ax[0,0].set_title("$\\tilde{\\sigma}="+str(sigma_1)+"$", fontsize=16)
    ax[0,1].plot(t_prime_array, p1, "k")
    ax[1,0].plot(t_prime_array, envelope_2, "b")
    ax[1,0].set_title("$\\tilde{\\sigma}="+str(sigma_2)+"$", fontsize=16)
    ax[1,1].plot(t_prime_array, p2, "b", label="Gaussian")
    ax[1,1].plot(t_prime_array[::interval], RWA[::interval], "ro", label="Constant")
    ax[1,1].legend(fontsize=14)

        
    
