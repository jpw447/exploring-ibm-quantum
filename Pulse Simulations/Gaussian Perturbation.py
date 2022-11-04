import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


'''
Solving the differential equations describing a Gaussian perturbation to a
two-level system. By varying the perturbation frequency and measuring the state at time
'''
omega_fi = 1
const = 0.5 # V_0/hbar
std = 0.5 
t0 = 5*std # defined to make gaussian start at same height every time at t=0

def gaussian_distr(x, mu, sigma):
    return np.exp(-(x-omega_fi*mu)**2/(2*std**2))

def gaussian_solver(parameters, t):
    c0r, c0i, c1r, c1i = parameters
    
    V = gaussian_distr(t, t0, std) # gaussian curve at time t
    
    dc0r = -const * V * c1r * np.sin(t) + const * V * c1i * np.cos(t)
    
    dc0i = -const * V * c1r * np.cos(t) - const * V * c1i * np.sin(t)
    
    dc1r = const * V * c0r * np.sin(t) + const * V * c0i * np.cos(t)
    
    dc1i = -const * V * c0r * np.cos(t) + const * V * c0i * np.sin(t)
    
    return [dc0r, dc0i, dc1r, dc1i]


# time array
t_array = np.linspace(0, 25, 1000)
t_array = omega_fi * t_array

initial_conditions = [1, 0, 0, 0]

# solution
sol = odeint(gaussian_solver, initial_conditions, t_array)

# coefficients and probabilities
c0 = sol[:,0] + 1j*sol[:,1]
c1 = sol[:,2] + 1j*sol[:,3]

p0 = c0 * np.conjugate(c0)
p1 = c1 * np.conjugate(c1)

variable_str = "$V_{0} = "+str(const) + ", \\sigma = "+str(0.5)+", t_{0} = "+str(t0)+"$"

# probability and perturbation plots
fig_time, ax_time = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(wspace=0.3, hspace=0.2, top=0.85)
fig_time.suptitle("Gaussian Perturbation", fontsize=18)
ax_time[0].plot(t_array, p0, "b", label="$P_{0,0}$")
ax_time[0].plot(t_array, p1, "r", label="$P_{0,1}$")
ax_time[0].set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax_time[0].set_ylabel("Probability", fontsize=16)
ax_time[0].set_title("Probabilities, "+variable_str, fontsize=16)

ax_time[1].plot(t_array, gaussian_distr(t_array, t0, std))
ax_time[1].set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax_time[1].set_ylabel("Perturbation $\\frac{V_{0}}{\\hbar}$", fontsize=16)
ax_time[1].set_title("Perturbation", fontsize=20)

