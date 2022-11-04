import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


'''
Solving the differential equations

dcf/dt = -(iV_0)/hbar * cos(omega * t) * e^(i * omega_fi * t)

Here, the mean for the gaussian curve is best defined in terms of the standard
deviation. This ensures that the gaussian always starts at the same level, or
y-value.
'''
N = 5
omega_fi = 1
omega = N * omega_fi
const = 2 # V_0/hbar
std = 0.5
t0 = 5*std # defined to make gaussian start at same height every time at t=0

def gaussian_distr(x, mu, sigma):
    return np.exp(-(x-mu*omega_fi)**2/(2*std**2)) * np.sin(N * x)

def gaussian_solver(parameters, t):
    c0r, c0i, c1r, c1i = parameters
    
    V = gaussian_distr(t, t0, std) # gaussian curve at time t
    
    dc0r = -const * V * c1r * np.sin(t) + const * V * c1i * np.cos(t)
    
    dc0i = -const * V * c1r * np.cos(t) - const * V * c1i * np.sin(t)
    
    dc1r = const * V * c0r * np.sin(t) + const * V * c0i * np.cos(t)
    
    dc1i = -const * V * c0r * np.cos(t) + const * V * c0i * np.sin(t)
    
    return [dc0r, dc0i, dc1r, dc1i]


# time array
t_array = np.linspace(0, 7, 1000)
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

# probability plot 
fig_prob = plt.figure(figsize=(8,6))
ax_prob = fig_prob.gca()
ax_prob.plot(t_array, p0, "b", label="$P_{0,0}$")
ax_prob.plot(t_array, p1, "r", label="$P_{0,1}$")
ax_prob.plot(t_array, gaussian_distr(t_array, t0, std), "k--", label="Gaussian Envelope")
ax_prob.set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax_prob.set_ylabel("Probability", fontsize=16)
ax_prob.set_title("Probabilities for Gaussian Perturbation "+variable_str, fontsize=16)
ax_prob.legend(fontsize=14)


