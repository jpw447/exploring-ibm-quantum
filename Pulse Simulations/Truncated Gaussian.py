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
omega_fi = 1
const = 0.5 # V_0/hbar
std = 0.5
t0 = 5*std # defined to make gaussian start at same height every time at t=0

tmax = t0+2*std
tmin = t0-2*std

def gaussian_distr(x, mu, sigma):
    if x > tmin and x < tmax:
        y = np.exp(-(x-omega_fi*mu)**2/(2*std**2))
    else:
        y = 0
    return y

def gaussian_distr_array(x, mu, sigma):
    func = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > tmin and x[i] < tmax:
            y = np.exp(-(x[i]-omega_fi*mu)**2/(2*std**2))
        else:
            y = 0
        func[i] = y
    return func

def gaussian_solver(parameters, t):
    c0r, c0i, c1r, c1i = parameters
    
    V = gaussian_distr(t, t0, std) # gaussian curve at time t
    
    dc0r = -const * V * c1r * np.sin(t) + const * V * c1i * np.cos(t)
    
    dc0i = -const * V * c1r * np.cos(t) - const * V * c1i * np.sin(t)
    
    dc1r = const * V * c0r * np.sin(t) + const * V * c0i * np.cos(t)
    
    dc1i = -const * V * c0r * np.cos(t) + const * V * c0i * np.sin(t)
    
    return [dc0r, dc0i, dc1r, dc1i]


# time array
t_array = np.linspace(0, 14, 10000)
t_array = t_array*omega_fi

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
ax_prob.set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax_prob.set_ylabel("Probability", fontsize=16)
ax_prob.set_title("Probabilities for Gaussian Perturbation "+variable_str, fontsize=16)
ax_prob.legend(fontsize=14)

fig_trunc = plt.figure(figsize=(8,6))
ax_trunc = fig_trunc.gca()
ax_trunc.plot(t_array, gaussian_distr_array(t_array, t0, std))
ax_trunc.set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax_trunc.set_ylabel("Perturbation $\\frac{V_{0}}{\\hbar}$", fontsize=16)
ax_trunc.set_title("Perturbation", fontsize=20)

fig, ax = plt.subplots(1, 2, figsize=(10,6))
plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
fig.suptitle("Truncated Gaussian Perturbation", fontsize=18)
ax[0].plot(t_array, p0, "b", label="$P_{0,0}$")
ax[0].plot(t_array, p1, "r", label="$P_{0,1}$")
ax[0].set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax[0].set_ylabel("Probability", fontsize=16)
ax[0].set_title("Probabilities, "+variable_str, fontsize=16)

ax[1].plot(t_array, gaussian_distr_array(t_array, t0, std))
ax[1].set_xlabel("Time $t'=\\omega_{01}t$", fontsize=16)
ax[1].set_ylabel("Perturbation $\\frac{V_{0}}{\\hbar}$", fontsize=16)
ax[1].set_title("Perturbation", fontsize=20)

