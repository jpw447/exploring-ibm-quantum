import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


'''
Solving the differential equation

dcf/dt = -(iV_0)/hbar * cos(omega * t) * e^(i * omega_fi * t)

which describes a harmonic perturbation, using the rotating wave approximation
and again without for comparison. The solutions should be comparable when
omega ≈ omega_fi

The differential equation is split into its real and imaginary components, and
then solved.
'''
omega_fi = 3    # units of hbar=1
const = 0.2     # V_0/hbar
N = 1.01      # omega/omega_fi
omega_fi = 1
omega = N*omega_fi
delta = omega_fi - omega # delta for difference
sigma = omega_fi + omega # sigma for sum

# rotating wave approximation equations
def RWA_equations(parameters, t):
    c0r, c0i, c1r, c1i = parameters
    
    dc0r = -1*const/2 * c1r * np.sin(delta*t) + const/2 * c1i * np.cos(delta*t)
    
    dc0i = -1*const/2 * c1r * np.cos(delta*t) - const/2 * c1i * np.sin(delta*t)
    
    dc1r = const/2 * c0r * np.sin(delta*t) + const/2 * c0i * np.cos(delta*t)
    
    dc1i = -const/2 * c0r * np.cos(delta*t) + const/2 * c0i * np.sin(delta*t)
        
    return [dc0r, dc0i, dc1r, dc1i]

# full ODEs without rotating wave approximation
def full_equations(parameters, t):
    c0r, c0i, c1r, c1i = parameters
    
    dc0r = const/2 * c1r * (np.sin(delta*t) - np.sin(sigma*t)) +\
           const/2 * c1i * (np.cos(delta*t) + np.cos(sigma*t))
    
    dc0i = -const/2 * c1r * (np.cos(delta*t) + np.cos(sigma*t)) +\
            const/2 * c1i * (np.sin(delta*t) - np.sin(sigma*t))
    
    dc1r = const/2 * c0r * (np.sin(sigma*t) - np.sin(delta*t)) +\
           const/2 * c0i * (np.cos(sigma*t) + np.cos(delta*t))
           
    dc1i = -const/2 * c0r * (np.cos(sigma*t) + np.cos(delta*t))+\
            const/2 * c0i * (np.sin(sigma*t) - np.sin(delta*t))
    
    return [dc0r, dc0i, dc1r, dc1i]

# time array
t_array = np.linspace(0, 20, 1000)

initial_conditions = [1, 0, 0, 0]

# solution
sol_RWA = odeint(RWA_equations, initial_conditions, t_array)

sol_full = odeint(full_equations, initial_conditions, t_array)

# coefficients and probabilities
c0_RWA = sol_RWA[:,0] + 1j*sol_RWA[:,1]
c1_RWA = sol_RWA[:,2] + 1j*sol_RWA[:,3]

c0_full = sol_full[:,0] + 1j*sol_full[:,1]
c1_full = sol_full[:,2] + 1j*sol_full[:,3]

p0_RWA = c0_RWA*np.conjugate(c0_RWA)
p1_RWA = c1_RWA*np.conjugate(c1_RWA)

p0_full = c0_full*np.conjugate(c0_full)
p1_full = c1_full*np.conjugate(c1_full)


# probability plot for RWA
fig_RWA_prob = plt.figure(figsize=(8,6))
ax_RWA_prob = fig_RWA_prob.gca()
ax_RWA_prob.plot(t_array, p0_RWA, "b", label="$P_{0,0}$")
ax_RWA_prob.plot(t_array, p1_RWA, "r", label="$P_{0,1}$")
ax_RWA_prob.set_xlabel("Time $t$ (s)", fontsize=16)
ax_RWA_prob.set_ylabel("Probability", fontsize=16)
ax_RWA_prob.set_title("Probabilities for Rotating Wave Approximation", fontsize=20)
ax_RWA_prob.legend(fontsize=14)

# probability plot for full ODES
fig_full_prob = plt.figure(figsize=(8,6))
ax_full_prob = fig_full_prob.gca()
ax_full_prob.plot(t_array, p0_full, "b", label="$P_{0,0}$")
ax_full_prob.plot(t_array, p1_full, "r", label="$P_{0,1}$")
ax_full_prob.set_xlabel("Time $t$ (s)", fontsize=16)
ax_full_prob.set_ylabel("Probability", fontsize=16)
ax_full_prob.set_title("Probabilities for Full ODES", fontsize=20)
ax_full_prob.legend(fontsize=14)

# analytical solutions for RWA
delta = omega_fi - omega
omega_rabi = np.sqrt(delta**2 + const**2)
p1_analytical = (const/omega_rabi)**2 * np.sin(omega_rabi*t_array/2)**2

# plot p_01 for RWA
fig_compare = plt.figure(figsize=(8,6))
ax_compare = fig_compare.gca()
ax_compare.plot(t_array[::10], p1_RWA[::10], "ro", label="RWA Numerical Solution", markersize=4)
ax_compare.plot(t_array[::10], p1_full[::10], "bo", label="Full Numerical Solution", markersize=4)
ax_compare.plot(t_array, p1_analytical, "r", label="RWA Analytical Solution")
ax_compare.set_xlabel("Time $t$ (s)", fontsize=16)
ax_compare.set_ylabel("Probability", fontsize=16)
ax_compare.set_title("Numerical and Analytical Solutions", fontsize=20)
ax_compare.legend()

