import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


'''
Solving the differential equation

dcf/dt = -(iV_0)/hbar * cos(omega * t) * e^(i * omega_fi * t)

which describes a harmonic perturbation.

The differential equation is split into its real and imaginary components, and
then solved.
'''
const = 0.2     # V_0/hbar
N = 0.5           # omega/omega_fi
omega_fi = 1
omega = N*omega_fi


def field_function(parameters, t):
    dcfr = const/2 * (np.sin((omega_fi + omega)*t) + np.sin((omega_fi - omega)*t))
    dcfi = -const/2 * (np.cos((omega_fi+omega)*t) + np.cos((omega_fi - omega)*t))
    return [dcfr, dcfi]


# cfr and cfi initial conditions. Both 0 because cf(0) = 0
# [real, imag]
initial_conditions = [0,0]

# t values for which the solver solves the equation of motion
tmax = 0.1 * 1/const            # t << 2*hbar/V0
t_array = np.linspace(0,10,500)

# solution from odeint
sol = odeint(field_function, initial_conditions, t_array)
cfr = sol[:,0]
cfi = sol[:,1]

cf = sol[:,0] + 1j*sol[:,1]
pf = cf*np.conjugate(cf)


# plotting ode solutions
# real
fig_coeff_real = plt.figure(figsize=(8,6))
ax_coeff_real = fig_coeff_real.gca()
ax_coeff_real.plot(t_array, cfr, "ro", label="Numerical Solution", markersize=4)
ax_coeff_real.set_xlabel("Time $t$", fontsize=16)
ax_coeff_real.set_ylabel("Re$(C_{f}$", fontsize=16)
ax_coeff_real.set_title("Re$(C_{f})$", fontsize=22)

# imaginary
fig_coeff_imag = plt.figure(figsize=(8,6))
ax_coeff_imag = fig_coeff_imag.gca()
ax_coeff_imag.plot(t_array, cfi, "bo", label="Numerical Solution", markersize=4)
ax_coeff_imag.set_xlabel("Time $t$", fontsize=16)
ax_coeff_imag.set_ylabel("Im$(C_{f})$", fontsize=16)
ax_coeff_imag.set_title("Im$(C_{f})$", fontsize=22)

# plotting analytic solutions
wfi = 1
w = N*wfi
real_sol = const/2 * (-(np.cos((wfi+w)*t_array)-1)/(wfi+w) - (np.cos((wfi-w)*t_array)-1)/(wfi-w)) 
imag_sol = -const/2 * (np.sin((wfi+w)*t_array)/(wfi+w) + np.sin((wfi-w)*t_array)/(wfi-w))
analytic_coeff = real_sol + 1j*imag_sol
analytic_prob = analytic_coeff * np.conjugate(analytic_coeff)

ax_coeff_real.plot(t_array, real_sol, "r", label="Analytic Solution")
ax_coeff_imag.plot(t_array, imag_sol, "b", label="Analytic Solution")

ax_coeff_real.legend()
ax_coeff_imag.legend()

# Probability plot
fig_prob = plt.figure(figsize=(8,6))
ax_prob = fig_prob.gca()
ax_prob.plot(t_array, pf, "go", label="Numerical Solution", markersize=4)
ax_prob.plot(t_array, analytic_prob, "g", label="Analytic Solution")
ax_prob.set_xlabel("Time $t$", fontsize=16)
ax_prob.set_ylabel("Probability $P_{C_{f}}(t)$", fontsize=16)
ax_prob.set_title("Probability of state being in $|f\\rangle$", fontsize=22)
ax_prob.legend()

#%%
'''
Dimensionless units version
'''
omega_fi = 1
omega = 3*omega_fi
gamma = omega_fi/omega  # Conversion factor
const = 0.2/omega       # V_0/hbar*omega

# Dimensionless version of the ODES
def dim_field_function(parameters, t):
    dcfr = const/2 *(np.sin((gamma+1)*t) + np.sin((gamma-1)*t))
    dcfi = const/2 * (np.cos((gamma+1)*t) + np.cos((gamma-1)*t))
    return [dcfr, dcfi]

# Think of a time you want to run it for (t_array) and convert to dimensionless
# with omega
t_array_dim = t_array*omega
initial_conditions = [0,0]

sol_dim = odeint(dim_field_function, initial_conditions, t_array_dim)
cfr_dim = sol_dim[:,0]
cfi_dim = sol_dim[:,1]

cf_dim = sol_dim[:,0] + 1j*sol_dim[:,1]
pf_dim = cf_dim*np.conjugate(cf_dim)


# plotting ode solutions
# real
fig_coeff_real_dim = plt.figure(figsize=(8,6))
ax_coeff_real_dim = fig_coeff_real_dim.gca()
ax_coeff_real_dim.plot(t_array_dim, cfr_dim, "ro", label="Numerical Solution", markersize=4)
ax_coeff_real_dim.set_xlabel("Time $t$", fontsize=16)
ax_coeff_real_dim.set_ylabel("Re$(C_{f}$", fontsize=16)
ax_coeff_real_dim.set_title("Re$(C_{f})$", fontsize=22)

# imaginary
fig_coeff_imag_dim = plt.figure(figsize=(8,6))
ax_coeff_imag_dim = fig_coeff_imag_dim.gca()
ax_coeff_imag_dim.plot(t_array_dim, cfi_dim, "bo", label="Numerical Solution", markersize=4)
ax_coeff_imag_dim.set_xlabel("Time $t$", fontsize=16)
ax_coeff_imag_dim.set_ylabel("Im$(C_{f})$", fontsize=16)
ax_coeff_imag_dim.set_title("Im$(C_{f})$", fontsize=22)

# Probability plot
fig_prob = plt.figure(figsize=(8,6))
ax_prob = fig_prob.gca()
ax_prob.plot(t_array_dim, pf_dim, "go", label="Numerical Solution", markersize=4)
ax_prob.set_xlabel("Time $t'=\\omega t$", fontsize=16)
ax_prob.set_ylabel("Probability $P_{C_{f}}(t)$", fontsize=16)
ax_prob.set_title("Probability of Transitioning to State $|f\\rangle$", fontsize=22)
ax_prob.legend()




