import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


'''
Solving the differential equation

dcf/dt = -(iV_0)/(hbar ω_fi)e^(i ω_fi t)

Converting to natural units through

t' = ω_fi 

Differential equation is solved in 2 parts because of complex values:
    
dcr/dt' = V_0/(hbar ω_fi) sin(t')
dci/dt' = -V_0/(hbar ω_fi) cos(t')
    
For real and imaginary parts respectively. Then combined at the end to get the
time dependent probability function.
'''
def field_function(parameters, t, constants):
    # need initial conditions added
    V, h, w = constants
    dcr = V/(h*w) * np.sin(t)
    dci = -V/(h*w) * np.cos(t)
    return [dcr, dci]

def real_fitter(t, a, b, c, d):
    return a*np.cos(b*t + c) + d

def imag_fitter(t, a, b, c, d):
    return a*np.sin(b*t + c) + d

def prob_fitter(t, a, b, c, d):
    return a*np.sin(b*t + c)**2 + d

# cfr and cfi initial conditions. Both 0 because cf(0) = 0
# [real, imag]
initial_conditions = [0,0]
# V, h, w
constants = [0.8, 1, 1]
omega = str(constants[-1])

# t values for which the solver solves the equation of motion
t_array = np.linspace(0,4*np.pi,5000)

# solution from odeint
sol = odeint(field_function, initial_conditions, t_array, args=(constants,))
cfr = sol[:,0]
cfi = sol[:,1]

cf = cfr * 1j*cfi
pf = cf*np.conjugate(cf)

param_real, _ = curve_fit(real_fitter, t_array, cfr)
param_imag, _ = curve_fit(imag_fitter, t_array, cfi)


# plotting
fig_coeff = plt.figure(figsize=(8,6))
ax_coeff = fig_coeff.gca()
ax_coeff.plot(t_array, cfr, "ro", label="Re($C_{f}$)", markersize=4)
ax_coeff.plot(t_array, cfi, "bo", label="Im($C_{f}$)", markersize=4)

ax_coeff.plot(t_array, real_fitter(t_array, param_real[0], param_real[1], param_real[2], param_real[3]), "r")
ax_coeff.plot(t_array, imag_fitter(t_array, param_imag[0], param_imag[1], param_imag[2], param_imag[3]), "b")
ax_coeff.legend()

fig_prob = plt.figure(figsize=(8,6))
ax_prob = fig_prob.gca()
ax_prob.plot(t_array, pf, "ro", markersize=4)
ax_prob.set_xlabel("Time $t'=\\omega t$", fontsize=16)
ax_prob.set_ylabel("Probability", fontsize=16)
ax_prob.set_title("Probability of Transitioning to State $|f\\rangle$", fontsize=20)
ax_prob.set_ylim(0,1)

ax_prob.legend()