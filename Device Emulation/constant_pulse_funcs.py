import numpy as np
from scipy.integrate import odeint

def const_envelope_function(t_prime_array, tmin, tmax, V_prime):
    '''
    Returns a function describing a square pulse, switched on at tmin and off
    at tmax. See const_probability_calculator for full description of parameters.
    '''
    return np.where((t_prime_array >= tmin) & (t_prime_array <= tmax), V_prime, 0)

def const_perturbation_equations(parameters, t_prime, V_prime, tmin, tmax):
    '''
    The dimensionless ODEs used to describe the coefficients for a two level 
    system under a square perturbation. See harm_probability_calculator for a 
    description of the parameters. 
    
    Parameters
    ----------
    coefficients : numpy.ndarray
        Coeffiecients of the state at the previous timestep according to odeint. 
        
    Returns
    -------
    list
        A list containing the changes in coefficients for odeint to use.
    '''
    c0r, c0i, c1r, c1i = parameters
    
    # switching pulse on and off
    if t_prime < tmin or t_prime > tmax:
        V = 0
    else:
        V = 1
    
    dc0r = -V_prime * V * c1r * np.sin(t_prime) + V_prime * V * c1i * np.cos(t_prime)
    
    dc0i = -V_prime * V * c1r * np.cos(t_prime) - V_prime * V * c1i * np.sin(t_prime)
    
    dc1r = V_prime * V * c0r * np.sin(t_prime) + V_prime * V * c0i * np.cos(t_prime)
    
    dc1i = -V_prime * V * c0r * np.cos(t_prime) + V_prime * V * c0i * np.sin(t_prime)
     
    return [dc0r, dc0i, dc1r, dc1i]

def const_probability_calculator(t_prime_array, V_prime, tmin, tmax, initial_conditions):
    '''
    Calculates the transition probabilities as a function of time, given the system
    of equtions defined by const_perturbation_equations, expressed in dimensionless 
    units.
    
    Parameters
    ----------
    t_prime_array : numpy.ndarray
        The dimensionless time array over which the pulse is applied.
    V_prime : float
        Dimensionless amplitude of perturbation.
    tmin : float
        Dimensionless time at which perturbation is switched on.
    tmax : float
        Dimensionless time at which perturbation is switched off.
    initial_conditions : numpy.ndarray
        Magnitudes of the real and complex parts of each state coefficient.
        E.g. [1,0,0,0] starts the system in |c0| = 1, |c1|=0 at t=0.
    Returns
    -------
    p0 : numpy.ndarray
        The probability-time curve for transitioning from |1> to |0>.
    p1 : numpy.ndarray
        The probability-time curve for transitioning from |0> to |1>. This is
        often the primary interest.
    '''
    
    # solution
    sol = odeint(const_perturbation_equations, initial_conditions, t_prime_array, args=(V_prime, tmin, tmax))
     
    # coefficients and probabilities
    c0 = sol[:,0] + 1j*sol[:,1]
    c1 = sol[:,2] + 1j*sol[:,3]
    
    # probability of being in state n, pn
    p0 = c0 * np.conjugate(c0)
    p1 = c1 * np.conjugate(c1)
    
    return p0, p1