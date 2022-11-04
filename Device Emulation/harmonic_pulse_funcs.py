import numpy as np
from scipy.integrate import odeint

def harm_envelope_function(t_prime, omega_prime, V_prime):
    '''
    Returns a function that describes the harmonic perturbation, given the 
    parameters. See harm_probability_calculator for a description of
    the parameters.
    '''
    return V_prime * np.cos(omega_prime*t_prime)

def harm_perturbation_equations(coefficients, t_prime, omega_prime, V_prime):
    '''
    The dimensionless ODEs used to describe the coefficients for a two level 
    system under a harmonic perturbation. See harm_probability_calculator for a 
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
    c0r, c0i, c1r, c1i = coefficients
    
    V = harm_envelope_function(t_prime, omega_prime, V_prime)
    
    dc0r = -V * c1r * np.sin(t_prime) + V * c1i * np.cos(t_prime)
    
    dc0i = -V * c1r * np.cos(t_prime) - V * c1i * np.sin(t_prime)
    
    dc1r = V * c0r * np.sin(t_prime) + V * c0i * np.cos(t_prime)
    
    dc1i = -V * c0r * np.cos(t_prime) + V * c0i * np.sin(t_prime)
     
    return [dc0r, dc0i, dc1r, dc1i]

def harm_probability_calculator(t_prime_array, omega_prime, V_prime, initial_conditions):
    '''
    Calculates the transition probabilities as a function of time, given the system
    of equtions defined by harm_perturbation_equations, expressed in dimensionless 
    units.
    
    Parameters
    ----------
    t_prime_array : numpy.ndarray
        The dimensionless time array over which the pulse is applied.
    omega_prime : float
        Dimensionless frequency of perturbation.
    V_prime : float
        Dimensionless amplitude of perturbation.
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
    sol = odeint(harm_perturbation_equations, initial_conditions, t_prime_array, args=(omega_prime, V_prime))
     
    # coefficients and probabilities
    c0 = sol[:,0] + 1j*sol[:,1]
    c1 = sol[:,2] + 1j*sol[:,3]
    
    # probability of being in state n, pn
    p0 = c0 * np.conjugate(c0)
    p1 = c1 * np.conjugate(c1)
    
    return p0, p1