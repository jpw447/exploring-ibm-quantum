import numpy as np
from scipy.integrate import odeint, quad

def gauss_envelope_function(t_prime, omega_prime, V_prime, t0_prime, sigma_prime):
    '''
    Provides the function for the pulse by returning a Gaussian pulse,
    given the specified parameters, evaluated at time t_prime.
    
    The area of the pulse is not parameterised, so you can't proactively 
    change it using this function. You need to use area_specified_gauss.
    
    Returns
    -------
    Float or numpy.ndarray that describes the pulse at time t.
    '''
    return V_prime*np.exp(-((t_prime)-(t0_prime))**2/(2*(sigma_prime)**2)) * np.cos(omega_prime*t_prime)


def gauss_odes(coefficients, t_prime, omega_prime, V_prime, t0_prime, sigma_prime):
    '''
    The dimensionless ODEs used to describe the coefficients for a two level 
    system under a Gaussian perturbation. See gauss_probability_calculator for a 
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
    
    V = gauss_envelope_function(t_prime, omega_prime, V_prime, t0_prime, sigma_prime) # gaussian curve at time t

    dc0r = -V * c1r * np.sin(t_prime) + V * c1i * np.cos(t_prime)
    
    dc0i = -V * c1r * np.cos(t_prime) - V * c1i * np.sin(t_prime)
    
    dc1r = V * c0r * np.sin(t_prime) + V * c0i * np.cos(t_prime)
    
    dc1i = -V * c0r * np.cos(t_prime) + V * c0i * np.sin(t_prime)
     
    return [dc0r, dc0i, dc1r, dc1i]

def gauss_probability_calculator(t_prime_array, omega_prime, V_prime, t0_prime, sigma_prime, initial_conditions):
    '''
    Calculates the transition probabilities as a function of time, given the system
    of equtions defined by gauss_odes. This works in dimensionless units.
    The area of the pulse is not parameterised, so you can't proactively 
    change it using this function. You need to use area_probability calculator.
    
    Sigma_prime and t0_prime are key to determining pulse width.
    
    Parameters
    ----------
    t_prime_array : numpy.ndarray
        Dimensionless time array for which the simulation is run.
    omega_prime : float
        Dimensionless frequency of perturbation.
    V_prime : numpy.ndarray
        Dimensionless amplitude of pulse.
    t0_prime : float
        Dimensionless mean where the maximum of the pulse envelope occurs.
    sigma_prime : float
        Dimensionless standard deviation of the pulse envelope.
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
    sol = odeint(gauss_odes, initial_conditions, t_prime_array, args=(omega_prime, V_prime, t0_prime, sigma_prime))
     
    # coefficients and probabilities
    c0 = sol[:,0] + 1j*sol[:,1]
    c1 = sol[:,2] + 1j*sol[:,3]
    
    # probability of being in state n, pn
    p0 = c0 * np.conjugate(c0)
    p1 = c1 * np.conjugate(c1)
    
    return p0, p1

def area_specified_gauss(t_prime, t0_prime, sigma_prime):
    '''
    Provides the envelope function for the pulse by returning a Gaussian function,
    which is not normalised not parameterised by V_prime.
    
    Returns
    -------
    Float or numpy.ndarray that describes the pulse at time t.
    '''
    return np.exp(-((t_prime)-(t0_prime))**2/(2*(sigma_prime)**2))
    
def area_odes(coefficients, t_prime, omega_prime, V_prime, t0_prime, sigma_prime, area_ratio):
    '''
    Identical to gauss_odes, except for area_ratio. See documentation of gauss_odes
    for full description of what this function does and most of its parameters
    and return values.
    
    Parameters
    ----------
    area_ratio : float
        Ratio of desired area to the area of the Gaussian envelope as it exists.
        This variable is used to control the area of the pulse.
    '''
    c0r, c0i, c1r, c1i = coefficients
    
    V = area_ratio * np.cos(omega_prime * t_prime) * area_specified_gauss(t_prime, t0_prime, sigma_prime) 
    
    dc0r = -V * c1r * np.sin(t_prime) + V * c1i * np.cos(t_prime)
    
    dc0i = -V * c1r * np.cos(t_prime) - V * c1i * np.sin(t_prime)
    
    dc1r = V * c0r * np.sin(t_prime) + V * c0i * np.cos(t_prime)
    
    dc1i = -V * c0r * np.cos(t_prime) + V * c0i * np.sin(t_prime)
     
    return [dc0r, dc0i, dc1r, dc1i]
    

def area_probability_calculator(t, omega, V, t0, sigma, initial_conditions, new_area):
    '''    
    Identical to gauss_probability calculator, but allows for a specified area 
    of Gaussian pulse. See documentation of gauss_odes for full description of 
    what this function does and most of its parameters and return values.
    
    Parameters
    ---------
    new_area : float
        The area that the user wants the Gaussian envelope to take.
    '''
    
    current_area = quad(area_specified_gauss, np.min(t), np.max(t), args=(t0, sigma))[0]
    area_ratio = new_area/current_area
    
    # solution
    sol = odeint(area_odes, initial_conditions, t, args=(omega, V, t0, sigma, area_ratio))
    
    # coefficients and probabilities
    c0 = sol[:,0] + 1j*sol[:,1]
    c1 = sol[:,2] + 1j*sol[:,3]
    
    # probability of being in state n, pn
    p0 = c0 * np.conjugate(c0)
    p1 = c1 * np.conjugate(c1)
    
    return p0, p1
     