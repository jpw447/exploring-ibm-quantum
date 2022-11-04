Each file contains the functions used to solve the ODEs, solutions to the ODEs, or frequency sweeps,
of various forms - constant (constant envelope, no oscillation), harmonic (constant envelope, 
oscillating signal), and Gaussian (Gaussian envelope, oscillating signal).

## Data Folder
---
Contains data from IBMQ machines, performing experiments as detailed in Qiskit Chapter 6.1
(as of time of writing 02/09/2022).
This experiment can be found in "Pulse Control.ipynb", which is a Jupyter notebook designed to
be run in IBMQ's Quantum Experience Lab, and NOT a regular Jupyter notebook. Note that the backend
being accessed may need to be changed, depending on what's available.

## Requirements
---
`numpy`, `scipy` and `matplotlib`.

## File and Contents
---
Constant Pulse.py --- Solving the ODEs for a square pulse (probability vs time).
constant_pulse_funcs.py --- Contains functions to solve the square pulse ODEs. Use this as a library.
Gaussian Frequency Sweep.py --- Performs frequency, amplitude and standard deviation sweeps.
Gaussian Integral.py --- Short file to perform a numerical integral of a Gaussian function.
Gaussian Pulse.py --- Solving the ODEs for a Gaussian pulse (probability vs time).
gaussian_pulse_funcs.py --- Contains functions to solve the Gaussian pulse ODEs. Use this as a library.
Harmonic Frequency Sweep.py --- Performs frequency and amplitude sweeps, and maps perturbation theory.
Harmonic Pulse.py --- Solving the ODEs for a constant harmonic pulse. Includes perturbation theory.
					  (Probability vs time)
harmonic_pulse_funcs.py --- Contains functions to solve the Gaussian pulse ODEs. Use this as a library.
Qiskit Data.py --- Reads in files from the "Data" folder and compares perturbation theory and Rabi
				   oscillations by mapping them to the data with SciPy.

