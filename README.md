# Exploring IBM Quantum to Teach Quantum Mechanics

## Description
---
This project was completed as part of an internship under Professor Andrew Armour of the University of Nottingham between June and September 2022. The aim of the project was to understand the physics of quantum bits and use them to produce physics experiments and demonstrate quantum phenomena for the purpose of teaching quantum mechanics visually.

This code was completed independently and without using Github, which is why the files have been uploaded in bulk after the project had ended.

As part of the project, much code resides on IBM Quantum's website (`https://quantum-computing.ibm.com/`) since this was used to directly interface with the quantum devices and retrieve data using Python and IBM's `Qiskit` library. Some data analysis was performed separately in Python and used to produce two reports that was provided to Professor Armour - one focussing on Bell's Theoreom and another on emulating a quantum device with numerical simulations to predict its behaviour quantitatively for demonstration purposes.

The code associated with Bell's Theorem is not provided here, although the code for device simulation is.

`Device Emulation` contains the files used to simulate the devices and can be run independently. Guidance for the files are contained within in the form of detailed comments. These files also include frequency and amplitude sweep experiments, with all variables expressed in dimensionless form.

`Pulse Simulators` demonstrate simple solutions to perturbation theory, including the Rabi oscillation solution and rotating wave approximations. Gaussian, constant and square pulses are simulated here. These files do not attempt perform frequency sweeps or emulate any quantum devices, and are instead qualitative explorations of the perturbation ODEs.

## Requirements
---
This project uses `numpy`, `scipy` and `matplotlib` libraries.
