MATLAB code for the constant speed model used to simulate the behavior of zebrafish schoals confined in a two-dimensional circular tank.

The model assumes that the fish move with a constant speed v0 and change their orientation according to overdamped equations of motion, in response to external forces and interactions.

The model includes a wall-avoidance force and Gaussian noise.

The fish-fish interactions consist of repulsion-attraction and alignment interactions within the field of view.

A form of hydrodynamic interactions is also included.

The outcomes are the x and y coordinates of all the fish denoted with "xol" and "yol" and their orientation angles "phi_ol".

All these are arrays with size(Nfores*N, Np) with "Nfores" denoting the number of sample iterations, "N" the number of time steps and  "Np" the number of particles.
