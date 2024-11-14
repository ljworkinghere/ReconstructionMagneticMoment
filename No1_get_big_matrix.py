import numpy as np
import matplotlib.pyplot as plt

# This Python script computes the relationship between\
# the magnetic moment intensity for reconstruction
# and the normal magnetic field (B_n) at points\
# on the surface of the region of interest,
# referred to as the Neumann boundary.

# Load data files:
store_coord = np.loadtxt("store_coord.txt")  # Coordinates of discrete points on the surface of the region of interest.
norm_direction = np.loadtxt("norm_direction.txt")  # Normal unit vectors at each point on the surface.
Bfield = np.loadtxt("BxByBz.csv")  # Magnetic field components (Bx, By, Bz) at each point on the surface.
extend_coord = np.loadtxt("extend_coord.txt")  # Coordinates of magnetic moments for reconstruction.

# Note: The direction of the magnetic moment is assumed to align with the normal direction
# at the corresponding surface point.

# Compute the normal magnetic field B_n at each point on the surface:
norm_B = Bfield * norm_direction  
norm_B = np.sum(norm_B, axis=1)  # obtain B_n for each point.
np.save("norm_B.npy", norm_B)  # Save the normal magnetic field for later use.
print(max(abs(Bz)))  # the dimension the the Matrix

# Initialize variables:
num_dipo = len(Bfield)  # Number of magnetic moments (same as number of surface points).
B_matrix = np.zeros((num_dipo, num_dipo))  # Matrix to store the contribution of every magnetic moments
                                        # to B_n for each point on the surface

mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (constant).

# Compute the magnetic field B_n for all points on the surface\
# derived from each magnetic moment.
for i in range(num_dipo):
    r_source = np.tile(extend_coord[i], (num_dipo, 1)) 
    r_temp = store_coord - r_source  # Vector from the source magnetic moment to observation point.
    cdot_temp1 = np.dot(r_temp, norm_direction[i])  
    cdot_temp2 = np.sum(r_temp * norm_direction, axis=1) 
    cdot_temp3 = np.sum(norm_direction[i] * norm_direction, axis=1) 
    r_mag = np.linalg.norm(r_temp, axis=1) 
    
    # Compute the magnetic field B_n using the formula:
    B_temp = 1.0e-7 * (3 * cdot_temp1 * cdot_temp2 / r_mag ** 5 - cdot_temp3 / r_mag ** 3)
    
    # Store the computed field values in the matrix:
    B_matrix[:, i] = B_temp
    
    # Print progress for every 300th dipole:
    if i % 300 == 0:
        print(i / num_dipo, cdot_temp1[i], cdot_temp2[i], cdot_temp3[i])

# Save the computed B_matrix for future use:
np.save("B_matrix.npy", B_matrix)  # Save the magnetic field matrix.

