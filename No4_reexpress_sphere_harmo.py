# This Python script expands spherical harmonics into\
# Cartesian coordinates and stores them in a dictionary.


import numpy as np
import math
import pickle

# Constants and initialization
L_MAX = 50
dir_coeff = {}

# Function to calculate the fundamental coefficient
def f_coeff_simple(l, m, k, r, s, p):
    upper_1 = math.factorial(2 * l - 2 * k)
    upper_2 = math.factorial(m) / math.factorial(p) / math.factorial(m - p)
    lower_1 = math.factorial(r)
    lower_2 = math.factorial(s)
    lower_3 = math.factorial(k - r - s)
    lower_4 = math.factorial(l - k)
    lower_5 = math.factorial(l - 2 * k - m)
    lower_6 = math.sqrt(math.factorial(l + m) / math.factorial(l - m))
    res = upper_1 / lower_4 / lower_5 / lower_3 / lower_2 / lower_1 * upper_2 / lower_6
    return res / 2**l * np.sqrt((2 * l + 1) / (2 * np.pi))

# Function for coefficient with positive m
def f_coeff_plus_m(l, m, k, r, s, p):
    if (m - p) % 4 in (1, 3):
        coe = 0
    elif (m - p) % 4 == 0:
        coe = 1 * (-1) ** k
    else:
        coe = -1 * (-1) ** k
    return coe * f_coeff_simple(l, m, k, r, s, p)

# Function for coefficient with negative m
def f_coeff_minus_m(l, m, k, r, s, p):
    if (-m - p) % 4 in (0, 2):
        coe = 0
    elif (-m - p) % 4 == 1:
        coe = 1 * (-1) ** k
    else:
        coe = -1 * (-1) ** k
    return coe * f_coeff_simple(l, -m, k, r, s, p)

# Function for coefficient when m = 0
def f_coeff_0(l, k, r, s):
    coe = 1 / np.sqrt(2) * (-1) ** k
    return coe * f_coeff_simple(l, 0, k, r, s, 0)

# Main computation loop for storing coefficients in dir_coeff
for L in range(L_MAX):
    for M in range(-L, L + 1):
        print(f"Processing L={L}, M={M}")
        for k in range(0, (L - abs(M)) // 2 + 1):
            for r in range(0, k + 1):
                for s in range(0, k - r + 1):
                    for p in range(0, abs(M) + 1):
                        # Compute based on the sign of M
                        if M > 0:
                            res = f_coeff_plus_m(L, M, k, r, s, p)
                            x_L, y_L, z_L = p + 2 * r, -p + 2 * s + M, L - M - 2 * s - 2 * r
                        elif M < 0:
                            res = f_coeff_minus_m(L, M, k, r, s, p)
                            x_L, y_L, z_L = p + 2 * r, -p + 2 * s + abs(M), L - abs(M) - 2 * s - 2 * r
                        else:
                            res = f_coeff_0(L, k, r, s)
                            x_L, y_L, z_L = 2 * r, 2 * s, L - 2 * s - 2 * r

                        # Store result in the dictionary
                        dir_coeff.setdefault((L, M), {}).setdefault((x_L, y_L, z_L), 0)
                        dir_coeff[(L, M)][(x_L, y_L, z_L)] += res

# Save the coefficient dictionary to a file
with open('dir_coeff.pkl', 'wb') as f_save:
    pickle.dump(dir_coeff, f_save)

