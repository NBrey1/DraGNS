import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
###############################################
#
#   Convert CIF Box coordinates to Cartesian 
#
###############################################

# Melamine is monoclinic P2_1 / a
# 4 molecules/unit cell

a = 1.04330 # unit cell a in nm
b = 0.74580 # unit cell b in nm
c = 0.72380 # unit cell c in nm

alpha = 90 # 90 deg
gamma = 90 # 90 deg
beta = 113.30 # 113.30 deg

cosB = -0.3955 # cos(beta) deg
sinB = 0.9184  # sin(beta) deg


# Pattern to get positions for every atom in each molcule
# 1: x y z
# 2: -x -y -z
# 3: -x+1/2 y+1/2 -z
# 4: x+1/2 -y+1/2 z

# Open CIF file to read atomic coordinates into te script
# Open and read the entire file
#with open('CIFcoords.txt', 'r') as file:
#    x0, y0, z0 = map(float, file.read().split())


# Coordinates for atom in each melamine molecule in CIF box coords
with open('CIFcoords.txt', 'r', encoding='utf-8') as file:
    content = file.read().replace('−', '-')

coords = np.loadtxt(content.splitlines())

#print(coords.shape)
#print(coords)


# Array for x,y and z positions in each melamine molecule
x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

mel1 = np.column_stack(( x,      y,      z))
mel2 = np.column_stack((-x,     -y,     -z))
mel3 = np.column_stack((-x+0.5,  y+0.5, -z))
mel4 = np.column_stack(( x+0.5, -y+0.5,  z))

#print(mel1.shape)

# Conversions (primed are original coords, capitals are new cartesian coords)
# X = a*x' + c*cosB*z'
# Y = b*y'
# Z = c*sinB*z'

# Convert the atomic position in each molecule to cartesian for GROMACS

# Apply symmetry operations
mel1 = np.column_stack((
    coords[:,0],
    coords[:,1],
    coords[:,2]
))

mel2 = np.column_stack((
    -coords[:,0],
    -coords[:,1],
    -coords[:,2]
))

mel3 = np.column_stack((
    -coords[:,0] + 0.5,
    coords[:,1] + 0.5,
    -coords[:,2]
))

mel4 = np.column_stack((
    coords[:,0] + 0.5,
    -coords[:,1] + 0.5,
    coords[:,2]
))

# Combine all four molecules
all_molecules = np.vstack((mel1, mel2, mel3, mel4))

# Convert CIF coordinates to Cartesian coordinates
Xn = a * all_molecules[:,0] + c * cosB * all_molecules[:,2]
Yn = b * all_molecules[:,1]
Zn = c * sinB * all_molecules[:,2]

# Final 60 x 3 Cartesian coordinate array
cartesian = np.column_stack((Xn, Yn, Zn))

#print(cartesian.shape)


# Define a 4x3 matrix representing new atomic positions - each row is 1 MEL molecule
conv = np.array([
    [Xn[0], Yn[0], Zn[0]],
    [Xn[1], Yn[1], Zn[1]],
    [Xn[2], Yn[2], Zn[2]],
    [Xn[3], Yn[3], Zn[3]]
])

# Define the melamine molecules as the rows from conv
MEL1 = cartesian[0:15]
MEL2 = cartesian[15:30]
MEL3 = cartesian[30:45]
MEL4 = cartesian[45:60]

# Print coordinates in each molecule
for i, coord in enumerate(cartesian):
    molecule = i // 15 + 1
    atom = i % 15 + 1

    if atom == 1:
        print(f"\n--- Molecule {molecule} ---")

    print(f"Atom {atom:2d}: "
          f"X = {coord[0]:.6f}, "
          f"Y = {coord[1]:.6f}, "
          f"Z = {coord[2]:.6f}")

