from ase.io import read, write
from ase.build import make_supercell
import numpy as np

# Read the CIF
crystal = read('2205105.cif')

# Print info
print(f"Atoms in unit cell: {len(crystal)}")
print(f"Cell: {crystal.cell}")

# Make a 2x2x2 supercell (gives you 8 unit cells)
P = np.diag([2, 2, 2])
supercell = make_supercell(crystal, P)

print(f"Atoms in supercell: {len(supercell)}")

# Write as PDB
write('melamine_supercell.pdb', supercell)
