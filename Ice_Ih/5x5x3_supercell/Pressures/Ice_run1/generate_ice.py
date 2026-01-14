import numpy as np

# Ice Ih unit cell parameters
a = 4.518  # Angstroms
c = 7.356  # Angstroms

# Fractional coordinates for oxygen atoms in the unit cell (wurtzite structure)
# These are the 4 oxygen positions in the Ice Ih unit cell
frac_coords_O = np.array([
    [1/3, 2/3, 1/16],    # O1
    [2/3, 1/3, 5/16],    # O2
    [1/3, 2/3, 9/16],    # O3
    [2/3, 1/3, 13/16]    # O4
])

# Function to convert fractional to Cartesian coordinates (hexagonal system)
def frac_to_cart_hex(frac, a, c):
    x = a * (frac[0] + frac[1] * np.cos(np.radians(120)))
    y = a * frac[1] * np.sin(np.radians(120))
    z = c * frac[2]
    return np.array([x, y, z])

# Generate 3x3x2 supercell
nx, ny, nz = 3, 3, 2
atom_id = 1
mol_id = 1

print("TITLE     Ice Ih 3x3x2 Supercell - Wurtzite Structure")
print("REMARK    72 water molecules with hexagonal rings")
print(f"CRYST1{a*nx:9.3f}{a*ny:9.3f}{c*nz:9.3f}  90.00  90.00 120.00 P 63/m m c   72")

all_coords = []

# Generate all oxygen positions
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            for i, frac_O in enumerate(frac_coords_O):
                # Translate fractional coordinates
                frac_translated = frac_O + np.array([ix, iy, iz])
                cart = frac_to_cart_hex(frac_translated, a, c)
                
                # Store oxygen position
                ox, oy, oz = cart
                
                # Add hydrogens (simplified - pointing toward tetrahedral neighbors)
                # H1: along one tetrahedral direction
                h1x = ox + 0.757
                h1y = oy
                h1z = oz
                
                # H2: along another tetrahedral direction  
                h2x = ox - 0.379
                h2y = oy + 0.656
                h2z = oz
                
                # MW virtual site (along bisector of H-O-H)
                mwx = ox + 0.063
                mwy = oy + 0.109
                mwz = oz + 0.158
                
                # Print atoms
                print(f"ATOM  {atom_id:5d}  O   SOL  {mol_id:4d}    {ox:8.3f}{oy:8.3f}{oz:8.3f}  1.00  0.00           O")
                atom_id += 1
                print(f"ATOM  {atom_id:5d}  H1  SOL  {mol_id:4d}    {h1x:8.3f}{h1y:8.3f}{h1z:8.3f}  1.00  0.00           H")
                atom_id += 1
                print(f"ATOM  {atom_id:5d}  H2  SOL  {mol_id:4d}    {h2x:8.3f}{h2y:8.3f}{h2z:8.3f}  1.00  0.00           H")
                atom_id += 1
                print(f"ATOM  {atom_id:5d}  MW  SOL  {mol_id:4d}    {mwx:8.3f}{mwy:8.3f}{mwz:8.3f}  1.00  0.00           M")
                atom_id += 1
                mol_id += 1

print("END")
