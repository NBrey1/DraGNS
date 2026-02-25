#!/usr/bin/env python3
"""
Generate GROMACS index file for 3-site Hexagonal Water Ice (OW, HW1, HW2)
"""

def generate_ice_bond_index(n_molecules=632):
    # Mapping for 3-site model:
    # OW  = mol * 3 + 1
    # HW1 = mol * 3 + 2
    # HW2 = mol * 3 + 3
    
    with open('ice_bonds.ndx', 'w') as f:
        # 1. System Group
        f.write("[ System ]\n")
        all_atoms = list(range(1, n_molecules * 3 + 1))
        for i, atom in enumerate(all_atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0: f.write("\n")
        f.write("\n\n")

        # 2. Oxygens (OW)
        f.write("[ Oxygens ]\n")
        for i in range(n_molecules):
            f.write(f"{i * 3 + 1} ")
            if (i + 1) % 15 == 0: f.write("\n")
        f.write("\n\n")

        # 3. Hydrogens (HW1 and HW2)
        f.write("[ Hydrogens ]\n")
        all_h = []
        for i in range(n_molecules):
            all_h.extend([i * 3 + 2, i * 3 + 3])
        for i, atom in enumerate(all_h):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0: f.write("\n")
        f.write("\n\n")

        # 4. O-H Bonds (Pairs for gmx distance or gmx angle)
        f.write("[ OH_bonds ]\n")
        oh_bonds = []
        for i in range(n_molecules):
            ow = i * 3 + 1
            hw1 = i * 3 + 2
            hw2 = i * 3 + 3
            oh_bonds.extend([ow, hw1, ow, hw2])
        
        for i, atom in enumerate(oh_bonds):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0: f.write("\n")
        f.write("\n")

    print(f"Done! Created index for {n_molecules} molecules ({n_molecules * 3} total atoms).")

if __name__ == "__main__":
    generate_ice_bond_index(632)
