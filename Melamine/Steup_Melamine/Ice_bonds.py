#!/usr/bin/env python3
"""
Generate index file for Ice Ih O-H bonds
Updated for TIP4P/Ice water molecules with virtual sites
"""

def generate_ice_bond_index(n_molecules=432):
    """Generate index file for Ice Ih O-H bonds
    
    Args:
        n_molecules: Number of water molecules (default: 432)
    """
    
    with open('ice_bonds.ndx', 'w') as f:
        # System - all atoms (4 atoms per molecule: OW, H1, H2, MW)
        f.write("[ System ]\n")
        atoms = list(range(1, n_molecules * 4 + 1))
        for i, atom in enumerate(atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:  # 15 atoms per line
                f.write("\n")
        if len(atoms) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All oxygens
        f.write("[ Oxygens ]\n")
        oxygens = [i * 4 + 1 for i in range(n_molecules)]
        for i, atom in enumerate(oxygens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(oxygens) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All H1 atoms
        f.write("[ H1_atoms ]\n")
        h1_atoms = [i * 4 + 2 for i in range(n_molecules)]
        for i, atom in enumerate(h1_atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(h1_atoms) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All H2 atoms
        f.write("[ H2_atoms ]\n")
        h2_atoms = [i * 4 + 3 for i in range(n_molecules)]
        for i, atom in enumerate(h2_atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(h2_atoms) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All hydrogens (H1 + H2 combined)
        f.write("[ All_Hydrogens ]\n")
        all_hydrogens = []
        for mol in range(n_molecules):
            all_hydrogens.append(mol * 4 + 2)  # H1
            all_hydrogens.append(mol * 4 + 3)  # H2
        for i, atom in enumerate(all_hydrogens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(all_hydrogens) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All MW virtual sites
        f.write("[ Virtual_Sites ]\n")
        mw_sites = [i * 4 + 4 for i in range(n_molecules)]
        for i, atom in enumerate(mw_sites):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(mw_sites) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # O-H1 bonds (pairs of atoms for each bond)
        f.write("[ OH1_bonds ]\n")
        oh1_bonds = []
        for mol in range(n_molecules):
            oxygen = mol * 4 + 1
            h1 = mol * 4 + 2
            oh1_bonds.extend([oxygen, h1])
        
        for i, atom in enumerate(oh1_bonds):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(oh1_bonds) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # O-H2 bonds (pairs of atoms for each bond)
        f.write("[ OH2_bonds ]\n")
        oh2_bonds = []
        for mol in range(n_molecules):
            oxygen = mol * 4 + 1
            h2 = mol * 4 + 3
            oh2_bonds.extend([oxygen, h2])
        
        for i, atom in enumerate(oh2_bonds):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(oh2_bonds) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All O-H bonds combined
        f.write("[ All_OH_bonds ]\n")
        all_oh_bonds = []
        for mol in range(n_molecules):
            oxygen = mol * 4 + 1
            h1 = mol * 4 + 2
            h2 = mol * 4 + 3
            all_oh_bonds.extend([oxygen, h1, oxygen, h2])
        
        for i, atom in enumerate(all_oh_bonds):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(all_oh_bonds) % 15 != 0:
            f.write("\n")
        f.write("\n")
    
    print(f"Generated ice_bonds.ndx for {n_molecules} water molecules ({n_molecules * 4} atoms)")
    print(f"\nGroups created:")
    print(f"  - System: all {n_molecules * 4} atoms")
    print(f"  - Oxygens: {n_molecules} atoms")
    print(f"  - H1_atoms: {n_molecules} atoms")
    print(f"  - H2_atoms: {n_molecules} atoms")
    print(f"  - All_Hydrogens: {n_molecules * 2} atoms")
    print(f"  - Virtual_Sites: {n_molecules} atoms")
    print(f"  - OH1_bonds: {n_molecules * 2} atoms ({n_molecules} bonds)")
    print(f"  - OH2_bonds: {n_molecules * 2} atoms ({n_molecules} bonds)")
    print(f"  - All_OH_bonds: {n_molecules * 4} atoms ({n_molecules * 2} bonds)")

if __name__ == "__main__":
    import sys
    
    # Allow command-line argument for number of molecules
    if len(sys.argv) > 1:
        try:
            n_mol = int(sys.argv[1])
            generate_ice_bond_index(n_mol)
        except ValueError:
            print(f"Error: Invalid number of molecules: {sys.argv[1]}")
            print("Usage: python Ice_bonds.py [number_of_molecules]")
            sys.exit(1)
    else:
        generate_ice_bond_index(432)
