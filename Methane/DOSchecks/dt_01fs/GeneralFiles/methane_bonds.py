#!/usr/bin/env python3
"""
Generate index file for methane C-H bonds
Updated for flexible number of molecules
"""

def generate_bond_index(n_molecules=200):
    """Generate index file for methane C-H bonds
    
    Args:
        n_molecules: Number of CH4 molecules (default: 200)
    """
    
    with open('methane_bonds.ndx', 'w') as f:
        # System
        f.write("[ System ]\n")
        atoms = list(range(1, n_molecules * 5 + 1))
        for i, atom in enumerate(atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:  # 15 atoms per line
                f.write("\n")
        if len(atoms) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All carbons
        f.write("[ Carbons ]\n")
        carbons = [i * 5 + 1 for i in range(n_molecules)]
        for i, atom in enumerate(carbons):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(carbons) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # All hydrogens
        f.write("[ Hydrogens ]\n")
        hydrogens = []
        for mol in range(n_molecules):
            for h in range(1, 5):  # H1, H2, H3, H4
                hydrogens.append(mol * 5 + 1 + h)
        for i, atom in enumerate(hydrogens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        if len(hydrogens) % 15 != 0:
            f.write("\n")
        f.write("\n")
        
        # Each type of C-H bond
        bond_types = ['CH1', 'CH2', 'CH3', 'CH4']
        
        for bond_idx, bond_name in enumerate(bond_types):
            f.write(f"[ {bond_name}_bonds ]\n")
            atoms_in_bond = []
            for mol in range(n_molecules):
                carbon = mol * 5 + 1
                hydrogen = mol * 5 + 2 + bond_idx
                atoms_in_bond.extend([carbon, hydrogen])
            
            for i, atom in enumerate(atoms_in_bond):
                f.write(f"{atom} ")
                if (i + 1) % 15 == 0:
                    f.write("\n")
            if len(atoms_in_bond) % 15 != 0:
                f.write("\n")
            f.write("\n")
    
    print(f"Generated methane_bonds.ndx for {n_molecules} molecules ({n_molecules * 5} atoms)")
    print(f"Groups created:")
    print(f"  - System: all {n_molecules * 5} atoms")
    print(f"  - Carbons: {n_molecules} atoms")
    print(f"  - Hydrogens: {n_molecules * 4} atoms")
    print(f"  - CH1_bonds through CH4_bonds: {n_molecules * 2} atoms each")

if __name__ == "__main__":
    import sys
    
    # Allow command-line argument for number of molecules
    if len(sys.argv) > 1:
        try:
            n_mol = int(sys.argv[1])
            generate_bond_index(n_mol)
        except ValueError:
            print(f"Error: Invalid number of molecules: {sys.argv[1]}")
            print("Usage: python methane_bonds.py [number_of_molecules]")
            sys.exit(1)
    else:
        # Default to 200 molecules
        generate_bond_index(200)
