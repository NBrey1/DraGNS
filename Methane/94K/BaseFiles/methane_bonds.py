#!/usr/bin/env python3
#methane_bonds.py

def generate_bond_index(n_molecules=50):
    """Generate index file for methane C-H bonds"""
    
    with open('methane_bonds.ndx', 'w') as f:
        # System
        f.write("[ System ]\n")
        atoms = list(range(1, n_molecules * 5 + 1))
        for i, atom in enumerate(atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:  # 15 atoms per line
                f.write("\n")
        f.write("\n\n")
        
        # All carbons
        f.write("[ Carbons ]\n")
        carbons = [i * 5 + 1 for i in range(n_molecules)]
        for i, atom in enumerate(carbons):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n\n")
        
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
        f.write("\n\n")
        
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
            f.write("\n\n")

if __name__ == "__main__":
    generate_bond_index(50)
    print("Generated methane_bonds.ndx")
