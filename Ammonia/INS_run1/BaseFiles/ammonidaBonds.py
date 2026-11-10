#!/usr/bin/env python3
# ammonia_bonds.py

def generate_bond_index(n_molecules=500):
    """Generate index file for ammonia N-H bonds"""
    
    with open('ammonia_bonds.ndx', 'w') as f:
        # System
        f.write("[ System ]\n")
        atoms = list(range(1, n_molecules * 4 + 1))  # 4 atoms per NH3
        for i, atom in enumerate(atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:  # 15 atoms per line
                f.write("\n")
        f.write("\n\n")
        
        # All nitrogens
        f.write("[ Nitrogens ]\n")
        nitrogens = [i * 4 + 1 for i in range(n_molecules)]  # N is first atom in each molecule
        for i, atom in enumerate(nitrogens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n\n")
        
        # All hydrogens
        f.write("[ Hydrogens ]\n")
        hydrogens = []
        for mol in range(n_molecules):
            for h in range(1, 4):  # H1, H2, H3
                hydrogens.append(mol * 4 + 1 + h)
        for i, atom in enumerate(hydrogens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n\n")
        
        # Individual hydrogen types
        hydrogen_types = ['H1', 'H2', 'H3']
        
        for h_idx, h_name in enumerate(hydrogen_types):
            f.write(f"[ {h_name}_atoms ]\n")
            h_atoms = []
            for mol in range(n_molecules):
                hydrogen = mol * 4 + 2 + h_idx  # H1=+2, H2=+3, H3=+4 from N
                h_atoms.append(hydrogen)
            
            for i, atom in enumerate(h_atoms):
                f.write(f"{atom} ")
                if (i + 1) % 15 == 0:
                    f.write("\n")
            f.write("\n\n")
        
        # Individual nitrogen atoms (for consistency with your methane script)
        f.write("[ N_atoms ]\n")
        for i, atom in enumerate(nitrogens):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n\n")
        
        # Each type of N-H bond
        bond_types = ['NH1', 'NH2', 'NH3']
        
        for bond_idx, bond_name in enumerate(bond_types):
            f.write(f"[ {bond_name}_bonds ]\n")
            atoms_in_bond = []
            for mol in range(n_molecules):
                nitrogen = mol * 4 + 1
                hydrogen = mol * 4 + 2 + bond_idx
                atoms_in_bond.extend([nitrogen, hydrogen])
            
            for i, atom in enumerate(atoms_in_bond):
                f.write(f"{atom} ")
                if (i + 1) % 15 == 0:
                    f.write("\n")
            f.write("\n\n")
        
        # All N-H bonds combined (useful for total bond analysis)
        f.write("[ All_NH_bonds ]\n")
        all_bond_atoms = []
        for mol in range(n_molecules):
            nitrogen = mol * 4 + 1
            for h_idx in range(3):  # 3 hydrogens
                hydrogen = mol * 4 + 2 + h_idx
                all_bond_atoms.extend([nitrogen, hydrogen])
        
        for i, atom in enumerate(all_bond_atoms):
            f.write(f"{atom} ")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n\n")

if __name__ == "__main__":
    # Allow command line argument for number of molecules
    import sys
    
    n_mols = 500  # default
    if len(sys.argv) > 1:
        try:
            n_mols = int(sys.argv[1])
        except ValueError:
            print("Usage: python3 ammonia_bonds.py [number_of_molecules]")
            print("Using default: 500 molecules")
    
    generate_bond_index(n_mols)
    print(f"Generated ammonia_bonds.ndx for {n_mols} NH3 molecules")
    
    # Print summary of groups created
    print("\nIndex groups created:")
    print("- System: All atoms")
    print("- Nitrogens: All N atoms")
    print("- Hydrogens: All H atoms") 
    print("- H1_atoms, H2_atoms, H3_atoms: Individual H types")
    print("- N_atoms: All N atoms (alias)")
    print("- NH1_bonds, NH2_bonds, NH3_bonds: Individual N-H bonds")
    print("- All_NH_bonds: All N-H bonds combined")
    
    # Calculate totals
    total_atoms = n_mols * 4
    total_bonds = n_mols * 3
    print(f"\nSystem totals:")
    print(f"- Total atoms: {total_atoms}")
    print(f"- Total N-H bonds: {total_bonds}")
   # print(f"- Vibrational modes: {3 * total_atoms - 6} (for solid) or {3 * total_atoms - 3} (for molecules)")
