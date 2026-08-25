#!/usr/bin/env python3
"""
Melamine Crystal Structure → GROMACS.gro file
Reads the COD CIF file, builds a supercell, and renames atoms to match the LigParGen topology naming convention.

Input:  2205105.cif (downloaded from https://www.crystallography.net/cod/2205105.html)
Output: melamine_cluster.gro (ready for GROMACS with LigParGen topology)
"""

import numpy as np
from ase.io import read
from ase.build import make_supercell
from ase.geometry.analysis import Analysis
from scipy.spatial.distance import cdist


def find_molecules_by_distance(positions, symbols, bond_cutoff=1.6):
    """
    Find molecules using simple distance-based bonding.
    No periodic boundary conditions — works on a supercell
    where molecules don't cross boundaries.
    
    bond_cutoff: maximum bond length in Angstroms
                 C-N ~ 1.34 A, N-H ~ 1.01 A, so 1.6 A is safe
    """
    n_atoms = len(positions)
    
    # Build adjacency list using pairwise distances
    # For efficiency, only compute for atoms that could be bonded
    adjacency = [[] for _ in range(n_atoms)]
    
    # Use element-pair specific cutoffs
    cutoffs = {
        ('C', 'N'): 1.50,
        ('N', 'C'): 1.50,
        ('N', 'H'): 1.15,
        ('H', 'N'): 1.15,
        ('C', 'C'): 1.60,
        ('N', 'N'): 1.60,
        ('C', 'H'): 1.20,
        ('H', 'C'): 1.20,
        ('H', 'H'): 0.90,
    }
    
    # Compute all pairwise distances
    # For large systems, do this in chunks to save memory
    if n_atoms < 5000:
        dist_matrix = cdist(positions, positions)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pair = (symbols[i], symbols[j])
                cutoff = cutoffs.get(pair, 1.6)
                if dist_matrix[i, j] < cutoff:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
    else:
        # For very large systems, use a KD-tree approach
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=1.6)
        for i, j in pairs:
            pair = (symbols[i], symbols[j])
            cutoff = cutoffs.get(pair, 1.6)
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # BFS to find connected components (molecules)
    visited = set()
    molecules = []
    
    for i in range(n_atoms):
        if i in visited:
            continue
        molecule = []
        queue = [i]
        while queue:
            atom_idx = queue.pop(0)
            if atom_idx in visited:
                continue
            visited.add(atom_idx)
            molecule.append(atom_idx)
            for j in adjacency[atom_idx]:
                if j not in visited:
                    queue.append(j)
        molecules.append(sorted(molecule))
    
    return molecules, adjacency


def assign_ligpargen_names(positions, symbols, molecule_indices, adjacency):
    """
    Given a list of atom indices belonging to one melamine molecule,
    assign LigParGen atom names based on connectivity analysis.
    
    LigParGen connectivity:
        N00 (amino) -- C01 (ring) -- N02 (ring) -- C03 (ring) -- N04 (amino)
                        |                            |
                       N08 (ring)                   N05 (ring)
                        |                            |
                       C06 (ring) ---- N05           C06 -- N07 (amino)
                        |
                       N07 (amino)
    
    Returns a dict: {atom_index: ligpargen_name}
    """
    
    mol_set = set(molecule_indices)
    
    # Classify atoms
    carbons = [i for i in molecule_indices if symbols[i] == 'C']
    nitrogens = [i for i in molecule_indices if symbols[i] == 'N']
    hydrogens = [i for i in molecule_indices if symbols[i] == 'H']
    
    if len(carbons) != 3 or len(nitrogens) != 6 or len(hydrogens) != 6:
        print(f"  WARNING: Unexpected composition: {len(carbons)}C, "
              f"{len(nitrogens)}N, {len(hydrogens)}H")
        return None
    
    # For each atom, find its neighbors within the molecule
    neighbors = {}
    for idx in molecule_indices:
        neighbors[idx] = [j for j in adjacency[idx] if j in mol_set]
    
    # Classify nitrogens
    n_info = {}
    for n_idx in nitrogens:
        c_neighbors = [j for j in neighbors[n_idx] if symbols[j] == 'C']
        h_neighbors = [j for j in neighbors[n_idx] if symbols[j] == 'H']
        is_ring = len(c_neighbors) == 2 and len(h_neighbors) == 0
        is_amino = len(c_neighbors) == 1 and len(h_neighbors) == 2
        n_info[n_idx] = {
            'c_neighbors': c_neighbors,
            'h_neighbors': h_neighbors,
            'is_ring': is_ring,
            'is_amino': is_amino
        }
    
    ring_nitrogens = [n for n in nitrogens if n_info[n]['is_ring']]
    amino_nitrogens = [n for n in nitrogens if n_info[n]['is_amino']]
    
    if len(ring_nitrogens) != 3 or len(amino_nitrogens) != 3:
        print(f"  WARNING: Found {len(ring_nitrogens)} ring N and "
              f"{len(amino_nitrogens)} amino N (expected 3 each)")
        for n_idx in nitrogens:
            print(f"    N[{n_idx}]: neighbors={neighbors[n_idx]}, "
                  f"C={n_info[n_idx]['c_neighbors']}, "
                  f"H={n_info[n_idx]['h_neighbors']}")
        return None
    
    # For each carbon, find its nitrogen neighbors
    c_info = {}
    for c_idx in carbons:
        ring_n = [j for j in neighbors[c_idx] if j in ring_nitrogens]
        amino_n = [j for j in neighbors[c_idx] if j in amino_nitrogens]
        c_info[c_idx] = {'ring_n': ring_n, 'amino_n': amino_n}
    
    # Assign names by tracing the ring
    # Start: pick first amino nitrogen → N00
    n00 = amino_nitrogens[0]
    c01 = n_info[n00]['c_neighbors'][0]
    
    # C01 has two ring nitrogen neighbors
    c01_ring_ns = c_info[c01]['ring_n']
    
    if len(c01_ring_ns) != 2:
        print(f"  WARNING: C01 has {len(c01_ring_ns)} ring N neighbors")
        return None
    
    # Try both possible ring traversal directions
    for n02_candidate, n08_candidate in [(c01_ring_ns[0], c01_ring_ns[1]),
                                          (c01_ring_ns[1], c01_ring_ns[0])]:
        # N02 → C03
        n02_carbons = [c for c in n_info[n02_candidate]['c_neighbors'] if c != c01]
        if len(n02_carbons) != 1:
            continue
        c03_candidate = n02_carbons[0]
        
        # C03 → N04 (amino)
        n04_candidates = c_info[c03_candidate]['amino_n']
        if len(n04_candidates) != 1:
            continue
        n04_candidate = n04_candidates[0]
        
        # C03 → N05 (ring, not N02)
        n05_candidates = [n for n in c_info[c03_candidate]['ring_n']
                          if n != n02_candidate]
        if len(n05_candidates) != 1:
            continue
        n05_candidate = n05_candidates[0]
        
        # N05 → C06 (not C03)
        c06_candidates = [c for c in n_info[n05_candidate]['c_neighbors']
                          if c != c03_candidate]
        if len(c06_candidates) != 1:
            continue
        c06_candidate = c06_candidates[0]
        
        # C06 → N07 (amino)
        n07_candidates = c_info[c06_candidate]['amino_n']
        if len(n07_candidates) != 1:
            continue
        n07_candidate = n07_candidates[0]
        
        # Verify: N08 should be a ring N neighbor of C06
        if n08_candidate not in c_info[c06_candidate]['ring_n']:
            continue
        
        # Success — build the name map
        name_map = {
            n00: 'N00',
            c01: 'C01',
            n02_candidate: 'N02',
            c03_candidate: 'C03',
            n04_candidate: 'N04',
            n05_candidate: 'N05',
            c06_candidate: 'C06',
            n07_candidate: 'N07',
            n08_candidate: 'N08',
        }
        
        # Assign hydrogens
        for h_idx in n_info[n00]['h_neighbors']:
            if 'H09' not in name_map.values():
                name_map[h_idx] = 'H09'
            else:
                name_map[h_idx] = 'H0A'
        
        for h_idx in n_info[n04_candidate]['h_neighbors']:
            if 'H0B' not in name_map.values():
                name_map[h_idx] = 'H0B'
            else:
                name_map[h_idx] = 'H0C'
        
        for h_idx in n_info[n07_candidate]['h_neighbors']:
            if 'H0D' not in name_map.values():
                name_map[h_idx] = 'H0D'
            else:
                name_map[h_idx] = 'H0E'
        
        return name_map
    
    print("  WARNING: Could not trace ring connectivity")
    return None


def write_gromacs_gro(filename, positions, molecules, name_maps, box_padding=0.5):
    """
    Write a GROMACS.gro file with LigParGen atom naming.
    Positions in Angstroms, output in nm.
    """
    
    # Convert to nm
    pos_nm = positions / 10.0
    
    # Collect all atom indices
    all_indices = []
    for mol in molecules:
        all_indices.extend(mol)
    
    mol_positions = pos_nm[all_indices]
    min_coords = mol_positions.min(axis=0)
    max_coords = mol_positions.max(axis=0)
    
    # Shift so minimum coordinate is at box_padding
    shift = -min_coords + box_padding
    
    # Box dimensions
    box = max_coords - min_coords + 2 * box_padding
    
    # Atom ordering within each molecule (LigParGen order)
    atom_order = ['N00', 'C01', 'N02', 'C03', 'N04', 'N05', 'C06', 'N07', 'N08',
                  'H09', 'H0A', 'H0B', 'H0C', 'H0D', 'H0E']
    
    total_atoms = sum(len(mol) for mol in molecules)
    
    with open(filename, 'w') as f:
        f.write("Melamine crystal cluster (from COD 2205105 + LigParGen naming)\n")
        f.write(f"{total_atoms:5d}\n")
        
        atom_counter = 0
        for mol_idx, (mol, name_map) in enumerate(zip(molecules, name_maps), 1):
            if name_map is None:
                print(f"  Skipping molecule {mol_idx} (naming failed)")
                continue
            
            reverse_map = {v: k for k, v in name_map.items()}
            
            for atom_name in atom_order:
                if atom_name not in reverse_map:
                    print(f"  WARNING: {atom_name} not found in molecule {mol_idx}")
                    continue
                
                atom_idx = reverse_map[atom_name]
                atom_counter += 1
                
                x, y, z = pos_nm[atom_idx] + shift
                
                f.write(f"{mol_idx:5d}MEL  {atom_name:>5s}{atom_counter:5d}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
        
        f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")
    
    print(f"\nWrote {atom_counter} atoms in {len(molecules)} molecules to {filename}")
    print(f"Box dimensions: {box[0]:.3f} x {box[1]:.3f} x {box[2]:.3f} nm")


def select_cluster(positions, molecules, n_molecules=4):
    """
    Select a cluster of n_molecules centered on the molecule
    closest to the geometric center of all molecules.
    """
    
    # Center of mass of each molecule
    mol_centers = []
    for mol in molecules:
        mol_pos = positions[mol]
        mol_centers.append(mol_pos.mean(axis=0))
    mol_centers = np.array(mol_centers)
    
    # Find molecule closest to the overall center
    overall_center = mol_centers.mean(axis=0)
    distances_to_center = np.linalg.norm(mol_centers - overall_center, axis=1)
    central_mol_idx = np.argmin(distances_to_center)
    
    # Find nearest neighbors
    central_pos = mol_centers[central_mol_idx]
    distances_to_central = np.linalg.norm(mol_centers - central_pos, axis=1)
    
    sorted_indices = np.argsort(distances_to_central)
    selected = sorted_indices[:n_molecules]
    
    print(f"\nSelected {n_molecules} molecules:")
    print(f"  Central molecule index: {central_mol_idx} "
          f"(center at {central_pos[0]:.2f}, {central_pos[1]:.2f}, "
          f"{central_pos[2]:.2f} A)")
    print(f"  Neighbor distances:")
    for i, idx in enumerate(selected):
        print(f"    Molecule {idx}: {distances_to_central[idx]:.2f} A "
              f"({len(molecules[idx])} atoms)")
    
    return [molecules[i] for i in selected]


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    cif_file = "2205105.cif"
    output_file = "melamine_cluster.gro"
    supercell_size = [3, 3, 3]
    n_molecules = 10
    box_padding = 0.8  # nm
    # ============================================================
    
    print(f"Reading CIF file: {cif_file}")
    try:
        crystal = read(cif_file)
    except Exception as e:
        print(f"ERROR reading CIF file: {e}")
        print("Make sure 2205105.cif is in the current directory.")
        print("Download from: https://www.crystallography.net/cod/2205105.cif")
        return
    
    print(f"Unit cell contains {len(crystal)} atoms")
    print(f"Cell parameters: a={crystal.cell.cellpar()[0]:.3f}, "
          f"b={crystal.cell.cellpar()[1]:.3f}, "
          f"c={crystal.cell.cellpar()[2]:.3f} A")
    print(f"Angles: alpha={crystal.cell.cellpar()[3]:.1f}, "
          f"beta={crystal.cell.cellpar()[4]:.1f}, "
          f"gamma={crystal.cell.cellpar()[5]:.1f} deg")
    
    # Build supercell
    print(f"\nBuilding {supercell_size} supercell...")
    P = np.diag(supercell_size)
    supercell = make_supercell(crystal, P)
    n_atoms = len(supercell)
    print(f"Supercell contains {n_atoms} atoms")
    
    # Get positions and symbols as plain arrays
    # (no more periodic boundary issues)
    positions = supercell.get_positions()
    symbols = supercell.get_chemical_symbols()
    
    print(f"Elements: {set(symbols)}")
    print(f"  C: {symbols.count('C')}, N: {symbols.count('N')}, "
          f"H: {symbols.count('H')}")
    
    # Find molecules using simple distance-based bonding
    print("\nIdentifying molecules by distance-based bonding...")
    molecules, adjacency = find_molecules_by_distance(positions, symbols)
    
    # Filter to only melamine-sized molecules (15 atoms)
    valid_molecules = []
    invalid_count = 0
    for mol in molecules:
        if len(mol) == 15:
            # Verify composition
            mol_symbols = [symbols[i] for i in mol]
            n_C = mol_symbols.count('C')
            n_N = mol_symbols.count('N')
            n_H = mol_symbols.count('H')
            if n_C == 3 and n_N == 6 and n_H == 6:
                valid_molecules.append(mol)
            else:
                invalid_count += 1
        elif len(mol) > 1:
            invalid_count += 1
    
    print(f"Found {len(valid_molecules)} valid melamine molecules "
          f"({invalid_count} invalid/fragments)")
    
    if len(valid_molecules) < n_molecules:
        print(f"ERROR: Not enough valid molecules ({len(valid_molecules)}) "
              f"for requested cluster size ({n_molecules})")
        print("Try increasing supercell_size or checking the CIF file.")
        return
    
    # Select cluster
    selected_molecules = select_cluster(positions, valid_molecules, n_molecules)
    
    # Assign LigParGen names
    print("\nAssigning LigParGen atom names...")
    name_maps = []
    for i, mol in enumerate(selected_molecules):
        name_map = assign_ligpargen_names(positions, symbols, mol, adjacency)
        if name_map is not None:
            print(f"  Molecule {i+1}: OK ({len(name_map)} atoms named)")
        else:
            print(f"  Molecule {i+1}: FAILED")
        name_maps.append(name_map)
    
    # Check all succeeded
    if any(nm is None for nm in name_maps):
        print("\nERROR: Some molecules could not be named. Check warnings above.")
        return
    
    # Write.gro file
    print(f"\nWriting: {output_file}")
    write_gromacs_gro(output_file, positions, selected_molecules, 
                      name_maps, box_padding)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Visualize {output_file} in VMD or PyMOL")
    print(f"2. Verify hydrogen bonds between molecules")
    print(f"3. Use with your LigParGen topology (topol.top):")
    print(f"")
    print(f"   [ system ]")
    print(f"   Melamine cluster")
    print(f"   ")
    print(f"   [ molecules ]")
    print(f"   MEL    {n_molecules}")
    print(f"")
    print(f"4. Energy minimize:")
    print(f"   gmx grompp -f em.mdp -c {output_file} -p topol.top -o em.tpr")
    print(f"   gmx mdrun -deffnm em")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
