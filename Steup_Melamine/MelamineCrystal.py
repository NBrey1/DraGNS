#!/usr/bin/env python3
"""
Build a periodic melamine crystal supercell for GROMACS.
Uses the COD 2205105 CIF file.
FIXED: Relaxed bond cutoffs for supercell geometry.
"""

import numpy as np
from ase.io import read
from ase.build import make_supercell
from scipy.spatial import cKDTree


def find_molecules_by_distance(positions, symbols):
    """
    Find molecules using distance-based bonding.
    Relaxed cutoffs to handle slight geometry variations
    in supercell expansion.
    """
    n_atoms = len(positions)
    
    # Relaxed cutoffs (Angstroms) - slightly larger to catch
    # all intramolecular bonds after supercell construction
    cutoffs = {
        ('C', 'N'): 1.55,
        ('N', 'C'): 1.55,
        ('N', 'H'): 1.20,
        ('H', 'N'): 1.20,
        ('C', 'H'): 1.25,
        ('H', 'C'): 1.25,
        ('C', 'C'): 1.65,
        ('N', 'N'): 1.65,
        ('H', 'H'): 0.95,
    }
    
    max_cutoff = 1.65
    
    adjacency = [[] for _ in range(n_atoms)]
    
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=max_cutoff)
    for i, j in pairs:
        pair = (symbols[i], symbols[j])
        cutoff = cutoffs.get(pair, max_cutoff)
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < cutoff:
            adjacency[i].append(j)
            adjacency[j].append(i)
    
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
    """Assign LigParGen atom names based on connectivity."""
    mol_set = set(molecule_indices)
    
    carbons = [i for i in molecule_indices if symbols[i] == 'C']
    nitrogens = [i for i in molecule_indices if symbols[i] == 'N']
    hydrogens = [i for i in molecule_indices if symbols[i] == 'H']
    
    if len(carbons) != 3 or len(nitrogens) != 6 or len(hydrogens) != 6:
        return None
    
    neighbors = {}
    for idx in molecule_indices:
        neighbors[idx] = [j for j in adjacency[idx] if j in mol_set]
    
    n_info = {}
    for n_idx in nitrogens:
        c_neighbors = [j for j in neighbors[n_idx] if symbols[j] == 'C']
        h_neighbors = [j for j in neighbors[n_idx] if symbols[j] == 'H']
        n_info[n_idx] = {
            'c_neighbors': c_neighbors,
            'h_neighbors': h_neighbors,
            'is_ring': len(c_neighbors) == 2 and len(h_neighbors) == 0,
            'is_amino': len(c_neighbors) == 1 and len(h_neighbors) == 2
        }
    
    ring_nitrogens = [n for n in nitrogens if n_info[n]['is_ring']]
    amino_nitrogens = [n for n in nitrogens if n_info[n]['is_amino']]
    
    if len(ring_nitrogens) != 3 or len(amino_nitrogens) != 3:
        return None
    
    c_info = {}
    for c_idx in carbons:
        ring_n = [j for j in neighbors[c_idx] if j in ring_nitrogens]
        amino_n = [j for j in neighbors[c_idx] if j in amino_nitrogens]
        c_info[c_idx] = {'ring_n': ring_n, 'amino_n': amino_n}
    
    n00 = amino_nitrogens[0]
    c01 = n_info[n00]['c_neighbors'][0]
    c01_ring_ns = c_info[c01]['ring_n']
    
    if len(c01_ring_ns) != 2:
        return None
    
    for n02_candidate, n08_candidate in [(c01_ring_ns[0], c01_ring_ns[1]),
                                          (c01_ring_ns[1], c01_ring_ns[0])]:
        n02_carbons = [c for c in n_info[n02_candidate]['c_neighbors'] 
                       if c != c01]
        if len(n02_carbons) != 1:
            continue
        c03_candidate = n02_carbons[0]
        
        n04_candidates = c_info[c03_candidate]['amino_n']
        if len(n04_candidates) != 1:
            continue
        n04_candidate = n04_candidates[0]
        
        n05_candidates = [n for n in c_info[c03_candidate]['ring_n']
                          if n != n02_candidate]
        if len(n05_candidates) != 1:
            continue
        n05_candidate = n05_candidates[0]
        
        c06_candidates = [c for c in n_info[n05_candidate]['c_neighbors']
                          if c != c03_candidate]
        if len(c06_candidates) != 1:
            continue
        c06_candidate = c06_candidates[0]
        
        n07_candidates = c_info[c06_candidate]['amino_n']
        if len(n07_candidates) != 1:
            continue
        n07_candidate = n07_candidates[0]
        
        if n08_candidate not in c_info[c06_candidate]['ring_n']:
            continue
        
        name_map = {
            n00: 'N00', c01: 'C01', n02_candidate: 'N02',
            c03_candidate: 'C03', n04_candidate: 'N04',
            n05_candidate: 'N05', c06_candidate: 'C06',
            n07_candidate: 'N07', n08_candidate: 'N08',
        }
        
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
    
    return None


def write_gromacs_gro(filename, positions, symbols, cell,
                      molecules, name_maps):
    """Write GROMACS.gro file for a periodic supercell."""
    pos_nm = positions / 10.0
    cell_nm = cell / 10.0
    
    ax, ay, az = cell_nm[0]
    bx, by, bz = cell_nm[1]
    cx, cy, cz = cell_nm[2]
    
    atom_order = ['N00', 'C01', 'N02', 'C03', 'N04', 'N05',
                  'C06', 'N07', 'N08', 'H09', 'H0A', 'H0B',
                  'H0C', 'H0D', 'H0E']
    
    total_atoms = sum(len(mol) for mol in molecules)
    
    with open(filename, 'w') as f:
        f.write("Melamine periodic crystal supercell "
                "(from COD 2205105)\n")
        f.write(f"{total_atoms:5d}\n")
        
        atom_counter = 0
        for mol_idx, (mol, name_map) in enumerate(
                zip(molecules, name_maps), 1):
            if name_map is None:
                continue
            reverse_map = {v: k for k, v in name_map.items()}
            for atom_name in atom_order:
                if atom_name not in reverse_map:
                    continue
                atom_idx = reverse_map[atom_name]
                atom_counter += 1
                x, y, z = pos_nm[atom_idx]
                f.write(f"{mol_idx:5d}MEL  {atom_name:>5s}"
                        f"{atom_counter:5d}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
        
        f.write(f"{ax:10.5f}{by:10.5f}{cz:10.5f}"
                f"{ay:10.5f}{az:10.5f}{bx:10.5f}"
                f"{bz:10.5f}{cx:10.5f}{cy:10.5f}\n")
    
    print(f"\nWrote {atom_counter} atoms in "
          f"{len(molecules)} molecules to {filename}")
    print(f"Box vectors (nm):")
    print(f"  a = ({ax:.4f}, {ay:.4f}, {az:.4f})")
    print(f"  b = ({bx:.4f}, {by:.4f}, {bz:.4f})")
    print(f"  c = ({cx:.4f}, {cy:.4f}, {cz:.4f})")


def main():
    # ============================================================
    # CONFIGURATION
    # ============================================================
    cif_file = "2205105.cif"
    output_file = "melamine_crystal.gro"
    nx, ny, nz = 2, 2, 2
    # ============================================================
    
    print(f"Reading CIF file: {cif_file}")
    crystal = read(cif_file)
    print(f"Unit cell: {len(crystal)} atoms")
    print(f"Cell: a={crystal.cell.cellpar()[0]:.3f}, "
          f"b={crystal.cell.cellpar()[1]:.3f}, "
          f"c={crystal.cell.cellpar()[2]:.3f} A")
    print(f"beta={crystal.cell.cellpar()[4]:.1f} deg")
    
    # Check what elements are in the unit cell
    symbols_uc = crystal.get_chemical_symbols()
    print(f"Unit cell composition: "
          f"C={symbols_uc.count('C')}, "
          f"N={symbols_uc.count('N')}, "
          f"H={symbols_uc.count('H')}")
    
    # Build supercell
    print(f"\nBuilding {nx}x{ny}x{nz} supercell...")
    P = np.diag([nx, ny, nz])
    supercell = make_supercell(crystal, P)
    n_atoms = len(supercell)
    print(f"Supercell: {n_atoms} atoms")
    
    positions = supercell.get_positions()
    symbols = supercell.get_chemical_symbols()
    cell = supercell.cell.array
    
    print(f"Supercell composition: "
          f"C={symbols.count('C')}, "
          f"N={symbols.count('N')}, "
          f"H={symbols.count('H')}")
    
    # ---- DIAGNOSTIC: Check all intramolecular distances ----
    print("\n--- Bond distance diagnostic ---")
    # Check a few C-N distances in the raw positions
    c_indices = [i for i, s in enumerate(symbols) if s == 'C']
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    # Find all C-N distances less than 2.0 A
    cn_dists = []
    tree_c = cKDTree(positions[c_indices])
    tree_n = cKDTree(positions[n_indices])
    results = tree_c.query_ball_tree(tree_n, r=2.0)
    for i, neighbors in enumerate(results):
        for j in neighbors:
            d = np.linalg.norm(positions[c_indices[i]] - 
                              positions[n_indices[j]])
            cn_dists.append(d)
    
    cn_dists = np.array(cn_dists)
    print(f"C-N distances < 2.0 A: {len(cn_dists)}")
    if len(cn_dists) > 0:
        print(f"  Min: {cn_dists.min():.3f} A")
        print(f"  Max: {cn_dists.max():.3f} A")
        # Histogram
        for low, high in [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6), 
                          (1.6, 1.8), (1.8, 2.0)]:
            count = np.sum((cn_dists >= low) & (cn_dists < high))
            print(f"  {low:.1f}-{high:.1f} A: {count}")
    
    # Find all N-H distances less than 1.5 A
    nh_dists = []
    tree_h = cKDTree(positions[h_indices])
    results = tree_n.query_ball_tree(tree_h, r=1.5)
    for i, neighbors in enumerate(results):
        for j in neighbors:
            d = np.linalg.norm(positions[n_indices[i]] - 
                              positions[h_indices[j]])
            nh_dists.append(d)
    
    nh_dists = np.array(nh_dists)
    print(f"\nN-H distances < 1.5 A: {len(nh_dists)}")
    if len(nh_dists) > 0:
        print(f"  Min: {nh_dists.min():.3f} A")
        print(f"  Max: {nh_dists.max():.3f} A")
        for low, high in [(0.8, 1.0), (1.0, 1.1), (1.1, 1.2), 
                          (1.2, 1.3), (1.3, 1.5)]:
            count = np.sum((nh_dists >= low) & (nh_dists < high))
            print(f"  {low:.1f}-{high:.1f} A: {count}")
    
    print("--- End diagnostic ---\n")
    
    # Find molecules
    print("Identifying molecules...")
    molecules, adjacency = find_molecules_by_distance(
        positions, symbols)
    
    # Diagnostic: molecule size distribution
    print("\nMolecule size distribution:")
    size_counts = {}
    for mol in molecules:
        size = len(mol)
        size_counts[size] = size_counts.get(size, 0) + 1
    for size in sorted(size_counts.keys()):
        print(f"  {size_counts[size]} molecules "
              f"with {size} atoms")
    
    # Filter valid molecules
    valid_molecules = []
    for mol in molecules:
        if len(mol) == 15:
            mol_symbols = [symbols[i] for i in mol]
            if (mol_symbols.count('C') == 3 and
                mol_symbols.count('N') == 6 and
                mol_symbols.count('H') == 6):
                valid_molecules.append(mol)
    
    n_expected = 4 * nx * ny * nz
    print(f"\nFound {len(valid_molecules)} valid melamine "
          f"molecules (expected {n_expected})")
    
    if len(valid_molecules) < n_expected:
        print(f"\nWARNING: Missing {n_expected - len(valid_molecules)} "
              f"molecules!")
        print("This is likely due to bonds being split at "
              "periodic boundaries.")
        print("The diagnostic above should show the issue.")
        
        if len(valid_molecules) == 0:
            print("ERROR: No valid molecules found. Exiting.")
            return
        else:
            print(f"Proceeding with {len(valid_molecules)} "
                  f"molecules...")
    
    # Assign names
    print("\nAssigning LigParGen atom names...")
    name_maps = []
    success = 0
    for mol in valid_molecules:
        name_map = assign_ligpargen_names(
            positions, symbols, mol, adjacency)
        if name_map is not None:
            success += 1
        name_maps.append(name_map)
    print(f"  Successfully named {success}/"
          f"{len(valid_molecules)} molecules")
    
    # Filter out failed
    good = [(m, nm) for m, nm in 
             zip(valid_molecules, name_maps) if nm is not None]
    valid_molecules = [m for m, nm in good]
    name_maps = [nm for m, nm in good]
    
    # Write.gro
    print(f"\nWriting: {output_file}")
    write_gromacs_gro(output_file, positions, symbols,
                      cell, valid_molecules, name_maps)
    
    n_mol = len(valid_molecules)
    print(f"\n{'='*60}")
    print("TOPOLOGY (topol.top):")
    print(f"{'='*60}")
    print(f"#include \"oplsaa.ff/forcefield.itp\"")
    print(f"#include \"melamine.itp\"")
    print(f"")
    print(f"[ system ]")
    print(f"Melamine crystal {nx}x{ny}x{nz}")
    print(f"")
    print(f"[ molecules ]")
    print(f"MEL    {n_mol}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
