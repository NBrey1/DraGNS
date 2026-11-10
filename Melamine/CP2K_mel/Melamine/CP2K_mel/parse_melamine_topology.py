{\rtf1\ansi\ansicpg1252\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import networkx as nx\
from collections import defaultdict\
import re\
import os\
\
\
def parse_topology(filename):\
    data = \{\
        'moleculetype': None,\
        'atoms': [],\
        'bonds': [],\
        'angles': [],\
        'proper_dihedrals': [],\
        'improper_dihedrals': []\
    \}\
\
    current_section = None\
    dihedral_block_count = 0\
\
    with open(filename, 'r') as f:\
        for line in f:\
            line = line.strip()\
            if not line or line.startswith(';'):\
                continue\
\
            section_match = re.match(r'\\[\\s*(\\w+)\\s*\\]', line)\
            if section_match:\
                section_name = section_match.group(1)\
                if section_name == 'dihedrals':\
                    dihedral_block_count += 1\
                current_section = (section_name, dihedral_block_count)\
                continue\
\
            tokens = line.split(';')[0].split()\
            if not tokens:\
                continue\
\
            if current_section is None:\
                continue\
\
            sec = current_section[0]\
\
            if sec == 'moleculetype':\
                data['moleculetype'] = (tokens[0], int(tokens[1]))\
\
            elif sec == 'atoms':\
                data['atoms'].append(\{\
                    'nr': int(tokens[0]),\
                    'type': tokens[1],\
                    'resnr': int(tokens[2]),\
                    'resname': tokens[3],\
                    'atom': tokens[4],\
                    'cgnr': int(tokens[5]),\
                    'charge': float(tokens[6]),\
                    'mass': float(tokens[7])\
                \})\
\
            elif sec == 'bonds':\
                data['bonds'].append(\{\
                    'ai': int(tokens[0]),\
                    'aj': int(tokens[1]),\
                    'funct': int(tokens[2]),\
                    'r0': float(tokens[3]),\
                    'kb': float(tokens[4])\
                \})\
\
            elif sec == 'angles':\
                data['angles'].append(\{\
                    'ai': int(tokens[0]),\
                    'aj': int(tokens[1]),\
                    'ak': int(tokens[2]),\
                    'funct': int(tokens[3]),\
                    'theta0': float(tokens[4]),\
                    'ktheta': float(tokens[5])\
                \})\
\
            elif sec == 'dihedrals':\
                funct = int(tokens[4])\
                if funct == 3:\
                    coeffs = [float(x) for x in tokens[5:11]]\
                    data['proper_dihedrals'].append(\{\
                        'ai': int(tokens[0]),\
                        'aj': int(tokens[1]),\
                        'ak': int(tokens[2]),\
                        'al': int(tokens[3]),\
                        'funct': funct,\
                        'coeffs': coeffs\
                    \})\
                elif funct == 4:\
                    data['improper_dihedrals'].append(\{\
                        'ai': int(tokens[0]),\
                        'aj': int(tokens[1]),\
                        'ak': int(tokens[2]),\
                        'al': int(tokens[3]),\
                        'funct': funct,\
                        'phi0': float(tokens[5]),\
                        'kphi': float(tokens[6])\
                    \})\
\
    return data\
\
\
def classify_atoms(data):\
    type_map = \{\
        'opls_903': ('N', 'ring_nitrogen'),\
        'opls_902': ('C', 'ring_carbon'),\
        'opls_900': ('N', 'amino_nitrogen'),\
        'opls_901': ('H', 'amino_hydrogen')\
    \}\
    classified = []\
    for atom in data['atoms']:\
        element, role = type_map.get(atom['type'], ('?', 'unknown'))\
        classified.append(\{\
            **atom,\
            'element': element,\
            'role': role\
        \})\
    return classified\
\
\
def build_connectivity_graph(data):\
    G = nx.Graph()\
    classified = classify_atoms(data)\
    for atom in classified:\
        G.add_node(atom['nr'], **atom)\
    for bond in data['bonds']:\
        ai, aj = bond['ai'], bond['aj']\
        ni = G.nodes[ai]['atom']\
        nj = G.nodes[aj]['atom']\
        ei = G.nodes[ai]['element']\
        ej = G.nodes[aj]['element']\
        if ei == 'H' or ej == 'H':\
            bond_type = 'N-H'\
        elif (G.nodes[ai]['role'] == 'amino_nitrogen' or\
              G.nodes[aj]['role'] == 'amino_nitrogen'):\
            bond_type = 'exocyclic_CN'\
        else:\
            bond_type = 'ring'\
        G.add_edge(ai, aj, r0=bond['r0'], kb=bond['kb'], bond_type=bond_type)\
    return G\
\
\
def validate_topology(data, G):\
    results = []\
\
    results.append(('Atom count == 15', len(data['atoms']) == 15))\
    results.append(('Bond count == 15', len(data['bonds']) == 15))\
\
    total_charge = sum(a['charge'] for a in data['atoms'])\
    results.append((f'Charge neutrality (sum=\{total_charge:.4f\})', abs(total_charge) < 1e-6))\
\
    results.append(('Graph is connected', nx.is_connected(G)))\
\
    cycles = nx.cycle_basis(G)\
    ring_6 = [c for c in cycles if len(c) == 6]\
    results.append(('Contains 6-membered ring', len(ring_6) >= 1))\
\
    valence_ok = True\
    for node in G.nodes():\
        deg = G.degree(node)\
        element = G.nodes[node]['element']\
        if element == 'C' and deg != 3:\
            valence_ok = False\
        elif element == 'N' and deg not in [2, 3]:\
            valence_ok = False\
        elif element == 'H' and deg != 1:\
            valence_ok = False\
    results.append(('Valence check', valence_ok))\
\
    results.append(('Angle count == 21', len(data['angles']) == 21))\
\
    max_idx = len(data['atoms'])\
    idx_ok = True\
    for bond in data['bonds']:\
        if bond['ai'] < 1 or bond['ai'] > max_idx or bond['aj'] < 1 or bond['aj'] > max_idx:\
            idx_ok = False\
    results.append(('Bond indices valid', idx_ok))\
\
    return results\
\
\
def get_summary(data):\
    classified = classify_atoms(data)\
    elements = defaultdict(int)\
    for atom in classified:\
        elements[atom['element']] += 1\
\
    summary = \{\
        'formula': ''.join(f"\{el\}\{cnt\}" for el, cnt in sorted(elements.items())),\
        'n_atoms': len(data['atoms']),\
        'n_bonds': len(data['bonds']),\
        'n_angles': len(data['angles']),\
        'n_proper_dihedrals': len(data['proper_dihedrals']),\
        'n_improper_dihedrals': len(data['improper_dihedrals']),\
        'total_charge': sum(a['charge'] for a in data['atoms']),\
        'total_mass': sum(a['mass'] for a in data['atoms']),\
        'elements': dict(elements),\
        'bond_types': \{\
            'ring_CN': sum(1 for b in data['bonds']\
                          if b['r0'] == 0.1338),\
            'exocyclic_CN': sum(1 for b in data['bonds']\
                                if b['r0'] == 0.1325),\
            'NH': sum(1 for b in data['bonds']\
                      if b['r0'] == 0.1010)\
        \}\
    \}\
    return summary\
\
\
def print_summary(data):\
    s = get_summary(data)\
    print(f"Molecular formula:     \{s['formula']\}")\
    print(f"Number of atoms:       \{s['n_atoms']\}")\
    print(f"Total mass:            \{s['total_mass']:.3f\} amu")\
    print(f"Total charge:          \{s['total_charge']:.6f\} e")\
    print(f"Bonds:                 \{s['n_bonds']\}  "\
          f"(ring C-N: \{s['bond_types']['ring_CN']\}, "\
          f"exocyclic C-N: \{s['bond_types']['exocyclic_CN']\}, "\
          f"N-H: \{s['bond_types']['NH']\})")\
    print(f"Angles:                \{s['n_angles']\}")\
    print(f"Proper dihedrals:      \{s['n_proper_dihedrals']\}")\
    print(f"Improper dihedrals:    \{s['n_improper_dihedrals']\}")\
\
\
def print_atoms(data):\
    print(f"\{'Nr':>3s\}  \{'Atom':>4s\}  \{'Type':<10s\}  \{'Element':>7s\}  "\
          f"\{'Role':<16s\}  \{'Charge':>7s\}  \{'Mass':>8s\}")\
    print("-" * 68)\
    for atom in classify_atoms(data):\
        print(f"\{atom['nr']:3d\}  \{atom['atom']:>4s\}  \{atom['type']:<10s\}  "\
              f"\{atom['element']:>7s\}  \{atom['role']:<16s\}  "\
              f"\{atom['charge']:7.3f\}  \{atom['mass']:8.3f\}")\
\
\
def print_bonds(data):\
    print(f"\{'ai':>3s\}  \{'aj':>3s\}  \{'r0 (nm)':>8s\}  \{'r0 (\'c5)':>8s\}  "\
          f"\{'kb':>10s\}")\
    print("-" * 40)\
    for bond in data['bonds']:\
        print(f"\{bond['ai']:3d\}  \{bond['aj']:3d\}  \{bond['r0']:8.4f\}  "\
              f"\{bond['r0']*10:8.3f\}  \{bond['kb']:10.1f\}")\
\
\
def print_angles(data):\
    print(f"\{'ai':>3s\}  \{'aj':>3s\}  \{'ak':>3s\}  \{'theta0 (deg)':>12s\}  "\
          f"\{'ktheta':>10s\}")\
    print("-" * 45)\
    for angle in data['angles']:\
        print(f"\{angle['ai']:3d\}  \{angle['aj']:3d\}  \{angle['ak']:3d\}  "\
              f"\{angle['theta0']:12.1f\}  \{angle['ktheta']:10.1f\}")\
\
\
def print_validation(data, G):\
    results = validate_topology(data, G)\
    for label, passed in results:\
        status = "PASS" if passed else "FAIL"\
        print(f"  [\{status\}]  \{label\}")\
\
\
if __name__ == '__main__':\
    top_file = 'melamine.top'\
    if not os.path.exists(top_file):\
        print(f"Error: \{top_file\} not found in current directory.")\
        raise SystemExit(1)\
\
    data = parse_topology(top_file)\
    G = build_connectivity_graph(data)\
\
    print("=" * 60)\
    print("  MELAMINE TOPOLOGY SUMMARY")\
    print("=" * 60)\
    print()\
    print_summary(data)\
    print()\
\
    print("--- Atoms ---")\
    print_atoms(data)\
    print()\
\
    print("--- Bonds ---")\
    print_bonds(data)\
    print()\
\
    print("--- Angles ---")\
    print_angles(data)\
    print()\
\
    print("--- Validation ---")\
    print_validation(data, G)\
}