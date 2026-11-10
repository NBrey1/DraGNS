import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import os

# Read VACF data
def read_xvg(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    return np.array(x), np.array(y)

# Calculate DOS from VACF
def calculate_dos(time, vacf, name, return_normalized=True):
    # Get time step
    dt = time[1] - time[0]  # in ps
    
    # Window function to reduce noise
    window = np.blackman(len(vacf))
    vacf_windowed = vacf * window
    
    # FFT to get power spectrum (DOS)
    n = len(vacf)
    dos_raw = np.abs(rfft(vacf_windowed))
    freq_hz = np.fft.rfftfreq(n, dt * 1e-12)  # Convert ps to s for frequency in Hz
    
    # Convert frequency from Hz to cm^-1
    c_light_cm_per_s = 2.998e10  # Speed of light in cm/s
    freq_cm = freq_hz / c_light_cm_per_s
    
    # Apply proper normalization for DOS
    # VACF units: (nm/ps)² = (nm²/ps²)
    # Time step dt in ps
    dos_spectral = dos_raw * dt  # Spectral density units: nm²⋅ps⋅cm⁻¹
    
    if return_normalized:
        # Convert to dimensionless DOS (mode density)
        # Normalize by total area to get proper mode counting
        total_area = np.trapz(dos_spectral, freq_cm)
        if total_area > 0:
            # For a system with N atoms, there are 3N-6 vibrational modes
            # (3N-3 for molecules, 3N-6 for polyatomic molecules in solid)
            n_atoms = 5 * 500  # 5 atoms per methane × 500 molecules
            n_modes = 3 * n_atoms - 6  # Total vibrational modes
            dos_normalized = dos_spectral * (n_modes / total_area)
            dos_units = "modes per cm⁻¹"
            dos = dos_normalized
        else:
            dos = dos_spectral
            dos_units = "nm²⋅ps⋅cm⁻¹ (spectral density)"
    else:
        dos = dos_spectral
        dos_units = "nm²⋅ps⋅cm⁻¹ (spectral density)"
    
    # Save DOS data with appropriate units
    with open(f'methanedos_{name}.xvg', 'w') as f:
        f.write('# DOS calculated from VACF\n')
        f.write(f'@    title "Density of States - {name}"\n')
        f.write('@    xaxis  label "Frequency (cm\\S-1\\N)"\n')
        if return_normalized:
            f.write('@    yaxis  label "DOS (modes per cm\\S-1\\N)"\n')
            f.write(f'# Total vibrational modes: {n_modes}\n')
        else:
            f.write('@    yaxis  label "Spectral Density (nm\\S2\\N⋅ps⋅cm\\S-1\\N)"\n')
        f.write('# VACF units: (nm/ps)², Time step: ps\n')
        for i in range(len(freq_cm)):
            f.write(f"{freq_cm[i]:.6f}    {dos[i]:.6e}\n")
    
    return freq_cm, dos, dos_units

# Define all groups
groups = {
    # Total system
    'System': 'Total system',
    # Individual atoms
    'C1': 'Carbon atom',
    'H1': 'Hydrogen 1', 
    'H2': 'Hydrogen 2',
    'H3': 'Hydrogen 3', 
    'H4': 'Hydrogen 4',
    # Bonds
    'CH1': 'C-H1 bond',
    'CH2': 'C-H2 bond', 
    'CH3': 'C-H3 bond',
    'CH4': 'C-H4 bond'
}

# Define filename mapping to match your actual files
filename_mapping = {
    'system': 'vacfsystem.xvg',
    'C1': 'vacf_C1.xvg',
    'H1': 'vacf_H1.xvg',
    'H2': 'vacf_H2.xvg', 
    'H3': 'vacf_H3.xvg',
    'H4': 'vacf_H4.xvg',
    'CH1': 'vacf_CH1.xvg',
    'CH2': 'vacf_CH2.xvg',
    'CH3': 'vacf_CH3.xvg',
    'CH4': 'vacf_CH4.xvg'
}

# Storage for results
all_freq = {}
all_dos = {}
all_units = {}

# Define filename mapping to match your actual files
filename_mapping = {
    'C1': 'vacf_C1.xvg',
    'H1': 'vacf_H1.xvg',
    'H2': 'vacf_H2.xvg', 
    'H3': 'vacf_H3.xvg',
    'H4': 'vacf_H4.xvg',
    'CH1': 'vacf_CH1.xvg',
    'CH2': 'vacf_CH2.xvg',
    'CH3': 'vacf_CH3.xvg',
    'CH4': 'vacf_CH4.xvg'
}

# Process each VACF file
print("Processing VACF files...")
for group, description in groups.items():
    filename = filename_mapping.get(group, f'vacf_{group.lower()}.xvg')
    try:
        time, vacf = read_xvg(filename)
        freq, dos, units = calculate_dos(time, vacf, group.lower(), return_normalized=True)
        all_freq[group] = freq
        all_dos[group] = dos
        all_units[group] = units
        print(f"✓ Processed {filename} - {description}")
    except FileNotFoundError:
        print(f"⚠ Warning: {filename} not found. Skipping {description}.")

# Check if we have any data
if not all_freq:
    print("No VACF files found! Make sure your files are named like 'methanevacf_c1.xvg', etc.")
    exit(1)

print(f"\nSuccessfully processed {len(all_freq)} groups.")

# Create individual DOS plots
print("\nCreating individual DOS plots...")
for group in all_freq.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(all_freq[group], all_dos[group], linewidth=1.5)
    plt.xlim(0, 4000)  
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('DOS (modes per cm⁻¹)')
    plt.title(f'Vibrational Density of States - {groups[group]}')
    plt.grid(True, alpha=0.3)
    
    # Add text box with integration check
    dos_integral = np.trapz(all_dos[group], all_freq[group])
    plt.text(0.02, 0.98, f'∫DOS dω = {dos_integral:.0f} modes', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.savefig(f'methane_dos_{group.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved methane_dos_{group.lower()}.png")

# Create comparison plots
print("\nCreating comparison plots...")

# 1. Compare atoms (C1 vs all H atoms) plus total system
atom_groups = {k: v for k, v in all_freq.items() if k in ['System', 'C1', 'H1', 'H2', 'H3', 'H4']}
if atom_groups:
    plt.figure(figsize=(12, 8))
    
    colors = ['black', 'darkred', 'red', 'blue', 'green', 'orange']
    linestyles = ['-', '-', '--', '--', '--', '--']
    linewidths = [2, 1.5, 1.5, 1.5, 1.5, 1.5]
    
    for i, (group, freq_data) in enumerate(atom_groups.items()):
        if group == 'System':
            # Don't normalize the total system for comparison
            plt.plot(freq_data, all_dos[group], label=groups[group], 
                    color=colors[i % len(colors)], linestyle=linestyles[i], 
                    linewidth=linewidths[i])
        else:
            # Normalize individual components for comparison
            normalized_dos = all_dos[group] / np.max(all_dos[group]) * np.max(all_dos.get('System', all_dos[group]))
            plt.plot(freq_data, normalized_dos, label=groups[group], 
                    color=colors[i % len(colors)], linestyle=linestyles[i], 
                    linewidth=linewidths[i])
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('DOS (modes per cm⁻¹)')
    plt.title('Total System vs Individual Atoms - Vibrational Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('methane_dos_atoms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved methane_dos_atoms_comparison.png")

# 2. Compare bonds (all C-H bonds)
bond_groups = {k: v for k, v in all_freq.items() if k.startswith('CH')}
if bond_groups:
    plt.figure(figsize=(12, 8))
    
    colors = ['purple', 'cyan', 'magenta', 'brown']
    for i, (group, freq_data) in enumerate(bond_groups.items()):
        normalized_dos = all_dos[group] / np.max(all_dos[group])
        plt.plot(freq_data, normalized_dos, label=groups[group], 
                color=colors[i % len(colors)], linewidth=1.5)
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Comparison of C-H Bonds - Vibrational Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('methane_dos_bonds_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved methane_dos_bonds_comparison.png")

# 3. Overall comparison (all groups)
if len(all_freq) > 1:
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: All atoms
    plt.subplot(2, 2, 1)
    for group in ['C1', 'H1', 'H2', 'H3', 'H4']:
        if group in all_freq:
            normalized_dos = all_dos[group] / np.max(all_dos[group])
            plt.plot(all_freq[group], normalized_dos, label=group, linewidth=1.5)
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Individual Atoms')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: All bonds
    plt.subplot(2, 2, 2)
    for group in ['CH1', 'CH2', 'CH3', 'CH4']:
        if group in all_freq:
            normalized_dos = all_dos[group] / np.max(all_dos[group])
            plt.plot(all_freq[group], normalized_dos, label=group, linewidth=1.5)
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('C-H Bonds')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Carbon vs Hydrogens combined
    plt.subplot(2, 2, 3)
    if 'C1' in all_freq:
        normalized_dos = all_dos['C1'] / np.max(all_dos['C1'])
        plt.plot(all_freq['C1'], normalized_dos, label='Carbon (C1)', 
                linewidth=2, color='black')
    
    # Average all hydrogen contributions
    h_groups = [g for g in all_freq.keys() if g.startswith('H')]
    if h_groups:
        avg_h_dos = np.zeros_like(all_dos[h_groups[0]])
        for h_group in h_groups:
            avg_h_dos += all_dos[h_group] / np.max(all_dos[h_group])
        avg_h_dos /= len(h_groups)
        plt.plot(all_freq[h_groups[0]], avg_h_dos, label='Average Hydrogens', 
                linewidth=2, color='red')
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Carbon vs Average Hydrogens')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Peak frequency analysis
    plt.subplot(2, 2, 4)
    peak_frequencies = {}
    peak_intensities = {}
    
    for group, freq_data in all_freq.items():
        dos_data = all_dos[group]
        # Find main peak (exclude low frequencies < 500 cm^-1)
        high_freq_mask = freq_data > 500
        if np.any(high_freq_mask):
            high_freq_dos = dos_data[high_freq_mask]
            high_frequencies = freq_data[high_freq_mask]
            peak_idx = np.argmax(high_freq_dos)
            peak_frequencies[group] = high_frequencies[peak_idx]
            peak_intensities[group] = high_freq_dos[peak_idx]
    
    # Bar plot of peak frequencies
    groups_list = list(peak_frequencies.keys())
    freqs_list = list(peak_frequencies.values())
    colors_bar = ['black' if g == 'C1' else 'red' if g.startswith('H') else 'blue' 
                  for g in groups_list]
    
    bars = plt.bar(range(len(groups_list)), freqs_list, color=colors_bar, alpha=0.7)
    plt.xlabel('Group')
    plt.ylabel('Peak Frequency (cm$^{-1}$)')
    plt.title('Main Peak Frequencies')
    plt.xticks(range(len(groups_list)), groups_list, rotation=45)
    
    # Add value labels on bars
    for bar, freq in zip(bars, freqs_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{freq:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('methane_dos_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved methane_dos_complete_analysis.png")

# Print peak analysis summary
print("\n" + "="*60)
print("PEAK ANALYSIS SUMMARY")
print("="*60)

# Group results by type
atom_peaks = {}
bond_peaks = {}

for group, freq_data in all_freq.items():
    dos_data = all_dos[group]
    # Find main peak (exclude low frequencies < 500 cm^-1)
    high_freq_mask = freq_data > 500
    if np.any(high_freq_mask):
        high_freq_dos = dos_data[high_freq_mask]
        high_frequencies = freq_data[high_freq_mask]
        peak_idx = np.argmax(high_freq_dos)
        peak_freq = high_frequencies[peak_idx]
        
        if group.startswith('CH'):
            bond_peaks[group] = peak_freq
        else:
            atom_peaks[group] = peak_freq

print("\nIndividual Atoms:")
print("-" * 30)
for atom, freq in sorted(atom_peaks.items()):
    print(f"{atom:4s} ({groups[atom]:12s}): {freq:6.0f} cm⁻¹")

print("\nC-H Bonds:")
print("-" * 30)
for bond, freq in sorted(bond_peaks.items()):
    print(f"{bond:4s} ({groups[bond]:12s}): {freq:6.0f} cm⁻¹")

# Check for equivalent hydrogens
if len([k for k in atom_peaks.keys() if k.startswith('H')]) > 1:
    h_freqs = [freq for atom, freq in atom_peaks.items() if atom.startswith('H')]
    if len(set([round(f, -1) for f in h_freqs])) == 1:  # Round to nearest 10
        print(f"\n✓ All hydrogens show equivalent behavior (~{np.mean(h_freqs):.0f} cm⁻¹)")
    else:
        print(f"\n! Hydrogens show different behaviors: {h_freqs}")

# Check for equivalent bonds
if len(bond_peaks) > 1:
    bond_freqs = list(bond_peaks.values())
    if len(set([round(f, -1) for f in bond_freqs])) == 1:  # Round to nearest 10
        print(f"✓ All C-H bonds show equivalent behavior (~{np.mean(bond_freqs):.0f} cm⁻¹)")
    else:
        print(f"! C-H bonds show different behaviors: {bond_freqs}")

print(f"\nAnalysis complete! Generated {len(all_freq)} DOS files and comparison plots.")
print("\nExpected methane vibrational frequencies (for reference):")
print("- C-H symmetric stretch:    ~2917 cm⁻¹") 
print("- C-H asymmetric stretch:   ~3019 cm⁻¹")
print("- H-C-H bending:            ~1534 cm⁻¹")
