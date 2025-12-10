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
    # Check for valid data
    if len(time) == 0 or len(vacf) == 0:
        raise ValueError(f"Empty data in VACF file for {name}")
    
    if np.all(np.isnan(vacf)) or np.all(vacf == 0):
        raise ValueError(f"Invalid VACF data (all zeros or NaN) for {name}")
    
    # Get time step - handle potential issues
    if len(time) < 2:
        raise ValueError(f"Insufficient time points for {name}")
        
    # Check if all time values are the same (problematic XVG file)
    if np.all(time == time[0]):
        print(f"Warning: All time values are identical ({time[0]}) for {name}")
        print("This suggests the XVG file may have formatting issues.")
        print("Attempting to use index-based time step...")
        # Use default time step for GROMACS (often 1 fs = 0.001 ps)
        dt = 0.001  # ps
        print(f"Using default dt = {dt} ps")
    else:
        dt = time[1] - time[0]  # in ps
        
        # Check for valid time step
        if dt <= 0 or np.isnan(dt):
            # Try using the time range divided by number of points
            time_range = time[-1] - time[0]
            if time_range > 0:
                dt = time_range / (len(time) - 1)
                print(f"Warning: Invalid dt from time[1]-time[0], using average: {dt} ps")
            else:
                raise ValueError(f"Invalid time step: {dt} ps for {name}")
    
    print(f"  Using time step: {dt} ps for {name}")
    
    # Window function to reduce noise
    window = np.blackman(len(vacf))
    vacf_windowed = vacf * window
    
    # FFT to get power spectrum (DOS)
    n = len(vacf)
    dos_raw = np.abs(rfft(vacf_windowed))
    
    # Handle potential issues with frequency calculation
    try:
        freq_hz = np.fft.rfftfreq(n, dt * 1e-12)  # Convert ps to s for frequency in Hz
    except Exception as e:
        raise ValueError(f"Error calculating frequencies for {name}: {e}")
    
    # Convert frequency from Hz to cm^-1
    c_light_cm_per_s = 2.998e10  # Speed of light in cm/s
    freq_cm = freq_hz / c_light_cm_per_s
    
    # Apply proper normalization for DOS
    # VACF units: (nm/ps)² = (nm²/ps²)
    # Time step dt in ps
    dos_spectral = dos_raw * dt  # Spectral density units: nm²⋅ps⋅cm⁻¹
    
    # Initialize n_modes outside the if block
    n_atoms = 5 * 500  # 5 atoms per methane × 500 molecules
    n_modes = 3 * n_atoms - 6  # Total vibrational modes
    
    if return_normalized:
        # Convert to dimensionless DOS (mode density)
        # Normalize by total area to get proper mode counting
        total_area = np.trapezoid(dos_spectral, freq_cm)  # Updated from trapz
        if total_area > 0 and not np.isnan(total_area):
            # For a system with N atoms, there are 3N-6 vibrational modes
            # (3N-3 for molecules, 3N-6 for polyatomic molecules in solid)
            dos_normalized = dos_spectral * (n_modes / total_area)
            dos_units = "modes per cm⁻¹"
            dos = dos_normalized
        else:
            print(f"Warning: Invalid total area ({total_area}) for {name}, using spectral density")
            dos = dos_spectral
            dos_units = "nm²⋅ps⋅cm⁻¹ (spectral density)"
    else:
        dos = dos_spectral
        dos_units = "nm²⋅ps⋅cm⁻¹ (spectral density)"
    
    # Check for NaN or inf values in results
    if np.any(np.isnan(dos)) or np.any(np.isinf(dos)):
        print(f"Warning: Invalid values detected in DOS for {name}")
        # Replace NaN/inf with zeros
        dos = np.where(np.isfinite(dos), dos, 0.0)
    
    # Save DOS data with appropriate units
    with open(f'methanedos_{name}.xvg', 'w') as f:
        f.write('# DOS calculated from VACF\n')
        f.write(f'@    title "Density of States - {name}"\n')
        f.write('@    xaxis  label "Frequency (cm\\S-1\\N)"\n')
        if return_normalized and total_area > 0 and not np.isnan(total_area):
            f.write('@    yaxis  label "DOS (modes per cm\\S-1\\N)"\n')
            f.write(f'# Total vibrational modes: {n_modes}\n')
        else:
            f.write('@    yaxis  label "Spectral Density (nm\\S2\\Nâ‹…psâ‹…cm\\S-1\\N)"\n')
        f.write('# VACF units: (nm/ps)², Time step: ps\n')
        f.write(f'# Actual time step used: {dt} ps\n')
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

# Define filename mapping to match your actual files - SINGLE MAPPING ONLY
filename_mapping = {
    'System': 'vacf_system.xvg',  # Fixed: System now maps correctly
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

# Process each VACF file
print("Processing VACF files...")
for group, description in groups.items():
    filename = filename_mapping.get(group, f'vacf_{group.lower()}.xvg')
    try:
        time, vacf = read_xvg(filename)
        print(f"  Read {filename}: {len(time)} points, VACF range: [{np.min(vacf):.2e}, {np.max(vacf):.2e}]")
        
        freq, dos, units = calculate_dos(time, vacf, group.lower(), return_normalized=True)
        all_freq[group] = freq
        all_dos[group] = dos
        all_units[group] = units
        print(f"✓ Processed {filename} - {description}")
        
    except FileNotFoundError:
        print(f"⚠  Warning: {filename} not found. Skipping {description}.")
    except ValueError as e:
        print(f"✗ Error processing {filename}: {e}")
    except Exception as e:
        print(f"✗ Unexpected error processing {filename}: {e}")

# Check if we have any data
if not all_freq:
    print("No VACF files found or processed successfully!")
    print("Make sure your files are named like 'vacf_C1.xvg', 'vacfsystem.xvg', etc.")
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
    dos_integral = np.trapezoid(all_dos[group], all_freq[group])  # Updated from trapz
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
            # Don't normalize the total system for comparison, but scale for visibility
            system_scale = 0.1  # Scale factor to make system visible
            scaled_dos = all_dos[group] * system_scale
            plt.plot(freq_data, scaled_dos, label=f'{groups[group]} (×{system_scale})', 
                    color=colors[i % len(colors)], linestyle=linestyles[i], 
                    linewidth=linewidths[i])
        else:
            # Normalize individual components for comparison
            max_dos = np.max(all_dos[group])
            if max_dos > 0:
                if 'System' in all_dos:
                    # Scale to match system scale
                    normalized_dos = all_dos[group] / max_dos * np.max(all_dos['System']) * 0.1
                else:
                    normalized_dos = all_dos[group] / max_dos
                plt.plot(freq_data, normalized_dos, label=groups[group], 
                        color=colors[i % len(colors)], linestyle=linestyles[i], 
                        linewidth=linewidths[i])
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('DOS (scaled for comparison)')
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
        max_dos = np.max(all_dos[group])
        if max_dos > 0:
            normalized_dos = all_dos[group] / max_dos
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
    
    # Subplot 1: All atoms including system
    plt.subplot(2, 2, 1)
    for group in ['System', 'C1', 'H1', 'H2', 'H3', 'H4']:
        if group in all_freq:
            if group == 'System':
                # Scale system for visibility
                system_scale = 0.05
                scaled_dos = all_dos[group] * system_scale
                plt.plot(all_freq[group], scaled_dos, label=f'System (×{system_scale})', 
                        linewidth=2, color='black')
            else:
                max_dos = np.max(all_dos[group])
                if max_dos > 0:
                    normalized_dos = all_dos[group] / max_dos
                    plt.plot(all_freq[group], normalized_dos, label=group, linewidth=1.5)
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized/Scaled DOS')
    plt.title('System vs Individual Atoms')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: All bonds
    plt.subplot(2, 2, 2)
    for group in ['CH1', 'CH2', 'CH3', 'CH4']:
        if group in all_freq:
            max_dos = np.max(all_dos[group])
            if max_dos > 0:
                normalized_dos = all_dos[group] / max_dos
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
        max_dos = np.max(all_dos['C1'])
        if max_dos > 0:
            normalized_dos = all_dos['C1'] / max_dos
            plt.plot(all_freq['C1'], normalized_dos, label='Carbon (C1)', 
                    linewidth=2, color='black')
    
    # Average all hydrogen contributions
    h_groups = [g for g in all_freq.keys() if g.startswith('H')]
    if h_groups:
        avg_h_dos = np.zeros_like(all_dos[h_groups[0]])
        valid_h_count = 0
        for h_group in h_groups:
            max_dos = np.max(all_dos[h_group])
            if max_dos > 0:
                avg_h_dos += all_dos[h_group] / max_dos
                valid_h_count += 1
        if valid_h_count > 0:
            avg_h_dos /= valid_h_count
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
        if np.any(high_freq_mask) and np.max(dos_data) > 0:
            high_freq_dos = dos_data[high_freq_mask]
            high_frequencies = freq_data[high_freq_mask]
            if len(high_freq_dos) > 0:
                peak_idx = np.argmax(high_freq_dos)
                peak_frequencies[group] = high_frequencies[peak_idx]
                peak_intensities[group] = high_freq_dos[peak_idx]
    
    # Bar plot of peak frequencies
    if peak_frequencies:
        groups_list = list(peak_frequencies.keys())
        freqs_list = list(peak_frequencies.values())
        colors_bar = ['gray' if g == 'System' else 'black' if g == 'C1' else 'red' if g.startswith('H') else 'blue' 
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
    if np.any(high_freq_mask) and np.max(dos_data) > 0:
        high_freq_dos = dos_data[high_freq_mask]
        high_frequencies = freq_data[high_freq_mask]
        if len(high_freq_dos) > 0:
            peak_idx = np.argmax(high_freq_dos)
            peak_freq = high_frequencies[peak_idx]
            
            if group.startswith('CH'):
                bond_peaks[group] = peak_freq
            else:
                atom_peaks[group] = peak_freq

if atom_peaks:
    print("\nTotal System + Individual Atoms:")
    print("-" * 40)
    for atom, freq in sorted(atom_peaks.items()):
        print(f"{atom:7s} ({groups[atom]:15s}): {freq:6.0f} cm⁻¹")

if bond_peaks:
    print("\nC-H Bonds:")
    print("-" * 30)
    for bond, freq in sorted(bond_peaks.items()):
        print(f"{bond:4s} ({groups[bond]:12s}): {freq:6.0f} cm⁻¹")

# Check for equivalent hydrogens
h_atom_peaks = {k: v for k, v in atom_peaks.items() if k.startswith('H')}
if len(h_atom_peaks) > 1:
    h_freqs = list(h_atom_peaks.values())
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

