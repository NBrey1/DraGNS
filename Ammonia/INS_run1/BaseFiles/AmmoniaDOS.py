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
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(x), np.array(y)

# Calculate DOS from VACF
def calculate_dos(time, vacf, name, return_normalized=True):
    # Validate input data
    if len(time) < 2:
        raise ValueError("Time array must have at least 2 points")
    if len(time) != len(vacf):
        raise ValueError("Time and VACF arrays must have the same length")
    
    # Handle GROMACS XVG files that might start with duplicate time points
    # Find the first valid time step
    dt = None
    for i in range(1, min(10, len(time))):  # Check first 10 points
        potential_dt = time[i] - time[0]
        if potential_dt > 1e-12:  # Found a non-zero time step
            dt = potential_dt / i  # Average time step
            break
    
    if dt is None or dt <= 0:
        # If we still can't find a valid dt, try using the total time range
        total_time = time[-1] - time[0]
        if total_time > 0:
            dt = total_time / (len(time) - 1)
            print(f"Warning: Using average time step for {name}: dt = {dt:.6f} ps")
        else:
            raise ValueError(f"Cannot determine valid time step for {name}. All time values appear to be the same.")
    
    # Check for uniform time spacing (allowing for small variations)
    time_diffs = np.diff(time)
    non_zero_diffs = time_diffs[time_diffs > 1e-12]
    if len(non_zero_diffs) > 0:
        avg_dt = np.mean(non_zero_diffs)
        if not np.allclose(non_zero_diffs, avg_dt, rtol=1e-3):
            print(f"Warning: Non-uniform time spacing detected for {name}. Using average dt = {avg_dt:.6f} ps")
            dt = avg_dt
    
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
    dos_spectral = dos_raw * dt  # Spectral density units: nm²·ps·cm⁻¹
    
    # Calculate number of vibrational modes
    n_atoms = 4 * 500  # 4 atoms per NH3 × 500 molecules
    n_modes = 3 * n_atoms - 6  # Total vibrational modes
    
    if return_normalized:
        # Convert to dimensionless DOS (mode density)
        # Normalize by total area to get proper mode counting
        total_area = np.trapezoid(dos_spectral, freq_cm)
        if total_area > 0:
            dos_normalized = dos_spectral * (n_modes / total_area)
            dos_units = "modes per cm⁻¹"
            dos = dos_normalized
        else:
            print(f"Warning: Zero total area for {name}. Using spectral density.")
            dos = dos_spectral
            dos_units = "nm²·ps·cm⁻¹ (spectral density)"
    else:
        dos = dos_spectral
        dos_units = "nm²·ps·cm⁻¹ (spectral density)"
    
    # Save DOS data with appropriate units
    with open(f'ammonados_{name}.xvg', 'w') as f:
        f.write('# DOS calculated from VACF\n')
        f.write(f'@    title "Density of States - {name}"\n')
        f.write('@    xaxis  label "Frequency (cm\\S-1\\N)"\n')
        if return_normalized and 'modes per' in dos_units:
            f.write('@    yaxis  label "DOS (modes per cm\\S-1\\N)"\n')
            f.write(f'# Total vibrational modes: {n_modes}\n')
        else:
            f.write('@    yaxis  label "Spectral Density (nm\\S2\\NÂ·psÂ·cm\\S-1\\N)"\n')
        f.write('# VACF units: (nm/ps)², Time step: ps\n')
        f.write(f'# Time step dt: {dt:.6f} ps\n')
        f.write(f'# Number of data points: {len(time)}\n')
        for i in range(len(freq_cm)):
            f.write(f"{freq_cm[i]:.6f}    {dos[i]:.6e}\n")
    
    return freq_cm, dos, dos_units

# Define all groups
groups = {
    # Total system
    'System': 'Total system',
    # Individual atoms
    'N': 'Nitrogen atom',
    'H1': 'Hydrogen 1', 
    'H2': 'Hydrogen 2',
    'H3': 'Hydrogen 3',
    # Bonds
    'NH1': 'N-H1 bond',
    'NH2': 'N-H2 bond', 
    'NH3': 'N-H3 bond'
}

# Define filename mapping to match your actual files
filename_mapping = {
    'System': 'vacf_system.xvg',
    'N': 'vacf_N.xvg',
    'H1': 'vacf_H1.xvg',
    'H2': 'vacf_H2.xvg', 
    'H3': 'vacf_H3.xvg',
    'NH1': 'vacf_NH1.xvg',
    'NH2': 'vacf_NH2.xvg',
    'NH3': 'vacf_NH3.xvg'
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
        
        # Validate data before processing
        if len(time) == 0 or len(vacf) == 0:
            print(f"⚠  Warning: {filename} is empty or contains no valid data. Skipping {description}.")
            continue
        
        print(f"   {filename}: {len(time)} data points, time range: {time[0]:.3f} to {time[-1]:.3f} ps")
        
        freq, dos, units = calculate_dos(time, vacf, group.lower(), return_normalized=True)
        all_freq[group] = freq
        all_dos[group] = dos
        all_units[group] = units
        print(f"✓ Processed {filename} - {description}")
        
    except FileNotFoundError:
        print(f"⚠  Warning: {filename} not found. Skipping {description}.")
    except Exception as e:
        print(f"✗ Error processing {filename}: {str(e)}")

# Check if we have any data
if not all_freq:
    print("No VACF files found or processed successfully!")
    print("Make sure your files are named correctly and contain valid data.")
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
    dos_integral = np.trapezoid(all_dos[group], all_freq[group])
    plt.text(0.02, 0.98, f'∫DOS dω = {dos_integral:.0f} modes', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.savefig(f'ammonia_dos_{group.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ammonia_dos_{group.lower()}.png")

# Create comparison plots
print("\nCreating comparison plots...")

# 1. Compare atoms (N vs all H atoms) plus total system
atom_groups = {k: v for k, v in all_freq.items() if k in ['System', 'N', 'H1', 'H2', 'H3']}
if atom_groups:
    plt.figure(figsize=(12, 8))
    
    colors = ['black', 'darkblue', 'red', 'green', 'orange']
    linestyles = ['-', '-', '--', '--', '--']
    linewidths = [2, 1.5, 1.5, 1.5, 1.5]
    
    for i, (group, freq_data) in enumerate(atom_groups.items()):
        if group == 'System':
            # Don't normalize the total system for comparison
            plt.plot(freq_data, all_dos[group], label=groups[group], 
                    color=colors[i % len(colors)], linestyle=linestyles[i], 
                    linewidth=linewidths[i])
        else:
            # Normalize individual components for comparison
            max_dos = np.max(all_dos[group])
            if max_dos > 0:
                normalized_dos = all_dos[group] / max_dos * np.max(all_dos.get('System', all_dos[group]))
                plt.plot(freq_data, normalized_dos, label=groups[group], 
                        color=colors[i % len(colors)], linestyle=linestyles[i], 
                        linewidth=linewidths[i])
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('DOS (modes per cm⁻¹)')
    plt.title('Total System vs Individual Atoms - Vibrational Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ammonia_dos_atoms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved ammonia_dos_atoms_comparison.png")

# 2. Compare bonds (all N-H bonds)
bond_groups = {k: v for k, v in all_freq.items() if k.startswith('NH')}
if bond_groups:
    plt.figure(figsize=(12, 8))
    
    colors = ['purple', 'cyan', 'magenta']
    for i, (group, freq_data) in enumerate(bond_groups.items()):
        max_dos = np.max(all_dos[group])
        if max_dos > 0:
            normalized_dos = all_dos[group] / max_dos
            plt.plot(freq_data, normalized_dos, label=groups[group], 
                    color=colors[i % len(colors)], linewidth=1.5)
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Comparison of N-H Bonds - Vibrational Density of States')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ammonia_dos_bonds_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved ammonia_dos_bonds_comparison.png")

# 3. Overall comparison (all groups)
if len(all_freq) > 1:
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: All atoms
    plt.subplot(2, 2, 1)
    for group in ['N', 'H1', 'H2', 'H3']:
        if group in all_freq:
            max_dos = np.max(all_dos[group])
            if max_dos > 0:
                normalized_dos = all_dos[group] / max_dos
                plt.plot(all_freq[group], normalized_dos, label=group, linewidth=1.5)
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Individual Atoms')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: All bonds
    plt.subplot(2, 2, 2)
    for group in ['NH1', 'NH2', 'NH3']:
        if group in all_freq:
            max_dos = np.max(all_dos[group])
            if max_dos > 0:
                normalized_dos = all_dos[group] / max_dos
                plt.plot(all_freq[group], normalized_dos, label=group, linewidth=1.5)
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('N-H Bonds')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Nitrogen vs Hydrogens combined
    plt.subplot(2, 2, 3)
    if 'N' in all_freq:
        max_dos = np.max(all_dos['N'])
        if max_dos > 0:
            normalized_dos = all_dos['N'] / max_dos
            plt.plot(all_freq['N'], normalized_dos, label='Nitrogen (N)', 
                    linewidth=2, color='darkblue')
    
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
    plt.title('Nitrogen vs Average Hydrogens')
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
            if len(high_freq_dos) > 0 and np.max(high_freq_dos) > 0:
                peak_idx = np.argmax(high_freq_dos)
                peak_frequencies[group] = high_frequencies[peak_idx]
                peak_intensities[group] = high_freq_dos[peak_idx]
    
    # Bar plot of peak frequencies
    if peak_frequencies:
        groups_list = list(peak_frequencies.keys())
        freqs_list = list(peak_frequencies.values())
        colors_bar = ['darkblue' if g == 'N' else 'red' if g.startswith('H') else 'purple' 
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
    plt.savefig('ammonia_dos_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved ammonia_dos_complete_analysis.png")

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
        if len(high_freq_dos) > 0 and np.max(high_freq_dos) > 0:
            peak_idx = np.argmax(high_freq_dos)
            peak_freq = high_frequencies[peak_idx]
            
            if group.startswith('NH'):
                bond_peaks[group] = peak_freq
            elif group != 'System':  # Exclude system from atom analysis
                atom_peaks[group] = peak_freq

print("\nIndividual Atoms:")
print("-" * 30)
for atom, freq in sorted(atom_peaks.items()):
    if atom in groups:
        print(f"{atom:4s} ({groups[atom]:12s}): {freq:6.0f} cm⁻¹")

print("\nN-H Bonds:")
print("-" * 30)
for bond, freq in sorted(bond_peaks.items()):
    print(f"{bond:4s} ({groups[bond]:12s}): {freq:6.0f} cm⁻¹")

# Check for equivalent hydrogens
h_atoms = [k for k in atom_peaks.keys() if k.startswith('H')]
if len(h_atoms) > 1:
    h_freqs = [freq for atom, freq in atom_peaks.items() if atom.startswith('H')]
    if len(set([round(f, -1) for f in h_freqs])) == 1:  # Round to nearest 10
        print(f"\n✓ All hydrogens show equivalent behavior (~{np.mean(h_freqs):.0f} cm⁻¹)")
    else:
        print(f"\n! Hydrogens show different behaviors: {h_freqs}")

# Check for equivalent bonds
if len(bond_peaks) > 1:
    bond_freqs = list(bond_peaks.values())
    if len(set([round(f, -1) for f in bond_freqs])) == 1:  # Round to nearest 10
        print(f"✓ All N-H bonds show equivalent behavior (~{np.mean(bond_freqs):.0f} cm⁻¹)")
    else:
        print(f"! N-H bonds show different behaviors: {bond_freqs}")

print(f"\nAnalysis complete! Generated {len(all_freq)} DOS files and comparison plots.")
