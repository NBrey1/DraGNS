#Atom-Resolved Vibrational Density of States (vDoS) for Melamine Crystal
#========================================================================

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# FUNCTION: Read a GROMACS .xvg file
# =============================================================================
# GROMACS outputs data in .xvg format (xmgrace compatible).
# Lines starting with '#' are comments, '@' are Grace formatting commands.
# Data lines have two columns: time (ps) and VACF value.

def read_xvg(filename):
    """
    Parse a GROMACS .xvg file and extract the numerical data.
    
    Parameters:
        filename : str - path to the .xvg file
    
    Returns:
        times : numpy array - time values in picoseconds
        values : numpy array - VACF values (arbitrary units)
    """
    times = []
    values = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip GROMACS header lines (comments and Grace commands)
            if line.startswith('#') or line.startswith('@'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))    # Column 1: time in ps
                values.append(float(parts[1]))   # Column 2: VACF value
    
    return np.array(times), np.array(values)


# =============================================================================
# FUNCTION: Convert VACF to vibrational Density of States
# =============================================================================
# The vDoS is computed by taking the Fourier transform of the VACF.
# A window function (Blackman) is applied first to reduce spectral 
# leakage — artifacts that appear as false broadening of peaks when 
# the VACF hasn't fully decayed to zero at the end of the data window.
#
# The frequency axis is converted from the natural FFT units (1/ps) 
# to wavenumbers (cm⁻¹) using the conversion factor:
#     1 ps⁻¹ = 33.3564 cm⁻¹

def vacf_to_dos(filename):
    """
    Read a VACF .xvg file and compute the vibrational density of states
    via Fourier transform.
    
    Parameters:
        filename : str - path to the VACF .xvg file
    
    Returns:
        freq_cm : numpy array - frequencies in cm⁻¹
        dos : numpy array - density of states (power spectrum)
    """
    # Read the VACF data from the .xvg file
    times, vacf = read_xvg(filename)
    
    # Determine the time step between data points (in picoseconds)
    # This sets the maximum frequency we can resolve (Nyquist limit):
    #   f_max = 1 / (2 * dt)
    dt_ps = times[1] - times[0]
    
    # ---- Apply a Blackman window function ----
    # Without windowing, the sharp cutoff at the end of the VACF data
    # causes "spectral leakage" — artificial broadening and ringing 
    # in the frequency domain. The Blackman window smoothly tapers 
    # the data to zero at both ends, producing cleaner peaks.
    # Other options: np.hanning(), np.hamming(), np.kaiser()
    window = np.blackman(len(vacf))
    vacf_windowed = vacf * window
    
    # ---- Compute the Fourier transform ----
    # np.fft.rfft computes the FFT for real-valued input, returning
    # only the positive-frequency half (since the spectrum is symmetric).
    # The power spectrum |FFT|² gives the density of states.
    n = len(vacf_windowed)
    fft_result = np.fft.rfft(vacf_windowed)
    dos = np.abs(fft_result) ** 2
    
    # ---- Build the frequency axis ----
    # np.fft.rfftfreq returns frequencies in units of 1/dt (i.e., 1/ps).
    # Multiply by 33.3564 to convert from ps⁻¹ to cm⁻¹:
    #   ν(cm⁻¹) = ν(ps⁻¹) × 33.3564
    # This conversion comes from: 1 cm⁻¹ = c/λ, and c = 2.998e10 cm/s
    freq_ps = np.fft.rfftfreq(n, d=dt_ps)   # Frequencies in 1/ps
    freq_cm = freq_ps * 33.3564              # Convert to cm⁻¹
    
    return freq_cm, dos


# =============================================================================
# MAIN: Process all VACF files and generate plots
# =============================================================================

def main():
    # ---- Define the atom groups and their VACF files ----
    # Each entry maps a descriptive label to the .xvg filename
    # produced by gmx velacc for that atom group.
    #
    # Melamine atom classification:
    #   Ring N:  N02, N05, N08 — nitrogen atoms in the triazine ring
    #   Ring C:  C01, C03, C06 — carbon atoms in the triazine ring
    #   Amino N: N00, N04, N07 — nitrogen atoms in the -NH2 groups
    #   Amino H: H09-H0E       — hydrogen atoms in the -NH2 groups
    
    groups = {
        'Ring N (N02, N05, N08)':    'vacf_ring_N.xvg',
        'Ring C (C01, C03, C06)':    'vacf_ring_C.xvg',
        'Amino N (N00, N04, N07)':   'vacf_amine_N.xvg',
        'Amino H (H09–H0E)':        'vacf_amine_H.xvg',
    }
    
    # Colors for each atom group (chemically intuitive)
    colors = {
        'Ring N (N02, N05, N08)':    'blue',
        'Ring C (C01, C03, C06)':    'dimgray',
        'Amino N (N00, N04, N07)':   'green',
        'Amino H (H09–H0E)':        'red',
    }
    
    # Maximum frequency to display (cm⁻¹)
    # N-H stretches appear around 3200-3500 cm⁻¹, so 4000 covers everything
    freq_max = 4000
    
    # Store computed DoS data for all groups
    dos_data = {}
    
    
    # ---- Compute partial DoS for each atom group ----
    print("Computing partial vibrational density of states...\n")
    
    for label, filename in groups.items():
        try:
            freq, dos = vacf_to_dos(filename)
            dos_data[label] = (freq, dos)
            print(f"  {label}: OK ({filename})")
        except FileNotFoundError:
            print(f"  {label}: FILE NOT FOUND ({filename})")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")
    
    if len(dos_data) == 0:
        print("\nNo VACF files found. Make sure the .xvg files are "
              "in the current directory.")
        return
    
    
    # =====================================================================
    # PLOT 1: Individual partial DoS (stacked subplots)
    # =====================================================================
    # This plot shows each atom group's contribution separately,
    # making it easy to identify which atoms dominate each frequency region.
    
    print("\nGenerating individual partial DoS plot...")
    
    fig, axes = plt.subplots(len(dos_data), 1, 
                              figsize=(12, 3 * len(dos_data)),
                              sharex=True)
    
    # Handle case of single subplot (returns Axes, not array)
    if len(dos_data) == 1:
        axes = [axes]
    
    for ax, (label, (freq, dos)) in zip(axes, dos_data.items()):
        # Apply frequency cutoff for display
        mask = freq <= freq_max
        freq_plot = freq[mask]
        dos_plot = dos[mask]
        
        # Normalize each spectrum to its own maximum
        # This allows visual comparison of peak positions
        # even though different atom types have very different
        # absolute intensities (H atoms move much faster than N or C)
        if dos_plot.max() > 0:
            dos_norm = dos_plot / dos_plot.max()
        else:
            dos_norm = dos_plot
        
        # Plot the partial DoS as a filled curve
        color = colors.get(label, 'black')
        ax.plot(freq_plot, dos_norm, color=color, linewidth=0.8)
        ax.fill_between(freq_plot, dos_norm, alpha=0.25, color=color)
        
        # Labels and formatting
        ax.set_ylabel('DoS (norm.)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlim(0, freq_max)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
    
    axes[-1].set_xlabel('Frequency (cm$^{-1}$)', fontsize=12)
    
    plt.suptitle('Atom-Resolved Vibrational Density of States\n'
                 'Melamine Crystal @ 300 K',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('partial_dos_individual.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: partial_dos_individual.png")
    
    
    # =====================================================================
    # PLOT 2: Overlay of all partial DoS on a single axis
    # =====================================================================
    # This overlay makes it easy to compare peak positions between
    # different atom types and identify coupled vibrations (modes where
    # multiple atom types move together).
    
    print("Generating overlay plot...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for label, (freq, dos) in dos_data.items():
        mask = freq <= freq_max
        freq_plot = freq[mask]
        dos_plot = dos[mask]
        
        # Normalize to maximum
        if dos_plot.max() > 0:
            dos_norm = dos_plot / dos_plot.max()
        else:
            dos_norm = dos_plot
        
        color = colors.get(label, 'black')
        ax.plot(freq_plot, dos_norm, color=color, linewidth=1.2,
                label=label, alpha=0.85)
    
    # ---- Annotate the key vibrational regions ----
    # These shaded bands and labels help identify the physical
    # meaning of each spectral region.
    
    # Lattice modes: collective motions of whole molecules
    # (translations, librations) — only present in crystals/solids
    ax.axvspan(0, 250, alpha=0.06, color='gray')
    ax.annotate('Lattice\nmodes', xy=(125, 0.92), fontsize=9,
                ha='center', style='italic', color='gray')
    
    # Ring deformation modes: bending and twisting of the triazine ring
    ax.annotate('Ring\ndeformation', xy=(700, 0.92), fontsize=9,
                ha='center', style='italic', color='dimgray')
    
    # C-N stretching / ring breathing modes
    ax.annotate('C–N stretch\nring breathing', xy=(1100, 0.92), 
                fontsize=9, ha='center', style='italic', color='dimgray')
    
    # NH₂ scissoring + ring C=N stretching (typically strongest peak)
    ax.annotate('NH₂ scissor\n+ C=N stretch', xy=(1600, 0.92), 
                fontsize=9, ha='center', style='italic', color='green')
    
    # N-H stretching region: sensitive to hydrogen bonding
    ax.axvspan(3100, 3600, alpha=0.06, color='red')
    ax.annotate('N–H\nstretch', xy=(3350, 0.92), fontsize=9,
                ha='center', style='italic', color='red')
    
    # Formatting
    ax.set_xlabel('Frequency (cm$^{-1}$)', fontsize=13)
    ax.set_ylabel('Partial DoS (normalized)', fontsize=13)
    ax.set_title('Atom-Resolved Vibrational Density of States — '
                 'Melamine Crystal @ 300 K',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', 
              framealpha=0.9, edgecolor='gray')
    ax.set_xlim(0, freq_max)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig('partial_dos_overlay.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: partial_dos_overlay.png")
    
    
    # =====================================================================
    # PLOT 3: Focus on the N-H stretching region (3000-3700 cm⁻¹)
    # =====================================================================
    # This zoomed view reveals the hydrogen bonding signature.
    # In crystalline melamine, H-bonding shifts and splits the N-H 
    # stretches:
    #   - Lower frequency (~3100-3300 cm⁻¹): H-bonded N-H stretches
    #   - Higher frequency (~3400-3500 cm⁻¹): free/weakly bonded N-H
    # The splitting pattern provides information about H-bond strength
    # and geometry in the crystal.
    
    if 'Amino H (H09–H0E)' in dos_data:
        print("Generating N-H stretch zoom plot...")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot amino H (dominant in this region)
        freq, dos = dos_data['Amino H (H09–H0E)']
        mask = (freq >= 2800) & (freq <= 3800)
        freq_plot = freq[mask]
        dos_plot = dos[mask]
        
        if dos_plot.max() > 0:
            dos_norm = dos_plot / dos_plot.max()
        else:
            dos_norm = dos_plot
        
        ax.plot(freq_plot, dos_norm, color='red', linewidth=1.5)
        ax.fill_between(freq_plot, dos_norm, alpha=0.2, color='red',
                        label='Amino H')
        
        # Also plot amino N contribution (shows coupling)
        if 'Amino N (N00, N04, N07)' in dos_data:
            freq_n, dos_n = dos_data['Amino N (N00, N04, N07)']
            mask_n = (freq_n >= 2800) & (freq_n <= 3800)
            dos_n_plot = dos_n[mask_n]
            if dos_n_plot.max() > 0:
                dos_n_norm = dos_n_plot / dos_n_plot.max()
                ax.plot(freq_n[mask_n], dos_n_norm, color='green',
                        linewidth=1.5, linestyle='--',
                        label='Amino N (scaled)')
        
        ax.set_xlabel('Frequency (cm$^{-1}$)', fontsize=13)
        ax.set_ylabel('Partial DoS (normalized)', fontsize=13)
        ax.set_title('N–H Stretching Region — Hydrogen Bonding '
                     'Signature', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_xlim(2800, 3800)
        ax.grid(True, alpha=0.15)
        
        # Annotate H-bonded vs free N-H
        ax.annotate('H-bonded\nN–H stretch', 
                    xy=(3200, 0.7), fontsize=10,
                    ha='center', style='italic', color='darkred')
        ax.annotate('Free/weak\nN–H stretch', 
                    xy=(3500, 0.7), fontsize=10,
                    ha='center', style='italic', color='darkred')
        
        plt.tight_layout()
        plt.savefig('nh_stretch_zoom.png', dpi=150, 
                    bbox_inches='tight')
        plt.show()
        print("  Saved: nh_stretch_zoom.png")
    
    
    # =====================================================================
    # Print summary of peak positions for each atom group
    # =====================================================================
    # Identify the frequency of the top peaks in each partial DoS.
    # This helps with quantitative comparison to experimental data.
    
    print("\n" + "=" * 65)
    print("PEAK POSITIONS (top 5 peaks per atom group)")
    print("=" * 65)
    
    for label, (freq, dos) in dos_data.items():
        mask = (freq > 50) & (freq <= freq_max)
        freq_masked = freq[mask]
        dos_masked = dos[mask]
        
        if dos_masked.max() == 0:
            continue
        
        # Find peaks using a simple local maximum detection
        # A point is a peak if it's larger than both neighbors
        peaks = []
        for i in range(1, len(dos_masked) - 1):
            if (dos_masked[i] > dos_masked[i-1] and 
                dos_masked[i] > dos_masked[i+1] and
                dos_masked[i] > 0.05 * dos_masked.max()):
                peaks.append((freq_masked[i], dos_masked[i]))
        
        # Sort by intensity (strongest first) and take top 5
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:5]
        
        print(f"\n{label}:")
        for j, (f, intensity) in enumerate(top_peaks):
            rel_intensity = intensity / dos_masked.max() * 100
            print(f"  Peak {j+1}: {f:7.1f} cm⁻¹  "
                  f"(relative intensity: {rel_intensity:5.1f}%)")
    
    print("\n" + "=" * 65)
    print("Done. All plots saved to current directory.")
    print("=" * 65)


# =============================================================================
# Run the script
# =============================================================================

if __name__ == "__main__":
    main()

