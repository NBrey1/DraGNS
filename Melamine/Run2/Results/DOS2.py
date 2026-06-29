#!/usr/bin/env python3
"""
Diagnostic version: VACF → DoS with detailed checks
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# =============================================================
# STEP 1: Read a GROMACS .xvg file
# =============================================================
def read_xvg(filename):
    """
    Read a GROMACS .xvg file.
    Handles the case where time values are rounded to 3 decimal
    places, causing apparent duplicate time entries.
    """
    times = []
    values = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                values.append(float(parts[1]))
    
    times = np.array(times)
    values = np.array(values)
    
    # Compute the CORRECT time step by averaging over all points
    # This avoids the rounding problem in GROMACS .xvg output
    dt = (times[-1] - times[0]) / (len(times) - 1)
    
    # Rebuild the time axis with correct spacing
    times_corrected = np.arange(len(times)) * dt
    
    print(f"  {filename}: {len(times)} points, "
          f"dt = {dt:.6f} ps, "
          f"total = {times_corrected[-1]:.3f} ps, "
          f"Nyquist = {1.0/(2*dt)*33.356:.0f} cm⁻¹")
    
    return times_corrected, values


def vacf_to_dos(times, vacf):
    """
    Convert VACF to vibrational density of states via FFT.
    """
    dt = times[1] - times[0]  # Now correct from read_xvg
    
    # Normalize VACF
    if vacf[0] != 0:
        vacf_norm = vacf / vacf[0]
    else:
        vacf_norm = vacf.copy()
    
    # Apply Blackman window
    window = np.blackman(len(vacf_norm))
    vacf_windowed = vacf_norm * window
    
    # FFT → power spectrum
    n = len(vacf_windowed)
    fft_result = np.fft.rfft(vacf_windowed)
    dos = np.abs(fft_result)
    
    # Frequency axis: 1/ps → cm⁻¹ (multiply by 33.356)
    freq_ps = np.fft.rfftfreq(n, d=dt)
    freq_cm = freq_ps * 33.356
    
    return freq_cm, dos


# =============================================================
# STEP 2: Diagnostic — inspect the raw VACF data
# =============================================================
def diagnose_vacf(filename):
    """
    Print diagnostic information about a VACF file.
    """
    print(f"\n{'='*50}")
    print(f"Diagnosing: {filename}")
    print(f"{'='*50}")
    
    if not os.path.exists(filename):
        print(f"  FILE NOT FOUND!")
        return None, None
    
    times, vacf = read_xvg(filename)
    
    print(f"  Number of data points: {len(times)}")
    print(f"  Time range: {times[0]:.4f} to {times[-1]:.4f} ps")
    print(f"  Time step (dt): {times[1]-times[0]:.6f} ps")
    print(f"  VACF at t=0: {vacf[0]:.6e}")
    print(f"  VACF at t=end: {vacf[-1]:.6e}")
    print(f"  VACF min: {vacf.min():.6e}")
    print(f"  VACF max: {vacf.max():.6e}")
    print(f"  VACF mean: {vacf.mean():.6e}")
    
    # Check if VACF is normalized (starts at 1.0)
    if abs(vacf[0] - 1.0) < 0.01:
        print(f"  VACF appears NORMALIZED (starts at ~1.0)")
    else:
        print(f"  VACF appears UNNORMALIZED (starts at {vacf[0]:.4e})")
    
    # Check if VACF decays
    ratio = abs(vacf[-1] / vacf[0]) if vacf[0] != 0 else 0
    print(f"  Decay ratio (end/start): {ratio:.4f}")
    if ratio > 0.5:
        print(f"  WARNING: VACF has not decayed much!")
        print(f"  Consider using a longer correlation time.")
    
    # Maximum resolvable frequency
    dt = (times[-1] - times[0]) / (len(times) - 1)
    f_nyquist = 1.0 / (2.0 * dt)  # in 1/ps
    f_nyquist_cm = f_nyquist * 33.356  # in cm^-1
    print(f"  Nyquist frequency: {f_nyquist_cm:.0f} cm^-1")
    
    # Frequency resolution
    total_time = times[-1] - times[0]
    f_resolution = 1.0 / total_time  # in 1/ps
    f_resolution_cm = f_resolution * 33.356
    print(f"  Frequency resolution: {f_resolution_cm:.2f} cm^-1")
    
    return times, vacf


# =============================================================
# STEP 3: Convert VACF to DoS via Fourier Transform
# =============================================================
def vacf_to_dos(times, vacf):
    """
    Compute the vibrational density of states from the VACF.
    
    The DoS g(w) is the Fourier transform of the VACF:
        g(w) = integral of <v(0).v(t)> * exp(-iwt) dt
    
    Steps:
    1. Apply Blackman window to reduce spectral leakage
    2. Compute real FFT (since VACF is real-valued)
    3. Take absolute value squared = power spectrum = DoS
    4. Convert frequency from 1/ps to cm^-1
    """
    dt = times[1] - times[0]  # time step in ps
    
    # Normalize VACF to start at 1.0
    # This ensures all atom groups are on comparable scales
    if vacf[0] != 0:
        vacf_norm = vacf / vacf[0]
    else:
        vacf_norm = vacf.copy()
    
    # Apply Blackman window
    # This smoothly tapers the data to zero at both ends,
    # preventing artifacts from the sharp cutoff
    window = np.blackman(len(vacf_norm))
    vacf_windowed = vacf_norm * window
    
    # Compute FFT
    n = len(vacf_windowed)
    fft_result = np.fft.rfft(vacf_windowed)
    
    # Power spectrum = |FFT|^2
    # Using just the real part of FFT can also work:
    #   dos = np.real(fft_result)
    # The power spectrum shows all modes regardless of phase
    dos = np.abs(fft_result)
    
    # Frequency axis
    # rfftfreq returns frequencies in units of 1/dt = 1/ps
    # Multiply by 33.356 to convert to cm^-1
    freq_ps = np.fft.rfftfreq(n, d=dt)
    freq_cm = freq_ps * 33.356
    
    return freq_cm, dos


# =============================================================
# MAIN
# =============================================================
def main():
    # Define the VACF files and their labels
    groups = {
        'Ring N (N02, N05, N08)':    'vacf_ring_N.xvg',
        'Ring C (C01, C03, C06)':    'vacf_ring_C.xvg',
        'Amino N (N00, N04, N07)':   'vacf_amine_N.xvg',
        'Amino H (H09-H0E)':        'vacf_amine_H.xvg',
    }
    
    colors = {
        'Ring N (N02, N05, N08)':    'blue',
        'Ring C (C01, C03, C06)':    'gray',
        'Amino N (N00, N04, N07)':   'green',
        'Amino H (H09-H0E)':        'red',
    }
    
    # ---- STEP A: Diagnose all VACF files ----
    print("=" * 60)
    print("VACF DIAGNOSTIC REPORT")
    print("=" * 60)
    
    vacf_data = {}
    for label, filename in groups.items():
        times, vacf = diagnose_vacf(filename)
        if times is not None:
            vacf_data[label] = (times, vacf)
    
    if len(vacf_data) == 0:
        print("\nNo VACF files found! Check filenames.")
        print("Expected files:")
        for label, filename in groups.items():
            print(f"  {filename}")
        sys.exit(1)
    
    # ---- STEP B: Plot raw VACFs ----
    print("\n\nPlotting raw VACFs...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (label, (times, vacf)) in enumerate(vacf_data.items()):
        ax = axes[i]
        
        # Normalize for display
        vacf_norm = vacf / vacf[0] if vacf[0] != 0 else vacf
        
        # Show first 2 ps (where most decay happens)
        mask = times < 2.0
        if mask.sum() > 0:
            ax.plot(times[mask], vacf_norm[mask], 
                    color=colors[label], linewidth=0.8)
        else:
            ax.plot(times, vacf_norm, 
                    color=colors[label], linewidth=0.8)
        
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Normalized VACF')
        ax.axhline(y=0, color='black', linewidth=0.5, 
                   linestyle='--')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Raw Velocity Autocorrelation Functions',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('diagnostic_vacf_raw.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: diagnostic_vacf_raw.png")
    
    # ---- STEP C: Compute and plot partial DoS ----
    print("\nComputing partial DoS...")
    
    dos_data = {}
    for label, (times, vacf) in vacf_data.items():
        freq, dos = vacf_to_dos(times, vacf)
        dos_data[label] = (freq, dos)
        print(f"  {label}: freq range 0 - {freq[-1]:.0f} cm^-1, "
              f"{len(freq)} points")
    
    # ---- Plot 1: Individual panels ----
    fig, axes = plt.subplots(len(dos_data), 1, 
                              figsize=(14, 3.5 * len(dos_data)),
                              sharex=True)
    if len(dos_data) == 1:
        axes = [axes]
    
    freq_max = 4000  # cm^-1
    
    for ax, (label, (freq, dos)) in zip(axes, dos_data.items()):
        mask = freq <= freq_max
        
        # Normalize each to its own max
        dos_plot = dos[mask]
        if dos_plot.max() > 0:
            dos_plot = dos_plot / dos_plot.max()
        
        ax.plot(freq[mask], dos_plot, 
                color=colors[label], linewidth=0.8)
        ax.fill_between(freq[mask], dos_plot, 
                        alpha=0.3, color=colors[label])
        ax.set_ylabel('DoS (norm.)')
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, freq_max)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Frequency (cm$^{-1}$)')
    plt.suptitle('Atom-Resolved Vibrational Density of States',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig('partial_dos_individual.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: partial_dos_individual.png")
    
    # ---- Plot 2: Overlay comparison ----
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for label, (freq, dos) in dos_data.items():
        mask = freq <= freq_max
        dos_plot = dos[mask]
        if dos_plot.max() > 0:
            dos_plot = dos_plot / dos_plot.max()
        
        ax.plot(freq[mask], dos_plot, color=colors[label],
                linewidth=1.2, label=label, alpha=0.8)
    
    ax.set_xlabel('Frequency (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Partial DoS (normalized)', fontsize=12)
    ax.set_title('Atom-Resolved vDoS — Melamine Crystal', 
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, freq_max)
    ax.grid(True, alpha=0.3)
    
    # Annotate key regions
    regions = [
        (50, 200, 'Lattice\nmodes', 'gray'),
        (600, 900, 'Ring\ndeform.', 'lightblue'),
        (1000, 1600, 'Ring C-N\nstretch', 'lightyellow'),
        (1600, 1700, 'NH₂\nscissor', 'lightgreen'),
        (3100, 3600, 'N-H\nstretch', 'lightyellow'),
    ]
    
    for xmin, xmax, text, color in regions:
        ax.axvspan(xmin, xmax, alpha=0.1, color=color)
        ax.annotate(text, xy=((xmin+xmax)/2, 0.95), 
                    fontsize=8, ha='center', style='italic',
                    va='top')
    
    plt.tight_layout()
    plt.savefig('partial_dos_overlay.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: partial_dos_overlay.png")
    
    # ---- Plot 3: Log scale to see weak features ----
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for label, (freq, dos) in dos_data.items():
        mask = (freq > 0) & (freq <= freq_max)
        dos_plot = dos[mask]
        if dos_plot.max() > 0:
            dos_plot = dos_plot / dos_plot.max()
        
        # Add small offset to avoid log(0)
        ax.semilogy(freq[mask], dos_plot + 1e-6, 
                    color=colors[label],
                    linewidth=1.0, label=label, alpha=0.8)
    
    ax.set_xlabel('Frequency (cm$^{-1}$)', fontsize=12)
    ax.set_ylabel('Partial DoS (log scale)', fontsize=12)
    ax.set_title('Atom-Resolved vDoS — Log Scale '
                 '(reveals weak features)', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0, freq_max)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('partial_dos_logscale.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("  Saved: partial_dos_logscale.png")


if __name__ == "__main__":
    main()
