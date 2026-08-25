import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

THZ_TO_CM1 = 33.35641

def read_xvg(path):
    """Read an XVG file and return freq, intensity (as 1D numpy arrays)."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("@", "#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                data.append((x, y))
    if len(data) == 0:
        raise ValueError("No numeric data found in file.")
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]

def guess_input_unit(freq):
    """Guess whether freq array is in THz or cm^-1 by max value heuristics."""
    maxf = np.nanmax(freq)
    # heuristic: THz spectra usually have max < ~50 (often <10). cm^-1 goes to thousands.
    if maxf < 50:
        return "THz"
    else:
        return "cm^-1"

def convert_to_cm1(freq, intensity, input_unit=None):
    """Return freq_cm1, intensity_modes_per_cm1, and the detected/used input_unit."""
    if input_unit is None:
        input_unit = guess_input_unit(freq)
    if input_unit.lower().startswith("thz"):
        freq_cm1 = freq * THZ_TO_CM1
        intensity_cm1 = intensity / THZ_TO_CM1   # modes per THz -> modes per cm^-1
    elif input_unit.lower() in ("cm^-1", "cm-1", "cm"):
        freq_cm1 = freq.copy()
        intensity_cm1 = intensity.copy()
    else:
        raise ValueError(f"Unknown input_unit: {input_unit}")
    return freq_cm1, intensity_cm1, input_unit

def analyze_peaks(freq_cm1, intensity_cm1,
                  min_prominence_frac=0.05,
                  min_height_frac=0.01,
                  min_fwhm=0.1,
                  use_adaptive_thresholds=True):
    """
    Find peaks in DOS spectrum and compute FWHM.
    
    Parameters
    ----------
    freq_cm1 : array-like
        Frequencies in cm^-1
    intensity_cm1 : array-like
        DOS values in modes per cm^-1
    min_prominence_frac : float
        Minimum peak prominence relative to max intensity (default: 0.05 = 5%)
    min_height_frac : float
        Minimum peak height relative to max intensity (default: 0.01 = 1%)
    min_fwhm : float
        Minimum full width at half maximum (cm^-1) to filter out spurious peaks (default: 0.1)
    use_adaptive_thresholds : bool
        If True, use multiple passes with different thresholds to catch peaks of varying heights
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: freq_cm^-1, intensity, fwhm_cm^-1, left_freq, right_freq, prominence
    """
    maxI = np.max(intensity_cm1)

    if use_adaptive_thresholds:
        # Multi-pass approach: find peaks with progressively lower thresholds
        all_peaks = []
        all_props = []
        
        # Pass 1: Find the major peaks (high threshold)
        peaks1, props1 = find_peaks(
            intensity_cm1,
            height=maxI * max(min_height_frac, 0.15),  # At least 15% or user threshold
            prominence=maxI * max(min_prominence_frac, 0.15)
        )
        
        # Pass 2: Find medium peaks (lower threshold) but exclude regions around major peaks
        mask = np.ones_like(intensity_cm1, dtype=bool)
        for peak in peaks1:
            # Exclude ±20 points around each major peak
            start = max(0, peak - 20)
            end = min(len(intensity_cm1), peak + 21)
            mask[start:end] = False
        
        # Create masked intensity for finding medium peaks
        masked_intensity = intensity_cm1.copy()
        masked_intensity[~mask] = 0
        
        peaks2, props2 = find_peaks(
            masked_intensity,
            height=maxI * min_height_frac,
            prominence=maxI * max(min_prominence_frac, 0.05)  # Lower prominence for medium peaks
        )
        
        # Combine results
        all_peaks = np.concatenate([peaks1, peaks2])
        # Combine properties dictionaries
        combined_props = {}
        for key in props1.keys():
            combined_props[key] = np.concatenate([props1[key], props2.get(key, [])])
        
        peaks = all_peaks
        props = combined_props
        
        print(f"Found {len(peaks1)} major peaks and {len(peaks2)} medium peaks")
        
    else:
        # Single-pass approach (original method)
        peaks, props = find_peaks(
            intensity_cm1,
            height=maxI * min_height_frac,
            prominence=maxI * min_prominence_frac
        )

    if len(peaks) == 0:
        print("No peaks found with current thresholds. Consider lowering min_prominence_frac or min_height_frac.")
        return pd.DataFrame(columns=["freq_cm^-1", "intensity", "fwhm_cm^-1", "left_freq", "right_freq", "prominence"])

    # Compute FWHM using peak_widths
    widths_samples, height_at_width, left_ips, right_ips = peak_widths(
        intensity_cm1, peaks, rel_height=0.5
    )

    # Convert fractional indices to actual frequencies
    idx = np.arange(len(freq_cm1))
    left_freq = np.interp(left_ips, idx, freq_cm1)
    right_freq = np.interp(right_ips, idx, freq_cm1)
    fwhm_cm1 = right_freq - left_freq

    # Create DataFrame
    df = pd.DataFrame({
        "freq_cm^-1": freq_cm1[peaks],
        "intensity": intensity_cm1[peaks],
        "fwhm_cm^-1": fwhm_cm1,
        "left_freq": left_freq,
        "right_freq": right_freq,
        "prominence": props["prominences"]
    })

    # Apply FWHM filter to remove spurious narrow peaks
    df_filtered = df[df["fwhm_cm^-1"] >= min_fwhm].copy()
    
    if len(df_filtered) < len(df):
        print(f"Filtered out {len(df) - len(df_filtered)} peaks with FWHM < {min_fwhm} cm^-1")

    # Sort by frequency
    return df_filtered.sort_values("freq_cm^-1").reset_index(drop=True)

def integrate_dos(freq_cm1, intensity_cm1):
    """Numerical integral of DOS (modes) using trapezoid rule: ∫ g(ν) dν."""
    return np.trapezoid(intensity_cm1, freq_cm1)  # Updated to use trapezoid

def plot_dos_and_peaks(freq_cm1, intensity_cm1, peaks_df=None, show_integral=True, 
                       figsize=(12, 6), freq_range=None):
    """
    Plot DOS spectrum with identified peaks and FWHM bars.
    
    Parameters
    ----------
    freq_range : tuple, optional
        (min_freq, max_freq) to limit the x-axis range. Default shows full range.
    """
    plt.figure(figsize=figsize)
    plt.plot(freq_cm1, intensity_cm1, 'b-', linewidth=1.5, label='DOS')
    
    if peaks_df is not None and len(peaks_df) > 0:
        # Mark peaks
        plt.scatter(peaks_df["freq_cm^-1"], peaks_df["intensity"], 
                   color='red', marker='o', s=50, zorder=5, label='Peaks')
        
        # Draw FWHM horizontal bars
        for _, row in peaks_df.iterrows():
            # Draw FWHM bar at half maximum
            half_h = row["intensity"] / 2.0
            plt.hlines(half_h, row["left_freq"], row["right_freq"], 
                      colors='red', linewidth=2, alpha=0.7)
            
            # Annotate peak frequency
            plt.annotate(f'{row["freq_cm^-1"]:.1f}', 
                        xy=(row["freq_cm^-1"], row["intensity"]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    if show_integral:
        total = integrate_dos(freq_cm1, intensity_cm1)
        plt.text(0.02, 0.95, f"∫DOS dν = {total:.0f} modes", 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(facecolor="wheat", alpha=0.8, boxstyle="round,pad=0.3"))
    
    plt.xlabel("Frequency (cm⁻¹)", fontsize=12)
    plt.ylabel("DOS (modes per cm⁻¹)", fontsize=12)
    plt.title("Vibrational Density of States", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set frequency range if specified
    if freq_range is not None:
        plt.xlim(freq_range)
    
    plt.tight_layout()
    plt.show()

def print_peak_summary(peaks_df):
    """Print a nicely formatted summary of the identified peaks."""
    if len(peaks_df) == 0:
        print("No peaks identified.")
        return
    
    print(f"\nIdentified {len(peaks_df)} peaks:")
    print("-" * 70)
    print(f"{'Peak #':<8} {'Freq (cm⁻¹)':<12} {'Intensity':<12} {'FWHM (cm⁻¹)':<12} {'Prominence':<12}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(peaks_df.iterrows(), 1):
        print(f"{i:<8} {row['freq_cm^-1']:<12.1f} {row['intensity']:<12.3f} "
              f"{row['fwhm_cm^-1']:<12.1f} {row['prominence']:<12.3f}")

# ---------------------------
# Main analysis function
# ---------------------------
def analyze_dos_file(path_to_xvg, n_atoms=None, 
                     min_prominence_frac=0.05, min_height_frac=0.01, min_fwhm=0.1,
                     use_adaptive_thresholds=True, plot=True, freq_range=(0, 5000)):
    """
    Complete DOS analysis workflow.
    
    Parameters
    ----------
    path_to_xvg : str
        Path to the XVG file
    n_atoms : int, optional
        Number of atoms for normalization check
    min_prominence_frac : float
        Minimum peak prominence as fraction of max intensity
    min_height_frac : float
        Minimum peak height as fraction of max intensity  
    min_fwhm : float
        Minimum FWHM in cm^-1 to filter spurious peaks (default: 0.1)
    use_adaptive_thresholds : bool
        Use multi-pass approach to find peaks of varying heights (default: True)
    plot : bool
        Whether to generate plot
    freq_range : tuple
        (min_freq, max_freq) for plot x-axis range. Default (0, 5000)
        
    Returns
    -------
    tuple
        (freq_cm1, intensity_cm1, peaks_df)
    """
    print(f"Analyzing DOS file: {path_to_xvg}")
    
    # Read data
    freq, intensity = read_xvg(path_to_xvg)
    print(f"Read {len(freq)} data points")
    
    # Convert to cm^-1 if needed
    freq_cm1, intensity_cm1, used_unit = convert_to_cm1(freq, intensity, input_unit=None)
    print(f"Input units detected/used: {used_unit}")
    
    # Check integral
    total_modes = integrate_dos(freq_cm1, intensity_cm1)
    print(f"Integrated modes (∫DOS dν) = {total_modes:.1f}")
    
    if n_atoms is not None:
        expected = 3 * int(n_atoms)
        print(f"Expected total modes (3×N_atoms) = {expected}")
        if abs(total_modes - expected) > 1:
            scale = expected / total_modes
            print(f"Rescaling intensities by factor {scale:.4f} to match 3N")
            intensity_cm1 *= scale
            total_modes = integrate_dos(freq_cm1, intensity_cm1)
    
    # Find peaks
    peaks_df = analyze_peaks(freq_cm1, intensity_cm1,
                           min_prominence_frac=min_prominence_frac, 
                           min_height_frac=min_height_frac,
                           min_fwhm=min_fwhm,
                           use_adaptive_thresholds=use_adaptive_thresholds)
    
    # Print results
    print_peak_summary(peaks_df)
    
    # Plot if requested
    if plot:
        plot_dos_and_peaks(freq_cm1, intensity_cm1, peaks_df, freq_range=freq_range)
    
    return freq_cm1, intensity_cm1, peaks_df

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Change these parameters as needed
    path_to_xvg = "methanedos_system.xvg"   # <-- replace with your path
    n_atoms = 250   # optional: set to number of atoms for normalization check
    
    # Analysis parameters for adaptive peak detection
    min_prominence_frac = 0.08  # Lower for medium peaks  
    min_height_frac = 0.05      # 5% minimum height
    min_fwhm = 0.1              # Very permissive FWHM
    
    # Run analysis
    freq_cm1, intensity_cm1, peaks_df = analyze_dos_file(
        path_to_xvg, n_atoms=n_atoms,
        min_prominence_frac=min_prominence_frac,
        min_height_frac=min_height_frac,
        min_fwhm=min_fwhm,
        freq_range=(0, 5000)  # Limit plot to 0-5000 cm^-1
    )
    
    # You can also save results to CSV
    if len(peaks_df) > 0:
        output_file = path_to_xvg.replace('.xvg', '_peaks.csv')
        peaks_df.to_csv(output_file, index=False)
        print(f"\nPeak data saved to: {output_file}")
