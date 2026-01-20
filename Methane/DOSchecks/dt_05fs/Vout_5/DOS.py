import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import windows
from scipy.interpolate import interp1d

def read_xvg(filename, max_time_ps=None):
    """Read XVG file"""
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                time_val = float(parts[0])
                if max_time_ps is not None and time_val > max_time_ps:
                    break
                x.append(time_val)
                y.append(float(parts[1]))
    return np.array(x), np.array(y)

def interpolate_to_uniform(time, vacf, target_dt=0.0001):
    """
    Interpolate VACF to uniform timesteps
    
    CRITICAL for FFT when trajectory has non-uniform timesteps
    """
    print(f"  Original data: {len(time):,} points")
    print(f"  Time range: {time[0]:.6f} to {time[-1]:.6f} ps")
    print(f"  Original dt: {np.mean(np.diff(time)):.9f} ps (mean)")
    print(f"  Original dt: {np.std(np.diff(time)):.9f} ps (std)")
    
    # Create uniform time grid
    time_uniform = np.arange(time[0], time[-1], target_dt)
    
    # Interpolate VACF onto uniform grid
    interpolator = interp1d(time, vacf, kind='cubic', fill_value='extrapolate')
    vacf_uniform = interpolator(time_uniform)
    
    print(f"  → Interpolated to uniform grid: {len(time_uniform):,} points")
    print(f"  → Uniform dt: {target_dt:.9f} ps ({target_dt*1000:.6f} fs)")
    
    return time_uniform, vacf_uniform

def calculate_dos(time, vacf, name, window_type='tukey', interpolate=True, target_dt=0.0001):
    """
    Calculate DOS from VACF
    
    Parameters:
    -----------
    interpolate : bool
        If True, interpolate to uniform timesteps (REQUIRED for non-uniform data)
    target_dt : float
        Target timestep in ps after interpolation
    """
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    # Check if interpolation is needed
    dt_values = np.diff(time)
    dt_mean = np.mean(dt_values)
    dt_std = np.std(dt_values)
    
    if dt_std / dt_mean > 0.01:  # More than 1% variation
        print(f"⚠ WARNING: Non-uniform timesteps detected!")
        print(f"  Mean: {dt_mean:.9f} ps, Std: {dt_std:.9f} ps")
        if interpolate:
            print(f"  → Applying interpolation to fix...")
            time, vacf = interpolate_to_uniform(time, vacf, target_dt)
        else:
            print(f"  ✗ Interpolation disabled - FFT results will be WRONG!")
    else:
        print(f"✓ Uniform timesteps detected")
        print(f"  dt = {dt_mean:.9f} ps ({dt_mean*1000:.6f} fs)")
    
    n = len(time)
    dt = np.mean(np.diff(time))
    
    # Calculate Nyquist frequency
    nyquist_cm = 1 / (2 * dt * 1e-12 * 2.998e10)
    print(f"  Data points: {n:,}")
    print(f"  Nyquist frequency: {nyquist_cm:.0f} cm⁻¹")
    
    # Window function
    if window_type == 'tukey':
        window = windows.tukey(n, alpha=0.25)
        print(f"  Window: Tukey (alpha=0.25)")
    elif window_type == 'hann':
        window = np.hanning(n)
        print(f"  Window: Hann")
    elif window_type is None:
        window = np.ones(n)
        print(f"  Window: None")
    else:
        window = np.blackman(n)
        print(f"  Window: Blackman")
    
    vacf_windowed = vacf * window
    
    # Zero padding
    n_pad = 2**int(np.ceil(np.log2(n * 2)))
    vacf_padded = np.pad(vacf_windowed, (0, n_pad - n), mode='constant')
    print(f"  Zero padding: {n:,} → {n_pad:,}")
    
    # FFT
    dos_raw = np.abs(rfft(vacf_padded))
    freq_hz = rfftfreq(n_pad, dt * 1e-12)
    freq_cm = freq_hz / 2.998e10
    
    # Normalize
    dos = dos_raw * dt
    
    # Normalize to mode count
    n_atoms = 5 * 500
    n_modes = 3 * n_atoms - 6
    mask = freq_cm <= 10000
    total_area = np.trapezoid(dos[mask], freq_cm[mask])
    
    if total_area > 0:
        dos = dos * (n_modes / total_area)
        print(f"  Normalized to {n_modes} modes")
    
    # Find peaks
    print(f"\n  Peak analysis:")
    regions = [
        ('Bending', 1000, 2000),
        ('C-H stretch', 2800, 3200),
        ('1st overtone', 5500, 6500),
        ('2nd overtone', 8000, 9000)
    ]
    
    for region_name, f_min, f_max in regions:
        mask_region = (freq_cm >= f_min) & (freq_cm <= f_max)
        if np.any(mask_region):
            region_dos = dos[mask_region]
            if np.max(region_dos) > 0:
                region_freq = freq_cm[mask_region]
                peak_idx = np.argmax(region_dos)
                intensity = region_dos[peak_idx]
                print(f"    {region_name:15s}: {region_freq[peak_idx]:6.0f} cm⁻¹ (intensity: {intensity:.2e})")
            else:
                print(f"    {region_name:15s}: No peak detected")
    
    print(f"{'='*70}\n")
    
    return freq_cm, dos

# =============================================================================
# CONFIGURATION
# =============================================================================

# CRITICAL: Set to True to fix non-uniform timesteps
INTERPOLATE = True

# Target timestep after interpolation (ps)
TARGET_DT = 0.0001  # 0.1 fs

# Maximum time to use (ps)
MAX_TIME_PS = 20

# Window type: None, 'tukey', 'hann', 'blackman'
WINDOW_TYPE = 'tukey'

# Files to process
files = {
    'System': 'vacf.xvg',
}

# =============================================================================
# MAIN
# =============================================================================

print("="*70)
print("DOS CALCULATION WITH NON-UNIFORM TIMESTEP FIX")
print("="*70)
print(f"\nSettings:")
print(f"  Interpolate: {INTERPOLATE}")
print(f"  Target dt: {TARGET_DT} ps ({TARGET_DT*1000} fs)")
print(f"  Max time: {MAX_TIME_PS} ps")
print(f"  Window: {WINDOW_TYPE}")
print("="*70)

all_freq = {}
all_dos = {}

for name, filename in files.items():
    try:
        time, vacf = read_xvg(filename, MAX_TIME_PS)
        freq, dos = calculate_dos(time, vacf, name, WINDOW_TYPE, INTERPOLATE, TARGET_DT)
        all_freq[name] = freq
        all_dos[name] = dos
    except FileNotFoundError:
        print(f"\n✗ File not found: {filename}\n")
        continue
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        continue

if not all_freq:
    print("\n❌ No files processed!")
    exit(1)

# Create plots
print("="*70)
print("CREATING PLOTS")
print("="*70)

for name in all_freq.keys():
    fig = plt.figure(figsize=(16, 12))
    
    freq = all_freq[name]
    dos = all_dos[name]
    
    # Panel 1: Full spectrum
    ax1 = plt.subplot(3, 2, 1)
    mask = freq <= 10000
    ax1.plot(freq[mask], dos[mask], 'b-', linewidth=1)
    ax1.set_xlim(0, 10000)
    ax1.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax1.set_ylabel('DOS', fontsize=10)
    ax1.set_title('Full Spectrum (0-10,000 cm⁻¹)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    for freq_val, label, color in [(1500, 'Bend', 'green'), (3000, 'C-H', 'red'), 
                                     (6000, '1st', 'purple'), (9000, '2nd', 'orange')]:
        ax1.axvline(freq_val, color=color, linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.text(freq_val, ax1.get_ylim()[1]*0.95, label, ha='center', fontsize=7)
    
    # Panel 2: Bending
    ax2 = plt.subplot(3, 2, 2)
    mask = (freq >= 500) & (freq <= 2000)
    ax2.plot(freq[mask], dos[mask], 'g-', linewidth=1.5)
    ax2.set_xlim(500, 2000)
    ax2.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax2.set_ylabel('DOS', fontsize=10)
    ax2.set_title('Bending Region (500-2000 cm⁻¹)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    if np.any(mask) and np.max(dos[mask]) > 0:
        peak_idx = np.argmax(dos[mask])
        ax2.plot(freq[mask][peak_idx], dos[mask][peak_idx], 'ro', markersize=8)
        ax2.text(freq[mask][peak_idx], dos[mask][peak_idx]*1.1, 
                f'{freq[mask][peak_idx]:.0f}', ha='center', fontsize=8, fontweight='bold')
    
    # Panel 3: C-H stretch
    ax3 = plt.subplot(3, 2, 3)
    mask = (freq >= 2500) & (freq <= 3500)
    ax3.plot(freq[mask], dos[mask], 'r-', linewidth=2.5)
    ax3.set_xlim(2500, 3500)
    ax3.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax3.set_ylabel('DOS', fontsize=10)
    ax3.set_title('C-H Stretch (2500-3500 cm⁻¹)', fontweight='bold', color='red', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(2917, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.text(2917, ax3.get_ylim()[1]*0.85, 'Exp:\n2917', ha='center', fontsize=7, color='gray')
    
    if np.any(mask) and np.max(dos[mask]) > 0.01:
        peak_idx = np.argmax(dos[mask])
        peak_freq = freq[mask][peak_idx]
        peak_dos = dos[mask][peak_idx]
        ax3.plot(peak_freq, peak_dos, 'ko', markersize=10)
        ax3.text(peak_freq, peak_dos*1.15, f'{peak_freq:.0f}', 
                ha='center', fontweight='bold', fontsize=11, color='darkred')
    else:
        ax3.text(0.5, 0.5, 'NO PEAK\nSTILL A PROBLEM!', 
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Panel 4: First overtone
    ax4 = plt.subplot(3, 2, 4)
    mask = (freq >= 5000) & (freq <= 7000)
    ax4.plot(freq[mask], dos[mask], 'm-', linewidth=1.5)
    ax4.set_xlim(5000, 7000)
    ax4.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax4.set_ylabel('DOS', fontsize=10)
    ax4.set_title('1st Overtone (5000-7000 cm⁻¹)', fontweight='bold', color='purple')
    ax4.grid(True, alpha=0.3)
    
    if np.any(mask) and np.max(dos[mask]) > 0.01:
        peak_idx = np.argmax(dos[mask])
        ax4.plot(freq[mask][peak_idx], dos[mask][peak_idx], 'ro', markersize=8)
        ax4.text(freq[mask][peak_idx], dos[mask][peak_idx]*1.1, 
                f'{freq[mask][peak_idx]:.0f}', ha='center', fontsize=8, fontweight='bold')
    
    # Panel 5: Second overtone
    ax5 = plt.subplot(3, 2, 5)
    mask = (freq >= 7500) & (freq <= 9500)
    ax5.plot(freq[mask], dos[mask], 'orange', linewidth=1.5)
    ax5.set_xlim(7500, 9500)
    ax5.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax5.set_ylabel('DOS', fontsize=10)
    ax5.set_title('2nd Overtone (7500-9500 cm⁻¹)', fontweight='bold', color='orange')
    ax5.grid(True, alpha=0.3)
    
    if np.any(mask) and np.max(dos[mask]) > 0.01:
        peak_idx = np.argmax(dos[mask])
        ax5.plot(freq[mask][peak_idx], dos[mask][peak_idx], 'ro', markersize=8)
        ax5.text(freq[mask][peak_idx], dos[mask][peak_idx]*1.1, 
                f'{freq[mask][peak_idx]:.0f}', ha='center', fontsize=8, fontweight='bold')
    
    # Panel 6: Log scale
    ax6 = plt.subplot(3, 2, 6)
    mask = (freq > 0) & (freq <= 10000) & (dos > 0)
    ax6.semilogy(freq[mask], dos[mask], 'b-', linewidth=1)
    ax6.set_xlim(0, 10000)
    ax6.set_xlabel('Frequency (cm⁻¹)', fontsize=10)
    ax6.set_ylabel('DOS (log scale)', fontsize=10)
    ax6.set_title('Full Spectrum - Log Scale', fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    
    for freq_val in [1500, 3000, 6000, 9000]:
        ax6.axvline(freq_val, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
    
    plt.suptitle(f'Vibrational Density of States - {name}\n' + 
                 f'(Interpolated to uniform dt={TARGET_DT} ps, window={WINDOW_TYPE})', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'dos_fixed_{name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: dos_fixed_{name}.png")
    plt.show()

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\nYou should now see:")
print("  ✓ Clear C-H stretch peak around 2900-3100 cm⁻¹")
print("  ✓ First overtone around 5800-6000 cm⁻¹ (weaker)")
print("  ✓ Second overtone around 8500-9000 cm⁻¹ (very weak)")
print("\nIf peaks are still missing, the issue is with your VACF quality,")
print("not the timestep. You may need to re-run the simulation.")
