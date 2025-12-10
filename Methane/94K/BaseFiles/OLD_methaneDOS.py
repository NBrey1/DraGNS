import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft

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
def calculate_dos(time, vacf, name):
    # Get time step
    dt = time[1] - time[0]  # in ps
    
    # Window function to reduce noise
    window = np.blackman(len(vacf))
    vacf_windowed = vacf * window
    
    # FFT to get power spectrum (DOS)
    n = len(vacf)
    dos = np.abs(rfft(vacf_windowed))
    freq = np.fft.rfftfreq(n, dt) * 33.356  # Convert from THz to cm^-1
    
    # Save DOS data
    with open(f'methanedos_{name}.xvg', 'w') as f:
        f.write('# DOS calculated from VACF\n')
        f.write(f'@    title "Density of States - {name}"\n')
        f.write('@    xaxis  label "Frequency (cm\\S-1\\N)"\n')
        f.write('@    yaxis  label "DOS (arbitrary units)"\n')
        for i in range(len(freq)):
            f.write(f"{freq[i]:.6f}    {dos[i]:.6f}\n")
    
    return freq, dos

# Process each group
groups = ["system", "C", "H"]
all_freq = {}
all_dos = {}

# Process each VACF file
for group in groups:
    filename = f'methanevacf_{group}.xvg'
    try:
        time, vacf = read_xvg(filename)
        all_freq[group], all_dos[group] = calculate_dos(time, vacf, group)
        print(f"Processed {filename}")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Skipping.")

# Plot individual DOS
for group in all_freq.keys():
    plt.figure(figsize=(10, 6))
    plt.plot(all_freq[group], all_dos[group])
    plt.xlim(0, 4000)  
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('DOS (arbitrary units)')
    plt.title(f'Vibrational Density of States - {group.upper()}')
    plt.savefig(f'methane_dos_{group}.png', dpi=300)
    plt.close()

# Plot combined DOS for comparison
if len(all_freq) > 1:
    plt.figure(figsize=(12, 7))
    
    for group in all_freq.keys():
        # Normalize DOS for better comparison
        normalized_dos = all_dos[group] / np.max(all_dos[group])
        plt.plot(all_freq[group], normalized_dos, label=group.upper())
    
    plt.xlim(0, 4000)
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Normalized DOS')
    plt.title('Comparison of Vibrational Density of States')
    plt.legend()
    plt.savefig('methane_dos_comparison.png', dpi=300)
    plt.show()
