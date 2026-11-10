#!/usr/bin/env python3
"""
Convert DOS frequency units from ps^-1 to cm^-1
"""

import numpy as np
import matplotlib.pyplot as plt

# Conversion factor: 1 ps^-1 = 33.356409 cm^-1
PS_TO_CM = 33.356409

# Read the DOS file (from gmx dos)
print("Reading dos.xvg...")
data = np.loadtxt('dos.xvg', comments=['#', '@'])

freq_ps = data[:, 0]  # Frequency in ps^-1
dos = data[:, 1]      # DOS values

# Convert to cm^-1
freq_cm = freq_ps * PS_TO_CM

print(f"Original frequency range: {freq_ps[0]:.3f} - {freq_ps[-1]:.3f} ps^-1")
print(f"Converted frequency range: {freq_cm[0]:.1f} - {freq_cm[-1]:.1f} cm^-1")

# Save converted data
output_file = 'dos_cm.xvg'
header = f"Frequency (cm^-1)  DOS\nConverted from ps^-1 using factor {PS_TO_CM}"
np.savetxt(output_file, 
           np.column_stack([freq_cm, dos]),
           header=header,
           comments='# ',
           fmt='%.6f  %.10e')

print(f"\nConverted file saved as: {output_file}")

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot in ps^-1
ax1.plot(freq_ps, dos, 'b-', linewidth=1)
ax1.set_xlabel('Frequency (ps⁻¹)', fontsize=12)
ax1.set_ylabel('DOS', fontsize=12)
ax1.set_title('Original (ps⁻¹)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot in cm^-1
ax2.plot(freq_cm, dos, 'r-', linewidth=1)
ax2.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
ax2.set_ylabel('DOS', fontsize=12)
ax2.set_title('Converted (cm⁻¹)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 8000)  # Show up to 8000 cm^-1


plt.tight_layout()
plt.savefig('dos_conversion.png', dpi=300, bbox_inches='tight')
print(f"Plot saved as: dos_conversion.png")

plt.show()

print("\nConversion complete!")
print(f"Conversion factor used: 1 ps^-1 = {PS_TO_CM} cm^-1")
