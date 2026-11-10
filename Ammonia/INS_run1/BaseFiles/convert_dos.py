import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, h, e
from scipy.fft import fft, fftfreq
import re

def parse_xvg(filename):
    """Parse GROMACS xvg file and extract data."""
    x = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            values = line.strip().split()
            if len(values) >= 2:
                x.append(float(values[0]))
                y.append(float(values[1]))
    return np.array(x), np.array(y)

def vacf_to_dos(time, vacf):
    """Convert velocity autocorrelation function to density of states via Fourier transform."""
    # Ensure the VACF ends at zero (apply windowing)
    window = np.hanning(len(vacf))
    vacf_windowed = vacf * window
    
    # Perform FFT
    dt = time[1] - time[0]  # time step in ps
    n = len(time)
    dos = np.abs(fft(vacf_windowed)[0:n//2])
    
    # Generate frequency axis in cm^-1
    # Convert from 1/ps to cm^-1 (1 ps^-1 = 33.356 cm^-1)
    freq = fftfreq(n, dt)[0:n//2] * 33.356
    
    return freq, dos

def convert_dos_units(input_file, output_prefix):
    # Physical constants
    light_speed = c * 100  # m/s to cm/s
    h_planck = h
    electron_volt = e
    
    # Read the VACF data from the xvg file
    time, vacf = parse_xvg(input_file)
    
    # Convert VACF to DOS
    wavenumber, dos_values = vacf_to_dos(time, vacf)
    
    # Filter out non-positive wavenumbers (to avoid division by zero)
    valid_idx = wavenumber > 0
    wavenumber = wavenumber[valid_idx]
    dos_values = dos_values[valid_idx]
    
    # Conversion factors
    # 2. Convert wavenumber (cm^-1) to energy (eV)
    energy_ev = (h_planck * light_speed * wavenumber) / electron_volt
    
    # 3. Convert wavenumber (cm^-1) to neutron wavelength (Angstrom)
    wavelength_angstrom = 10000 / wavenumber  # 1/wavenumber (cm) to Angstrom
    
    # Create plots with different x-axes
    # Figure 1: Wavenumber (cm^-1)
    plt.figure(figsize=(10, 6))
    plt.plot(wavenumber, dos_values)
    plt.xlabel('Wavenumber (cm$^{-1}$)')
    plt.ylabel('Density of States')
    plt.title('DOS vs Wavenumber')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_wavenumber.png', dpi=300)
    
    # Figure 2: Energy (eV)
    plt.figure(figsize=(10, 6))
    plt.plot(energy_ev, dos_values)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Density of States')
    plt.title('DOS vs Energy T=94K, P=1.45 bar')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_energy.png', dpi=300)
    
    # Figure 3: Neutron Wavelength (Angstrom)
    plt.figure(figsize=(10, 6))
    # Filter out extremely large wavelengths
    mask = wavelength_angstrom < 100
    plt.plot(wavelength_angstrom[mask], dos_values[mask])
    plt.xlabel('Neutron Wavelength (Å)')
    plt.ylabel('Density of States')
    plt.title('DOS vs Neutron Wavelength T=94K, P=1.45 bar')
    plt.grid(True)
    plt.savefig(f'{output_prefix}_wavelength.png', dpi=300)
    
    # Save data in different units
    output_data = np.column_stack((wavenumber, energy_ev, wavelength_angstrom, dos_values))
    np.savetxt(f'{output_prefix}_converted.dat', output_data, 
               header='Wavenumber(cm^-1) Energy(eV) Wavelength(Angstrom) DOS', 
               fmt='%12.6f %12.6e %12.6f %12.6e')
    
    print(f"Conversion complete. Data saved to {output_prefix}_converted.dat")
    print(f"Plots saved as {output_prefix}_wavenumber.png, {output_prefix}_energy.png, and {output_prefix}_wavelength.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python convert_vacf_to_dos.py input_vacf_file output_prefix")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    
    convert_dos_units(input_file, output_prefix)
