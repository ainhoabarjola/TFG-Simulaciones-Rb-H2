"""
Análisis de Distribución Radial (Gaussian Broadening).
Calcula y grafica la función de densidad radial de las moléculas de disolvente 
respecto al núcleo iónico, aplicando un ensanchamiento gaussiano a las distancias 
discretas para revelar la separación topológica de las capas de solvatación.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List

# ==============================================================================
# 1. CONFIGURACIÓN ESTÉTICA GLOBAL (ESTÁNDAR ACADÉMICO / TFG)
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.labelweight'] = 'bold'


# ==============================================================================
# 2. PROCESAMIENTO DE DISTANCIAS
# ==============================================================================

def load_radial_distances(filename: str, central_ion_symbol: str) -> np.ndarray:
    """
    Parsea las coordenadas espaciales del clúster y extrae la matriz de 
    distancias euclidianas entre el catión central y los pseudo-átomos de H2.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[ERROR] El archivo de topología '{filename}' no fue localizado.")

    with open(filename, 'r') as f:
        lines = f.readlines()

    coordinate_lines = [line.strip() for line in lines[2:] if line.strip()]

    solvent_coords = []
    central_ion_coord = None
    
    for line in coordinate_lines:
        parts = line.split()
        if len(parts) >= 4:
            atom_symbol = parts[0]
            try:
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                
                # 'H' representa el pseudo-átomo de H2 molecular en los .xyz
                if atom_symbol == 'H': 
                    solvent_coords.append(coords)
                elif atom_symbol == central_ion_symbol:
                    central_ion_coord = coords
            except ValueError:
                continue

    if central_ion_coord is None:
        raise ValueError(f"[ERROR] Núcleo iónico '{central_ion_symbol}' no detectado en la matriz.")
    if not solvent_coords:
        raise ValueError("[ERROR] No se detectaron moléculas de disolvente en el archivo.")

    solvent_coords_array = np.array(solvent_coords)
    radial_distances = np.linalg.norm(solvent_coords_array - central_ion_coord, axis=1)
    
    return radial_distances


# ==============================================================================
# 3. ENSANCHAMIENTO GAUSSIANO Y REPRESENTACIÓN GRÁFICA
# ==============================================================================

def generate_radial_distribution_plot(filename: str, central_ion_symbol: str, sigma: float = 0.10) -> str:
    """
    Aplica una función de ensanchamiento gaussiano sobre las distancias 
    radiales discretas y renderiza el gráfico de densidad correspondiente.
    """
    print("===================================================================")
    print(f" Generando Función de Distribución Radial: {filename} ")
    print("===================================================================\n")
    
    try:
        distances = load_radial_distances(filename, central_ion_symbol)
    except Exception as e:
        print(f"[ERROR CRÍTICO] {e}")
        return ""

    num_atoms = len(distances)
    if num_atoms == 0:
        print("[AVISO] Espacio de fases vacío. No hay distancias a procesar.")
        return ""

    print(f" -> Ligandos procesados: {num_atoms}")
    print(f" -> Parámetro de ensanchamiento (Sigma): {sigma:.2f} Å\n")

    # Definición del dominio espacial (r)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    r_min = max(0, min_dist - 0.5)
    r_max = max_dist + 0.5
    r_values = np.linspace(r_min, r_max, 500) 

    # Construcción analítica de la densidad D(r)
    distribution_function = np.zeros_like(r_values) 
    for d_i in distances:
        gaussian_peak = np.exp(-(r_values - d_i)**2 / (2 * sigma**2))
        distribution_function += gaussian_peak

    # Renderizado gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Curva de densidad de probabilidad
    ax.plot(r_values, distribution_function, label=r'$D(r)$', color='darkblue', linewidth=3)
    
    # Trazado de las coordenadas discretas subyacentes
    if num_atoms < 50:
        for d_i in distances:
            ax.axvline(d_i, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formato de ejes
    ax.set_xlabel(r'Distancia $r$ (Å)', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'Densidad Radial $D(r)$', fontsize=16, fontweight='bold')
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_fontsize(14)
    
    ax.legend(fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Exportación en alta resolución para el manuscrito
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_radial_dist.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Gráfica renderizada con éxito: '{output_filename}'")
    return output_filename


# ==============================================================================
# 4. ORQUESTACIÓN DEL SCRIPT
# ==============================================================================

def main():
    # Parámetros de ejecución
    input_topology = 'N38_Lisa.xyz'
    core_ion = 'Rb'
    broadening_sigma = 0.10
    
    generate_radial_distribution_plot(
        filename=input_topology, 
        central_ion_symbol=core_ion, 
        sigma=broadening_sigma
    )

if __name__ == '__main__':
    main()