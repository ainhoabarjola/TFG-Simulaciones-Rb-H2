"""
Algoritmo de Optimización Inversa (Secuencia de Decapado).
Desgrana secuencialmente agregados moleculares macroscópicos eliminando 
iterativamente la molécula de disolvente más distante del núcleo iónico, 
evaluando la energía estática residual para establecer la serie termodinámica completa.
"""

import numpy as np
import os
from typing import List, Dict, Tuple
from collections import Counter

# ==============================================================================
# 1. PARÁMETROS GLOBALES DEL ALGORITMO
# ==============================================================================
FILE_INPUT = 'N76_Hidrogenos.xyz'               # Macroestructura inicial a decapar
FILE_OUT_SUMMARY = 'energia_pelado_secuencial.txt'  # Registro termodinámico de salida
SAVE_INTERMEDIATE_XYZ = True                    # Extraer topología en cada paso N


# ==============================================================================
# 2. EVALUACIÓN DE LA SUPERFICIE DE ENERGÍA POTENCIAL (SEP)
# ==============================================================================

def v_rb_h2(x1: float, y1: float, z1: float, x_cs: float, y_cs: float, z_cs: float) -> float:
    """ Interacción Soluto-Disolvente (Rb+ - H2) mediante el modelo ILJ. """
    m = 4.0
    eps = 50.4594   # Profundidad del pozo (meV)
    beta = 7.5
    rm = 3.2133     # Distancia de equilibrio (Angstroms)

    r = np.sqrt((x_cs - x1)**2 + (y_cs - y1)**2 + (z_cs - z1)**2)
    
    if r < 1e-8:
        return 0.0

    x = r / rm
    n = beta + 4 * (x**2)

    term_rep = (m / (n - m)) * (1/x)**n
    term_att = (n / (n - m)) * (1/x)**m

    return eps * (term_rep - term_att)


def v_h2_h2(r: float) -> float:
    """ Interacción Disolvente-Disolvente (H2 - H2) mediante el modelo ILJ. """
    m = 6.0
    eps = 3.07      # Profundidad del pozo (meV)
    beta = 7.0
    rm = 3.47       # Distancia de equilibrio (Angstroms)

    if r < 1e-8:
        return 0.0

    x = r / rm
    n = beta + 4 * (x**2)

    term_rep = (m / (n - m)) * (1/x)**n
    term_att = (n / (n - m)) * (1/x)**m

    return eps * (term_rep - term_att)


# ==============================================================================
# 3. DISECCIÓN TOPOLÓGICA Y ORDENACIÓN ESPACIAL
# ==============================================================================

def read_and_sort_by_distance(filename: str) -> Tuple[Dict, List[Dict]]:
    """
    Lee el agregado .xyz, identifica automáticamente el centro de fuerzas (catión)
    y clasifica las moléculas de disolvente ordenándolas por su radio vector 
    de mayor a menor distancia (preparación para el decapado).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    atoms = []
    for line in lines[2:]:  # Omitir cabeceras estándar
        parts = line.split()
        if len(parts) >= 4:
            symb = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            atoms.append({'elem': symb, 'coords': coords})

    # Identificación heurística del catión central (elemento de cardinalidad 1)
    element_counts = Counter([a['elem'] for a in atoms])
    unique_elements = [e for e, c in element_counts.items() if c == 1]

    if not unique_elements:
        raise ValueError("[ERROR] Topología inválida: No se detectó un núcleo iónico central único.")

    core_elem = unique_elements[0]
    core_idx = next(i for i, a in enumerate(atoms) if a['elem'] == core_elem)
    core_atom = atoms.pop(core_idx)

    # Evaluación de distancias radiales euclidianas
    for a in atoms:
        a['r'] = np.linalg.norm(a['coords'] - core_atom['coords'])

    # Ordenación descendente (Las moléculas periféricas serán las primeras en ser eliminadas)
    atoms.sort(key=lambda a: a['r'], reverse=True)

    return core_atom, atoms


# ==============================================================================
# 4. EVALUACIÓN TERMODINÁMICA GLOBAL
# ==============================================================================

def compute_static_energy(core_atom: Dict, ligands: List[Dict]) -> float:
    """ 
    Calcula la energía de estabilización total del clúster actual
    bajo la aproximación estática de adición por pares. 
    Retorna el valor directamente en meV.
    """
    total_energy = 0.0
    n_ligands = len(ligands)
    core_coords = core_atom['coords']

    ligand_coords = np.array([a['coords'] for a in ligands])

    # Interacción Ion-Disolvente
    for i in range(n_ligands):
        total_energy += v_rb_h2(ligand_coords[i, 0], ligand_coords[i, 1], ligand_coords[i, 2],
                                core_coords[0], core_coords[1], core_coords[2])

        # Interacción Disolvente-Disolvente (Fuerzas de dispersión)
        for j in range(i + 1, n_ligands):
            dist_hh = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
            total_energy += v_h2_h2(dist_hh)

    return total_energy 


# ==============================================================================
# 5. EXPORTACIÓN DE RESULTADOS ESPACIALES
# ==============================================================================

def export_xyz_step(filename: str, core_atom: Dict, ligands: List[Dict], current_n: int, energy_mev: float):
    """ Escribe la topología del paso actual (N) preservando la estructura del clúster interior. """
    with open(filename, 'w') as f:
        f.write(f"{current_n + 1}\n")
        f.write(f"E_total = {energy_mev:.5f} meV | N = {current_n}\n")

        c = core_atom['coords']
        f.write(f"{core_atom['elem']:<4} {c[0]:15.5E} {c[1]:15.5E} {c[2]:15.5E}\n")

        # Invertir la lista para escribirlos en orden normal (de más cercano a más lejano)
        for a in reversed(ligands):
            coord = a['coords']
            f.write(f"{a['elem']:<4} {coord[0]:15.5E} {coord[1]:15.5E} {coord[2]:15.5E}\n")


# ==============================================================================
# 6. ORQUESTACIÓN DEL PROCESO DE DECAPADO (INVERSE OPTIMIZATION)
# ==============================================================================

def main():
    print("===================================================================")
    print(f" Iniciando Optimización Inversa (Decapado): {FILE_INPUT} ")
    print("===================================================================\n")

    try:
        core_atom, ligands = read_and_sort_by_distance(FILE_INPUT)
    except FileNotFoundError:
        print(f"[ERROR] Archivo fuente '{FILE_INPUT}' no localizado en el directorio.")
        return

    n_initial = len(ligands)
    print(f"[INFO] Núcleo Iónico detectado: {core_atom['elem']} | Solvatación máxima: N = {n_initial}")

    energies_log = []

    print("\n  N      Energía Absoluta (meV)      Radio de Extracción (Å)")
    print("-" * 62)

    # Bucle de decapado secuencial (desde N_max hasta N=1)
    while len(ligands) >= 1:
        current_n = len(ligands)

        # 1. Evaluación energética del agregado actual
        current_energy = compute_static_energy(core_atom, ligands)

        # 2. Respaldo topológico
        if SAVE_INTERMEDIATE_XYZ:
            export_xyz_step(f"peeled_N{current_n:02d}.xyz", core_atom, ligands, current_n, current_energy)

        energies_log.append((current_n, current_energy))

        # 3. Decapado: Extracción de la molécula periférica de mayor radio
        peeled_atom = ligands.pop(0)  
        extraction_radius = peeled_atom['r']

        print(f" {current_n:02d} | {current_energy:20.3f} | {extraction_radius:20.3f}")

    # Compilación del registro termodinámico
    with open(FILE_OUT_SUMMARY, 'w') as f:
        f.write("N\tEnergia_Absoluta_meV\n")
        f.write("-" * 28 + "\n")
        for n_val, e_val in energies_log:
            f.write(f"{n_val}\t{e_val:.5f}\n")

    print("-" * 62)
    print(f"[FINALIZADO] Trayectoria termodinámica registrada en '{FILE_OUT_SUMMARY}'")


if __name__ == "__main__":
    main()