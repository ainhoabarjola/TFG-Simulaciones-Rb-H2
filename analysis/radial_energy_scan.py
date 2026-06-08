"""
Optimizador de Barrido Radial Unidimensional (Radial Energy Scan).
Segmenta topológicamente un agregado cuántico en capas de solvatación,
congela el núcleo interior y ejecuta un escaneo determinista para localizar
el mínimo termodinámico del radio de la macroestructura exterior.
"""

import numpy as np
import os
from collections import Counter
from typing import List, Dict, Tuple, Any

# ==============================================================================
# 1. ARCHIVOS DE ENTRADA Y PARÁMETROS GLOBALES DEL ALGORITMO
# ==============================================================================

FILE_IN = 'N74_escalado_manual_2.xyz'     # Topología inicial a evaluar
FILE_OUT = 'N74_Capa_Externa_Optimizada_2.xyz'   # Topología relajada de salida

# Criterio topológico para discriminar capas concéntricas (en Radios de Bohr)
SHELL_TOLERANCE_BOHR = 1.0  

# Constantes de conversión física
BOHR_TO_ANG = 0.5291772
ANG_TO_BOHR = 1.889725989
HARTREE_TO_MEV = 27211.3957
MEV_TO_HARTREE = 3.6749309e-5


# ==============================================================================
# 2. EVALUACIÓN DE LA SUPERFICIE DE ENERGÍA POTENCIAL (SEP)
# ==============================================================================

def v_rb_h2(x1: float, y1: float, z1: float, x_cs: float, y_cs: float, z_cs: float) -> float:
    """ Interacción Soluto-Disolvente (Rb+ - H2) mediante modelo ILJ. """
    m = 4.0
    eps = 50.4594  # meV
    beta = 7.5
    r_m = 3.21     # Angstroms
    
    rm_bohr = r_m * ANG_TO_BOHR
    r = np.sqrt((x_cs - x1)**2 + (y_cs - y1)**2 + (z_cs - z1)**2)
    
    if r == 0:
        return 0.0 
    
    x = rm_bohr / r
    x2_inv = (r / rm_bohr)**2
    x4 = x**4
    n = beta + 4.0 * x2_inv
    
    term1 = (m / (n - m)) * (x**n)
    term2 = (n / (n - m)) * x4
    
    v_mev = eps * (term1 - term2)
    return v_mev * MEV_TO_HARTREE


def v_h2_h2(r_bohr: float) -> float:
    """ Interacción Disolvente-Disolvente (H2 - H2) mediante modelo ILJ. """
    m = 6.0
    rm_ang = 3.47  # Angstroms
    eps = 3.07     # meV
    beta = 8.0     # Dureza de la pared repulsiva fijada en 8.0
    
    r_ang = r_bohr * BOHR_TO_ANG
    if r_ang == 0:
        return 0.0
    
    x = r_ang / rm_ang
    n = beta + 4.0 * (x**2)
    n_minus_m = n - m
    
    term_a = m / n_minus_m
    term_b = x**(-n)
    term_c = n / n_minus_m
    term_d = x**(-m)
    
    v_mev = eps * (term_a * term_b - term_c * term_d)
    return v_mev / HARTREE_TO_MEV


# ==============================================================================
# 3. ANÁLISIS TOPOLÓGICO Y SEGREGACIÓN DE CAPAS
# ==============================================================================

def prepare_system(filename: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], float, str]:
    """
    Parsea la geometría, detecta el núcleo catiónico y segrega las moléculas 
    de disolvente en subestructuras (capas) en función de su distancia radial.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    comment_line = lines[1].strip()
    
    atoms = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symb = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * ANG_TO_BOHR
            atoms.append({'elem': symb, 'coords': coords})
            
    # Detección heurística del núcleo (el elemento de multiplicidad 1)
    counts = Counter([a['elem'] for a in atoms])
    unique_elements = [e for e, c in counts.items() if c == 1]
    
    if not unique_elements:
        raise ValueError("[ERROR] Multiplicidad central no definida. No se detectó un ion único.")
        
    center_elem = unique_elements[0]
    idx_center = next(i for i, a in enumerate(atoms) if a['elem'] == center_elem)
    center_atom = atoms.pop(idx_center)
    center_coord = center_atom['coords']
    
    # Cálculo de métricas radiales para preservar simetría direccional
    for a in atoms:
        vector = a['coords'] - center_coord
        r = np.linalg.norm(vector)
        a['r'] = r
        a['u'] = vector / r if r > 0 else np.zeros(3)
        
    atoms.sort(key=lambda a: a['r'])
    
    # Algoritmo de segregación por clústers radiales
    shells = []
    current_shell = []
    ref_radius = atoms[0]['r'] if atoms else 0
    
    for a in atoms:
        if abs(a['r'] - ref_radius) <= SHELL_TOLERANCE_BOHR:
            current_shell.append(a)
        else:
            shells.append(current_shell)
            current_shell = [a]
            ref_radius = a['r']
            
    if current_shell:
        shells.append(current_shell)
        
    print(f"   [INFO] Se han segregado {len(shells)} macroestructuras concéntricas.")
    
    # Aislamiento termodinámico: núcleo congelado vs envoltura móvil
    outer_shell = shells[-1]
    inner_shells = [a for shell in shells[:-1] for a in shell]
    
    # Determinación de la frontera de colapso repulsivo (pared dura)
    if len(shells) > 1:
        penultimate_radius = max(a['r'] for a in shells[-2])
        r_min = penultimate_radius + 0.1  # Margen de seguridad estricto
    else:
        r_min = 1.0  
        
    return center_atom, inner_shells, outer_shell, r_min, comment_line


# ==============================================================================
# 4. MOTOR UNIDIMENSIONAL DE EVALUACIÓN Y BARRIDO
# ==============================================================================

def compute_total_energy(r_test: float, center_atom: Dict, inner_shells: List[Dict], outer_shell: List[Dict]) -> float:
    """ Computa la interacción de n-cuerpos bajo la aproximación par a par. """
    n_total = len(inner_shells) + len(outer_shell)
    all_coords = np.zeros((n_total, 3))
    
    for i, a in enumerate(inner_shells):
        all_coords[i] = a['coords']
        
    offset = len(inner_shells)
    for i, a in enumerate(outer_shell):
        all_coords[offset + i] = center_atom['coords'] + a['u'] * r_test
        
    energy = 0.0
    c_ca = center_atom['coords']
    
    for i in range(n_total):
        energy += v_rb_h2(all_coords[i,0], all_coords[i,1], all_coords[i,2], c_ca[0], c_ca[1], c_ca[2])
        for j in range(i + 1, n_total):
            dist = np.linalg.norm(all_coords[i] - all_coords[j])
            energy += v_h2_h2(dist)
            
    return energy


def execute_radial_sweep(center_atom: Dict, inner_shells: List[Dict], outer_shell: List[Dict], r_min: float) -> Tuple[float, float]:
    """ 
    Estrategia de minimización híbrida: Barrido paramétrico grueso (Grid Search) 
    seguido de un refinamiento localizado en el mínimo del pozo.
    """
    # 1. Búsqueda Gruesa (Localización de la cuenca de atracción)
    coarse_radii = np.linspace(r_min, r_min + 15.0, 500)
    coarse_energies = []
    
    for r in coarse_radii:
        e = compute_total_energy(r, center_atom, inner_shells, outer_shell)
        coarse_energies.append(e * HARTREE_TO_MEV)
        
    idx_min = np.argmin(coarse_energies)
    best_coarse_r = coarse_radii[idx_min]
    
    # 2. Refinamiento Fino (Zoom in analítico)
    fine_radii = np.linspace(best_coarse_r - 0.2, best_coarse_r + 0.2, 500)
    fine_radii = fine_radii[fine_radii >= r_min] 
    
    best_r = best_coarse_r
    best_e = coarse_energies[idx_min]
    
    for r in fine_radii:
        e = compute_total_energy(r, center_atom, inner_shells, outer_shell) * HARTREE_TO_MEV
        if e < best_e:
            best_e = e
            best_r = r
            
    return best_r, best_e


def export_optimized_topology(filename: str, center_atom: Dict, inner_shells: List[Dict], outer_shell: List[Dict], r_opt: float, e_opt: float, comment: str):
    """ Restituye la simetría final y compila la estructura en un estándar .xyz. """
    num_atoms = 1 + len(inner_shells) + len(outer_shell)
    
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"{comment} | E_min_absoluta = {e_opt:.3f} meV | R_externo_optimo = {r_opt * BOHR_TO_ANG:.3f} Angstroms\n")
        
        c = center_atom['coords'] * BOHR_TO_ANG
        f.write(f"{center_atom['elem']:<4} {c[0]:15.5E} {c[1]:15.5E} {c[2]:15.5E}\n")
        
        for a in inner_shells:
            coord = a['coords'] * BOHR_TO_ANG
            f.write(f"{a['elem']:<4} {coord[0]:15.5E} {coord[1]:15.5E} {coord[2]:15.5E}\n")
            
        c_bohr = center_atom['coords']
        for a in outer_shell:
            coord_ang = (c_bohr + a['u'] * r_opt) * BOHR_TO_ANG
            f.write(f"{a['elem']:<4} {coord_ang[0]:15.5E} {coord_ang[1]:15.5E} {coord_ang[2]:15.5E}\n")


# ==============================================================================
# 5. ORQUESTACIÓN PRINCIPAL
# ==============================================================================

def main():
    print("===================================================================")
    print("       Desplegando Optimizador de Barrido Radial 1D                ")
    print("===================================================================\n")
    
    try:
        center_atom, inner_shells, outer_shell, r_min, comment = prepare_system(FILE_IN)
    except FileNotFoundError:
        print(f"[ERROR] Topología de entrada '{FILE_IN}' no localizada en el path.")
        return
        
    print(f"   [ESTADO] Ion central parametrizado: {center_atom['elem']}")
    print(f"   [ESTADO] Moléculas ancladas en la matriz rígida: {len(inner_shells)}")
    print(f"   [ESTADO] Moléculas proyectadas en la capa libre: {len(outer_shell)}")
    
    print(f"   [MÉTRICA] Límite de colapso repulsivo: > {r_min * BOHR_TO_ANG:.3f} Å")
    print("\n---> Iniciando escaneo determinista del pozo de potencial...")
    
    best_r_bohr, best_e = execute_radial_sweep(center_atom, inner_shells, outer_shell, r_min)
    
    export_optimized_topology(FILE_OUT, center_atom, inner_shells, outer_shell, best_r_bohr, best_e, comment)
    
    print("-" * 65)
    print("   [ÉXITO] Relajación estructural consolidada.")
    print(f"   [RESULTADO] Distancia radial óptima de la corteza: {best_r_bohr * BOHR_TO_ANG:.3f} Å")
    print(f"   [RESULTADO] Energía fundamental del macroagregado: {best_e:.3f} meV")
    print(f"   [SISTEMA]   Coordenadas volcadas en: '{FILE_OUT}'\n")

if __name__ == "__main__":
    main()