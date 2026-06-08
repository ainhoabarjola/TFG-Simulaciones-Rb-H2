"""
Barrido Radial Unidimensional (Lambda Energy Scan).
Ejecuta un barrido de energía potencial escalando topológicamente una capa 
de solvatación periférica (identificada mediante máscara elemental) mediante 
un factor de homotecia continuo (\lambda) respecto a un núcleo congelado.
"""

import numpy as np

# ==============================================================================
# 1. PARÁMETROS GLOBALES Y CONSTANTES
# ==============================================================================
FILE_INPUT = 'N52-Rb+_H2.xyz'
FILE_OUTPUT = 'N76-opt_radial.xyz'
TARGET_MASKS = ['O', 'Cl']  # Máscaras de los elementos periféricos a escalar

BOHR_TO_ANG = 0.5291772
ANG_TO_BOHR = 1.889725989
HARTREE_TO_MEV = 27211.3957
MEV_TO_HARTREE = 3.6749309e-5


# ==============================================================================
# 2. EVALUACIÓN DE LA SUPERFICIE DE ENERGÍA POTENCIAL
# ==============================================================================

def v_rb_h2(x1: float, y1: float, z1: float, x_cs: float, y_cs: float, z_cs: float) -> float:
    m, eps, beta, r_m = 4.0, 50.4594, 7.5, 3.21 
    rm_bohr = r_m * ANG_TO_BOHR
    r = np.sqrt((x_cs - x1)**2 + (y_cs - y1)**2 + (z_cs - z1)**2)
    
    if r == 0: return 0.0 
    
    x = rm_bohr / r
    x4 = x**4
    n = beta + 4.0 * (r / rm_bohr)**2
    
    v_mev = eps * ((m / (n - m)) * (x**n) - (n / (n - m)) * x4)
    return v_mev * MEV_TO_HARTREE

def v_h2_h2(r_bohr: float) -> float:
    m, Rm, eps, beta = 6.0, 3.47, 3.07, 7.0 
    R_ang = r_bohr * BOHR_TO_ANG
    if R_ang == 0: return 0.0
    
    x = R_ang / Rm
    n = beta + 4.0 * (x**2)
    
    v_mev = eps * ((m / (n - m)) * (x**-n) - (n / (n - m)) * (x**-m))
    return v_mev / HARTREE_TO_MEV


# ==============================================================================
# 3. BARRIDO DE ENERGÍA RADIAL
# ==============================================================================

def execute_lambda_scan(input_file: str, output_file: str, target_elements: list):
    fixed_atoms, template_atoms = [], []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symb = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * ANG_TO_BOHR
            
            fixed_atoms.append((symb, coords))
            if symb in target_elements:
                template_atoms.append((symb, coords))
                
    if not template_atoms:
        print("[ERROR] No se localizaron máscaras válidas para escalar.")
        return

    print("===================================================================")
    print(" Exploración de SEP 1D (Barrido de Homotecia \lambda) ")
    print("===================================================================")
    print(f" -> Topología base congelada: {len(fixed_atoms)} partículas.")
    print(f" -> Capa móvil identificada: {len(template_atoms)} partículas {target_elements}.")

    # Dominio de escaneo escalar (1.2 a 1.8 cubre ~6.7 a 10.0 Å)
    lambda_scales = np.linspace(1.2, 1.8, 600)
    
    best_scale = 1.0
    best_energy = float('inf')
    best_shell = []

    print(" -> Desplegando barrido unidimensional del pozo de potencial...\n")

    for scale in lambda_scales:
        current_e_hartree = 0.0
        scaled_shell = [(sym, pos * scale) for sym, pos in template_atoms]
        
        for i, (sym_new, pos_new) in enumerate(scaled_shell):
            # Interacción N-Cuerpos (Nueva capa vs Núcleo)
            for sym_fixed, pos_fixed in fixed_atoms:
                if sym_fixed in ['Rb', 'Rb+']:
                    current_e_hartree += v_rb_h2(pos_new[0], pos_new[1], pos_new[2], pos_fixed[0], pos_fixed[1], pos_fixed[2])
                else:
                    current_e_hartree += v_h2_h2(np.linalg.norm(pos_new - pos_fixed))
                    
            # Interacción Intra-capa (Dispersión London)
            for j in range(i + 1, len(scaled_shell)):
                current_e_hartree += v_h2_h2(np.linalg.norm(pos_new - scaled_shell[j][1]))

        if current_e_hartree < best_energy:
            best_energy = current_e_hartree
            best_scale = scale
            best_shell = scaled_shell

    e_min_mev = best_energy * HARTREE_TO_MEV
    dist_bohr = np.linalg.norm(best_shell[0][1])
    dist_ang = dist_bohr * BOHR_TO_ANG

    print("-" * 67)
    print(f" [OK] Barrido finalizado. Mínimo de energía localizado.")
    print(f"    * E_min absoluta: {e_min_mev:.4f} meV")
    print(f"    * Factor de escala óptimo (\lambda): {best_scale:.5f}")
    print(f"    * Radio proyectado de la nueva capa: ~{dist_ang:.4f} Å")
    print("-" * 67)

    with open(output_file, 'w') as f:
        f.write(f"{len(fixed_atoms) + len(best_shell)}\n")
        f.write(f"E_min capa exterior = {e_min_mev:.4f} meV | Escala = {best_scale:.5f}\n")
        
        for sym, pos in fixed_atoms:
            f.write(f"{sym:<2}  {pos[0]*BOHR_TO_ANG:>12.5f}  {pos[1]*BOHR_TO_ANG:>12.5f}  {pos[2]*BOHR_TO_ANG:>12.5f}\n")
            
        for sym, pos in best_shell:
            f.write(f"{sym:<2}  {pos[0]*BOHR_TO_ANG:>12.5f}  {pos[1]*BOHR_TO_ANG:>12.5f}  {pos[2]*BOHR_TO_ANG:>12.5f}\n")

if __name__ == "__main__":
    execute_lambda_scan(FILE_INPUT, FILE_OUTPUT, TARGET_MASKS)