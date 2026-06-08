"""
Algoritmo Evolutivo con Restricción Espacial (Capas Fijas).
Optimiza topológicamente la posición de una única partícula móvil (etiquetada 
como 'C' en el .xyz) dentro del pozo de potencial generado por un núcleo de 
moléculas congeladas (H) y un catión central (Rb).
"""

import numpy as np
import os
from typing import List, Tuple

# ==============================================================================
# 1. ARCHIVOS DE ENTRADA Y PARÁMETROS GLOBALES DEL ALGORITMO
# ==============================================================================

# Lote de archivos a procesar (Añadir rutas de los .xyz a optimizar)
FILE_CONFIGS = [
    'opt_C_opt_C_N53_Rb+_H2_1.xyz',
    # 'N52-Rb+_H2_HIDROGENOS.xyz', 
]

# Parámetros del motor evolutivo
P = 50                # Población base
Q_TOURNAMENT = 39     # Tamaño del torneo competitivo
MAX_GEN = 5500        # Límite de generaciones
N_RUNS = 80           # Ciclos de ejecución por archivo
THRESHOLD = 0.0001    # Umbral mínimo de mutación
E_TOLERANCE = 0.001   # Criterio de convergencia energética (meV)
INITIAL_DELTA = 0.5   # Amplitud inicial de mutación gaussiana
INITIAL_STR = 0.35    # Tasa de mutación estructural base
JUMP_DELTA = 0.85     # Factor de reescalado espacial

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
    beta = 7.0     # Parámetro de dureza fijado
    
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
# 3. MÓDULO DE LECTURA, ESCRITURA Y FITNESS RESTRINGIDO
# ==============================================================================

def parse_geometry_from_file(filename: str) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """ 
    Lee el archivo .xyz, aísla el átomo móvil etiquetado como 'C' 
    y congela las coordenadas espaciales del resto del clúster. 
    """
    fixed_atoms = []
    moving_c_pos = np.zeros(3)
    c_found = False
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines[2:]:  # Omitir cabeceras estándar de .xyz
        parts = line.split()
        if len(parts) < 4:
            continue
            
        symb = parts[0]
        coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * ANG_TO_BOHR
        
        if symb == 'C':
            moving_c_pos = coords
            c_found = True
        else:
            fixed_atoms.append((symb, coords))
            
    if not c_found:
        print(f"  [Aviso] Partícula móvil 'C' no detectada en {filename}. Iniciando en el origen (0,0,0).")
        
    return fixed_atoms, moving_c_pos


def export_constrained_xyz(filename: str, fixed_atoms: List[Tuple[str, np.ndarray]], best_c_pos: np.ndarray, energy: float):
    """ Compila y guarda la topología final manteniendo el formato .xyz estándar. """
    num_atoms = len(fixed_atoms) + 1
    
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"Energia Minima Particula Movil = {energy:.5f} meV\n")
        
        for sym, coords in fixed_atoms:
            f.write(f"{sym:2s}  {coords[0]*BOHR_TO_ANG:12.5f}  {coords[1]*BOHR_TO_ANG:12.5f}  {coords[2]*BOHR_TO_ANG:12.5f}\n")
            
        f.write(f"C   {best_c_pos[0]*BOHR_TO_ANG:12.5f}  {best_c_pos[1]*BOHR_TO_ANG:12.5f}  {best_c_pos[2]*BOHR_TO_ANG:12.5f}\n")


def evaluate_mobile_fitness(pos_c: np.ndarray, fixed_atoms: List[Tuple[str, np.ndarray]]) -> float:
    """ Evalúa la energía de interacción exclusiva de la partícula móvil frente al entorno estático. """
    fit_val = 0.0
    for sym, f_pos in fixed_atoms:
        if sym in ['Rb', 'Rb+']:
            fit_val += v_rb_h2(pos_c[0], pos_c[1], pos_c[2], f_pos[0], f_pos[1], f_pos[2])
        else:
            dx = pos_c[0] - f_pos[0]
            dy = pos_c[1] - f_pos[1]
            dz = pos_c[2] - f_pos[2]
            r_dist = np.sqrt(dx**2 + dy**2 + dz**2)
            fit_val += v_h2_h2(r_dist)
            
    return fit_val


# ==============================================================================
# 4. MOTOR DEL ALGORITMO EVOLUTIVO (BÚSQUEDA LOCALIZADA)
# ==============================================================================

def execute_constrained_evolution(initial_seed_x: np.ndarray, fixed_atoms: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, float]:
    """ Despliega el motor de minimización exclusivamente sobre los 3 grados de libertad de la partícula libre. """
    ndim = 3
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(ndim))
    taup = 1.0 / np.sqrt(2.0 * ndim)
    
    p2 = 2 * P
    population = np.zeros((p2, ndim))
    str_arr = np.full(p2, INITIAL_STR)
    fitness = np.zeros(p2)
    delta = INITIAL_DELTA
    
    for i in range(p2):
        population[i] = initial_seed_x + np.random.normal(0, 1, ndim) * delta
        
    old_e = 0.0
    best_x = np.zeros(ndim)
    best_e = float('inf')
    
    for irun in range(1, N_RUNS + 1):
        for i_g in range(1, MAX_GEN + 1):
            
            # Reproducción y Mutación
            for i_parent in range(P):
                i_son = i_parent + P
                g1 = np.random.normal(0, 1)
                g2 = np.random.normal(0, 1, ndim)
                
                population[i_son] = population[i_parent] + str_arr[i_parent] * g2 * delta
                temp_str = str_arr[i_parent] * np.exp(taup * g1 + tau * g2[0])
                str_arr[i_son] = max(temp_str, THRESHOLD)
            
            # Evaluación Termodinámica
            for i in range(p2):
                fitness[i] = evaluate_mobile_fitness(population[i], fixed_atoms)
                
            # Torneo Competitivo
            points = np.zeros(p2, dtype=int)
            for i in range(p2):
                opponents = np.random.choice([idx for idx in range(p2) if idx != i], Q_TOURNAMENT, replace=False)
                for op in opponents:
                    if fitness[i] <= fitness[op]:
                        points[i] += 1
                        
            # Supervivencia
            best_indices = np.argsort(points)[::-1][:P]
            population[:P] = population[best_indices]
            str_arr[:P] = str_arr[best_indices]
            fitness[:P] = fitness[best_indices]
            
        # Validación y Convergencia
        current_e = np.min(fitness[:P]) * HARTREE_TO_MEV
        i_best = np.argmin(fitness[:P])
        
        if current_e < best_e:
            best_e = current_e
            best_x = population[i_best].copy()
            
        if abs(current_e - old_e) < E_TOLERANCE:
            break
            
        old_e = current_e
        
        # Reescalado del entorno de búsqueda
        delta *= JUMP_DELTA
        population[0] = best_x
        for i_ind in range(1, P):
            population[i_ind] = best_x + np.random.normal(0, 1, ndim) * delta
            str_arr[i_ind] = 1.0

    return best_x, best_e


# ==============================================================================
# 5. ORQUESTACIÓN DE LA SIMULACIÓN POR LOTES
# ==============================================================================

def main():
    print("===================================================================")
    print(" Iniciando Optimización Evolutiva por Lotes (Entorno Congelado) ")
    print("===================================================================\n")
    
    for filename in FILE_CONFIGS:
        if not os.path.exists(filename):
            print(f"[ERROR] Archivo '{filename}' no localizado. Omitiendo análisis.")
            continue
            
        print(f"-> Procesando topología base: {filename}")
        
        # 1. Disección de la geometría espacial
        fixed_atoms, initial_c_pos = parse_geometry_from_file(filename)
        print(f"   [INFO] Átomos anclados en el espacio de fases: {len(fixed_atoms)}")
        
        e_ini = evaluate_mobile_fitness(initial_c_pos, fixed_atoms) * HARTREE_TO_MEV
        print(f"   [INFO] Energía inicial (Partícula libre): {e_ini:.5f} meV")
        
        # 2. Despliegue del motor evolutivo
        print("   [INFO] Minimizando pozo de potencial periférico...")
        best_c_pos, final_e = execute_constrained_evolution(initial_c_pos, fixed_atoms)
        
        # 3. Compilación de resultados
        output_name = f"opt_C_{filename}"
        export_constrained_xyz(output_name, fixed_atoms, best_c_pos, final_e)
        
        print(f"   [OK] Optimización finalizada. E_min absoluta = {final_e:.5f} meV")
        print(f"   [OK] Configuración estable exportada a: {output_name}\n")


if __name__ == "__main__":
    main()