"""
Algoritmo de Relajación Estructural Global (Unconstrained Optimization).
Somete una topología macroscópica preensamblada a una minimización evolutiva
sin restricciones espaciales ni penalizaciones radiales. Permite la libre 
permutación de partículas para validar la estabilidad termodinámica real del clúster.
"""

import numpy as np
import os
from typing import Tuple

# ==============================================================================
# 1. ARCHIVOS DE ENTRADA Y PARÁMETROS GLOBALES DEL ALGORITMO
# ==============================================================================

FILE_INPUT = 'N74_ELEGIDO.xyz'       # Topología base a auditar
FILE_OUTPUT = 'N74_optimizado.xyz'   # Topología relajada resultante

# Parámetros del motor evolutivo (Ajustados para no destruir la semilla inicial)
P = 50                # Población base
Q_TOURNAMENT = 39     # Tamaño del torneo competitivo
MAX_GEN = 5500        # Límite de generaciones por ciclo
N_RUNS = 80           # Ciclos de ejecución
THRESHOLD = 0.0001    # Umbral mínimo de mutación
E_TOLERANCE = 0.001   # Criterio de convergencia energética (meV)
INITIAL_DELTA = 0.2   # Reducido: Amplitud de mutación inicial (evita explosión topológica)
INITIAL_STR = 0.35    # Tasa de mutación estructural base
JUMP_DELTA = 0.85     # Factor de atenuación geométrica

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
    beta = 7.0
    
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
# 3. MÓDULO DE LECTURA, ESCRITURA Y FITNESS NO RESTRINGIDO
# ==============================================================================

def read_full_topology(filename: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """ 
    Lee la arquitectura molecular completa, extrayendo el catión 
    y aplanando (flatten) la red de disolvente para el algoritmo genético. 
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    x_ca = np.zeros(3)
    solvents = []
    
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            symb = parts[0]
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * ANG_TO_BOHR
            
            if symb in ['Rb', 'Rb+']:
                x_ca = coords
            elif symb == 'H':
                solvents.append(coords)
                
    n_ligands = len(solvents)
    if n_ligands == 0:
        raise ValueError("[ERROR] No se detectaron moléculas de disolvente en el archivo.")

    # Vector 1D de dimensión (3 * N) requerido por el motor evolutivo
    x_seed = np.array(solvents).flatten()
    
    return x_ca, x_seed, n_ligands


def export_relaxed_xyz(filename: str, x_ca: np.ndarray, x_best: np.ndarray, energy: float, current_n: int):
    """ Escribe la configuración termodinámica relajada en formato estándar .xyz. """
    with open(filename, 'w') as f:
        f.write(f"{current_n + 1}\n")
        f.write(f"E_opt_global = {energy:.5e} meV\n")
        f.write(f"Rb  {x_ca[0]*BOHR_TO_ANG:15.5E}  {x_ca[1]*BOHR_TO_ANG:15.5E}  {x_ca[2]*BOHR_TO_ANG:15.5E}\n")
        
        x_reshaped = x_best.reshape((current_n, 3))
        for h in x_reshaped:
            f.write(f"H   {h[0]*BOHR_TO_ANG:15.5E}  {h[1]*BOHR_TO_ANG:15.5E}  {h[2]*BOHR_TO_ANG:15.5E}\n")


def evaluate_unconstrained_fitness(x_ind: np.ndarray, x_ca: np.ndarray, current_n: int) -> float:
    """
    Fitness físico puro. Se omiten heurísticas topológicas para permitir
    la libre exploración y el reordenamiento molecular sin sesgos.
    """
    fit_val = 0.0
    x_reshaped = x_ind.reshape((current_n, 3))
    
    for i in range(current_n):
        fit_val += v_rb_h2(x_reshaped[i,0], x_reshaped[i,1], x_reshaped[i,2], x_ca[0], x_ca[1], x_ca[2])
        for j in range(i + 1, current_n):
            dx = x_reshaped[i,0] - x_reshaped[j,0]
            dy = x_reshaped[i,1] - x_reshaped[j,1]
            dz = x_reshaped[i,2] - x_reshaped[j,2]
            r_hh = np.sqrt(dx**2 + dy**2 + dz**2)
            fit_val += v_h2_h2(r_hh)
                
    return fit_val


# ==============================================================================
# 4. MOTOR DEL ALGORITMO EVOLUTIVO DE BÚSQUEDA GLOBAL
# ==============================================================================

def execute_global_relaxation(current_n: int, initial_seed_x: np.ndarray, x_ca: np.ndarray) -> Tuple[np.ndarray, float]:
    """ Despliega el motor de minimización sobre los 3N grados de libertad del clúster. """
    ndim = 3 * current_n
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(ndim))
    taup = 1.0 / np.sqrt(2.0 * ndim)
    
    p2 = 2 * P
    population = np.zeros((p2, ndim))
    str_arr = np.full(p2, INITIAL_STR)
    fitness = np.zeros(p2)
    delta = INITIAL_DELTA
    
    # Inoculación de ruido térmico inicial
    for i in range(p2):
        population[i] = initial_seed_x + np.random.normal(0, 1, ndim) * delta
        
    old_e = 0.0
    best_overall_x = np.zeros(ndim)
    best_overall_e = float('inf')
    
    for irun in range(1, N_RUNS + 1):
        for i_g in range(1, MAX_GEN + 1):
            
            # Reproducción
            for i_parent in range(P):
                i_son = i_parent + P
                g1 = np.random.normal(0, 1)
                g2 = np.random.normal(0, 1, ndim)
                
                population[i_son] = population[i_parent] + str_arr[i_parent] * g2 * delta
                temp_str = str_arr[i_parent] * np.exp(taup * g1 + tau * g2[0])
                str_arr[i_son] = max(temp_str, THRESHOLD)
            
            # Evaluación
            for i in range(p2):
                fitness[i] = evaluate_unconstrained_fitness(population[i], x_ca, current_n)
                
            # Torneo
            points = np.zeros(p2, dtype=int)
            for i in range(p2):
                opponents = np.random.choice([idx for idx in range(p2) if idx != i], Q_TOURNAMENT, replace=False)
                for op in opponents:
                    if fitness[i] <= fitness[op]:
                        points[i] += 1
                        
            # Selección y supervivencia
            best_indices = np.argsort(points)[::-1][:P]
            population[:P] = population[best_indices]
            str_arr[:P] = str_arr[best_indices]
            fitness[:P] = fitness[best_indices]
            
        # Validación de convergencia del ciclo
        current_e = np.min(fitness[:P]) * HARTREE_TO_MEV
        i_best = np.argmin(fitness[:P])
        
        if current_e < best_overall_e:
            best_overall_e = current_e
            best_overall_x = population[i_best].copy()
            
        print(f"   -> [Run {irun:02d}/{N_RUNS}] Mínimo actual consolidado: {best_overall_e:.3f} meV")
            
        if abs(current_e - old_e) < E_TOLERANCE:
            print("   [INFO] Criterio de convergencia termodinámica alcanzado. Deteniendo minimización.")
            break
            
        old_e = current_e
        
        # Reescalado para escapar de mínimos locales superficiales
        delta *= JUMP_DELTA
        population[0] = best_overall_x
        for i_ind in range(1, P):
            population[i_ind] = best_overall_x + np.random.normal(0, 1, ndim) * delta
            str_arr[i_ind] = 1.0

    return best_overall_x, best_overall_e


# ==============================================================================
# 5. ORQUESTACIÓN PRINCIPAL
# ==============================================================================

def main():
    print("===================================================================")
    print(f" Iniciando Relajación Estructural Global: {FILE_INPUT} ")
    print("===================================================================\n")
    
    try:
        x_ca, x_seed, n_ligands = read_full_topology(FILE_INPUT)
        print(f"[INFO] Topología cargada: Núcleo (1 Rb) solvatado por N = {n_ligands} ligandos libres.")
    except FileNotFoundError:
        print(f"[ERROR] No se localizó el archivo semilla '{FILE_INPUT}'.")
        return

    print("\n[INFO] Desplegando motor evolutivo. Esperando convergencia energética...")
    best_topology, final_energy = execute_global_relaxation(n_ligands, x_seed, x_ca)
    
    export_relaxed_xyz(FILE_OUTPUT, x_ca, best_topology, final_energy, n_ligands)
    
    print("-" * 67)
    print(" [OK] Relajación global finalizada con éxito.")
    print(f" -> Energía del estado fundamental hallado: {final_energy:.5f} meV")
    print(f" -> Topología exportada a: '{FILE_OUTPUT}'\n")


if __name__ == "__main__":
    main()