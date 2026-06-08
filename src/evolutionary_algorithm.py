"""
Algoritmo Evolutivo para la optimización estructural de clústeres Rb+(H2)_N.
Implementa el potencial analítico Improved Lennard-Jones (ILJ) y un motor
de búsqueda global continuo para la determinación de los números mágicos.
"""

import numpy as np
import os
from typing import Tuple

# ==============================================================================
# 1. PARÁMETROS GLOBALES DEL ALGORITMO EVOLUTIVO
# ==============================================================================
P = 50                # Tamaño de la población base
N_MAX = 70            # Tamaño máximo de la capa de solvatación a explorar
Q_TOURNAMENT = 39     # Tamaño del torneo de selección competitiva
MAX_GEN = 5500        # Límite máximo de generaciones por ciclo de ejecución
N_RUNS = 80           # Número de ciclos de optimización por cada N
THRESHOLD = 0.0001    # Umbral mínimo de mutación
E_TOLERANCE = 0.001   # Criterio de convergencia energética (meV)
INITIAL_DELTA = 0.5   # Amplitud inicial de la mutación gaussiana
INITIAL_STR = 0.35    # Tasa de mutación estructural base
JUMP_DELTA = 0.85     # Factor de atenuación para el reescalado del espacio de búsqueda

FILE_INIT = 'Rb+_H2_1.xyz'         # Semilla topológica inicial
FILE_OUT = 'energias_N.txt'        # Registro termodinámico de salida

# Constantes de conversión
BOHR_TO_ANG = 0.5291772
ANG_TO_BOHR = 1.889725989
HARTREE_TO_MEV = 27211.3957
MEV_TO_HARTREE = 3.6749309e-5


# ==============================================================================
# 2. EVALUACIÓN DE LA SUPERFICIE DE ENERGÍA POTENCIAL (SEP)
# ==============================================================================

def v_rb_h2(x1: float, y1: float, z1: float, x_cs: float, y_cs: float, z_cs: float) -> float:
    """
    Evalúa la interacción Soluto-Disolvente (Rb+ - H2) mediante el modelo ILJ.
    Parámetros adaptados para la aproximación de pseudo-átomo (J=0).
    """
    m = 4.0
    eps = 50.4594  # Profundidad del pozo (meV)
    beta = 7.5
    r_m = 3.21     # Distancia de equilibrio (Angstroms)
    
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
    """
    Evalúa la interacción Disolvente-Disolvente (H2 - H2) mediante el modelo ILJ.
    Gobernado fundamentalmente por las fuerzas de dispersión de London.
    """
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
# 3. MÓDULO DE GESTIÓN TOPOLÓGICA Y FUNCIÓN DE COSTO
# ==============================================================================

def read_initial_topology(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Lee las coordenadas del archivo .xyz base para iniciar la secuencia en N=1. 
    Retorna los vectores de posición en unidades atómicas (Bohr).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    x_ca = np.zeros(3)
    x_h2 = np.zeros(3)
    
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
            
        symb = parts[0]
        coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) * ANG_TO_BOHR
        
        if symb in ['Rb', 'Rb+']:
            x_ca = coords
        elif symb == 'H':
            x_h2 = coords
            break  # Aislar únicamente el primer pseudo-átomo para N=1
            
    return x_ca, x_h2


def export_xyz(filename: str, x_ca: np.ndarray, x_best: np.ndarray, energy: float, current_n: int):
    """ Escribe la configuración de equilibrio termodinámico en formato estándar .xyz """
    with open(filename, 'w') as f:
        f.write(f"{current_n + 1}\n")
        f.write(f"E = {energy:.5e} meV\n")
        f.write(f"Rb  {x_ca[0]*BOHR_TO_ANG:.5e}  {x_ca[1]*BOHR_TO_ANG:.5e}  {x_ca[2]*BOHR_TO_ANG:.5e}\n")
        
        x_reshaped = x_best.reshape((current_n, 3))
        for h in x_reshaped:
            f.write(f"H   {h[0]*BOHR_TO_ANG:.5e}  {h[1]*BOHR_TO_ANG:.5e}  {h[2]*BOHR_TO_ANG:.5e}\n")


def evaluate_cluster_fitness(x_ind: np.ndarray, x_ca: np.ndarray, current_n: int) -> float:
    """ 
    Calcula la energía potencial total de la configuración estática bajo la 
    aproximación de adición por pares e impone el control estructural radial.
    """
    fit_val = 0.0
    x_reshaped = x_ind.reshape((current_n, 3))
    
    # 1. Evaluación Termodinámica (Interacciones de dos cuerpos)
    for i in range(current_n):
        fit_val += v_rb_h2(x_reshaped[i,0], x_reshaped[i,1], x_reshaped[i,2], x_ca[0], x_ca[1], x_ca[2])
        for j in range(i + 1, current_n):
            dx = x_reshaped[i,0] - x_reshaped[j,0]
            dy = x_reshaped[i,1] - x_reshaped[j,1]
            dz = x_reshaped[i,2] - x_reshaped[j,2]
            r_hh = np.sqrt(dx**2 + dy**2 + dz**2)
            fit_val += v_h2_h2(r_hh)
            
    # 2. Control Topológico (Restricción de ordenación de capas)
    if current_n > 1:
        dists = np.linalg.norm(x_reshaped - x_ca, axis=1)
        for i in range(current_n - 1):
            if dists[i+1] < dists[i]:
                # Penalización heurística para garantizar el crecimiento esférico secuencial
                fit_val += 0.05 * (dists[i] - dists[i+1]) 
                
    return fit_val


# ==============================================================================
# 4. NÚCLEO DEL ALGORITMO EVOLUTIVO DE BÚSQUEDA GLOBAL
# ==============================================================================

def execute_evolutionary_search(current_n: int, initial_seed_x: np.ndarray, x_ca: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Despliega el motor de minimización para determinar el estado fundamental
    del espacio de fases de dimensiones (3 * N).
    """
    ndim = 3 * current_n
    tau = 1.0 / np.sqrt(2.0 * np.sqrt(ndim))
    taup = 1.0 / np.sqrt(2.0 * ndim)
    
    p2 = 2 * P
    population = np.zeros((p2, ndim))
    str_arr = np.full(p2, INITIAL_STR)
    fitness = np.zeros(p2)
    delta = INITIAL_DELTA
    
    # Inoculación de ruido gaussiano sobre la arquitectura base
    for i in range(p2):
        population[i] = initial_seed_x + np.random.normal(0, 1, ndim) * delta
        
    old_e = 0.0
    best_x = np.zeros(ndim)
    best_e = float('inf')
    
    for irun in range(1, N_RUNS + 1):
        for ig in range(1, MAX_GEN + 1):
            # Fase 1: Mutación y Reproducción
            for i_parent in range(P):
                i_son = i_parent + P
                g1 = np.random.normal(0, 1)
                g2 = np.random.normal(0, 1, ndim)
                
                population[i_son] = population[i_parent] + str_arr[i_parent] * g2 * delta
                temp_str = str_arr[i_parent] * np.exp(taup * g1 + tau * g2[0])
                str_arr[i_son] = max(temp_str, THRESHOLD)
            
            # Fase 2: Evaluación Termodinámica
            for i in range(p2):
                fitness[i] = evaluate_cluster_fitness(population[i], x_ca, current_n)
                
            # Fase 3: Selección Competitiva (Torneo)
            points = np.zeros(p2, dtype=int)
            for i in range(p2):
                opponents = np.random.choice([idx for idx in range(p2) if idx != i], Q_TOURNAMENT, replace=False)
                for op in opponents:
                    if fitness[i] <= fitness[op]:
                        points[i] += 1
                        
            # Fase 4: Supervivencia
            best_indices = np.argsort(points)[::-1][:P]
            population[:P] = population[best_indices]
            str_arr[:P] = str_arr[best_indices]
            fitness[:P] = fitness[best_indices]
            
        # Validación del ciclo de ejecución
        current_min_e = np.min(fitness[:P]) * HARTREE_TO_MEV
        i_best = np.argmin(fitness[:P])
        
        if current_min_e < best_e:
            best_e = current_min_e
            best_x = population[i_best].copy()
            
        if abs(current_min_e - old_e) < E_TOLERANCE:
            break
            
        old_e = current_min_e
        
        # Reescalado heurístico para el siguiente bloque de generaciones
        delta *= JUMP_DELTA
        population[0] = best_x
        for i_ind in range(1, P):
            population[i_ind] = best_x + np.random.normal(0, 1, ndim) * delta
            str_arr[i_ind] = 1.0

    return best_x, best_e


# ==============================================================================
# 5. ORQUESTACIÓN DE LA SIMULACIÓN
# ==============================================================================

def main():
    print("===================================================================")
    print(" Iniciando Simulación Cuántica: Crecimiento Secuencial Rb+(H2)_N ")
    print("===================================================================\n")
    
    try:
        x_ca, x_h2_1 = read_initial_topology(FILE_INIT)
    except FileNotFoundError:
        print(f"ERROR: Topología base '{FILE_INIT}' no localizada en el directorio.")
        return

    energies_log = []
    current_seed = x_h2_1.copy()

    for n_cluster in range(1, N_MAX + 1):
        print(f"-> Ensamblando y optimizando capa de solvatación N = {n_cluster}")
        
        best_topology, final_energy = execute_evolutionary_search(n_cluster, current_seed, x_ca)
        
        filename = f"opt_Rb_H2_{n_cluster}.xyz"
        export_xyz(filename, x_ca, best_topology, final_energy, n_cluster)
        print(f"   [OK] Topología extraída: {filename} | E_min: {final_energy:.5f} meV")
        
        energies_log.append((n_cluster, final_energy))
        
        # Proyección espacial heurística para inaugurar la siguiente configuración (N+1)
        if n_cluster < N_MAX:
            old_coords = best_topology.reshape((n_cluster, 3))
            dists = np.linalg.norm(old_coords - x_ca, axis=1)
            
            # Ubicación de la nueva molécula sonda en la periferia del pozo de potencial
            r_new = np.max(dists) + 2.0 
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            new_hx = x_ca[0] + r_new * np.sin(theta) * np.cos(phi)
            new_hy = x_ca[1] + r_new * np.sin(theta) * np.sin(phi)
            new_hz = x_ca[2] + r_new * np.cos(theta)
            
            current_seed = np.append(best_topology, [new_hx, new_hy, new_hz])

    # Compilación final del registro termodinámico
    with open(FILE_OUT, 'w') as f:
        f.write("N\tEnergia_Absoluta(meV)\n")
        f.write("-" * 28 + "\n")
        for n_val, e_val in energies_log:
            f.write(f"{n_val}\t{e_val:.5f}\n")
            
    print(f"\n[FINALIZADO] Trayectoria termodinámica registrada en '{FILE_OUT}'")


if __name__ == "__main__":
    main()