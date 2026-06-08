"""
Herramienta de Proyección Esférica y Homotecia (Spherical Projection Tool).
Permite forzar geometrías mediante el uso de 'máscaras de selección' 
(cambiando temporalmente el símbolo del elemento en el .xyz para aislar una capa).
Aplica transformaciones radiales respecto al núcleo iónico.
"""

import numpy as np
import os
from typing import Tuple, List

def load_xyz(filename: str) -> Tuple[int, str, np.ndarray, np.ndarray]:
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    num_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    symbols, coords = [], []
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            symbols.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            
    return num_atoms, comment, np.array(symbols), np.array(coords)

def export_xyz(filename: str, num_atoms: int, comment: str, symbols: np.ndarray, coords: np.ndarray):
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"{comment} - Transformacion Geometrica Aplicada\n")
        for s, c in zip(symbols, coords):
            f.write(f"{s:<4} {c[0]:14.5f} {c[1]:14.5f} {c[2]:14.5f}\n")

def main():
    print("===================================================================")
    print(" Transformación Geométrica: Homotecia y Proyección Esférica ")
    print("===================================================================\n")
    
    input_file = 'N38 - elementos.xyz'
    
    try:
        num_atoms, comment, symbols, coords = load_xyz(input_file)
        print(f"[OK] Topología cargada: {num_atoms} partículas.")
    except FileNotFoundError:
        print(f"[ERROR] No se encuentra el archivo '{input_file}'.")
        return

    # Localizar el núcleo de coordenadas (Catión Rb)
    idx_core = np.where((symbols == 'Rb') | (symbols == 'Rb+'))[0]
    if len(idx_core) == 0:
        print("[ERROR] Ausencia de núcleo iónico central (Rb) en la topología.")
        return
    
    core_origin = coords[idx_core[0]]
    modified_elements = []

    while True:
        print("-" * 67)
        target_elem = input("-> Símbolo de la máscara a modificar (o 'salir' para terminar): ").strip()
        
        if target_elem.lower() == 'salir':
            break
            
        idx_elem = np.where(symbols == target_elem)[0]
        if len(idx_elem) == 0:
            print(f"   [AVISO] Máscara '{target_elem}' no detectada.")
            continue

        elem_coords = coords[idx_elem]
        rel_vectors = elem_coords - core_origin
        distances = np.linalg.norm(rel_vectors, axis=1)
        current_mean_dist = np.mean(distances)
        
        print(f"   [INFO] {len(idx_elem)} partículas aisladas mediante la máscara '{target_elem}'.")
        print(f"   [INFO] Radio radial actual: Mín={np.min(distances):.4f} | Máx={np.max(distances):.4f} | Media={current_mean_dist:.4f} Å")
        
        try:
            new_distance = float(input(f"   -> Introduce el radio objetivo (Å): "))
        except ValueError:
            print("   [ERROR] Formato numérico inválido.")
            continue

        print("\n   [MODO DE TRANSFORMACIÓN]")
        print("   1. HOMOTECIA: Preserva simetría angular (escala distancia media).")
        print("   2. PROYECCIÓN ESFÉRICA: Fuerza el colapso radial exacto (pierde corrugación original).")
        
        mode = input("   -> Selección (1 o 2): ").strip()
        
        if mode == '1':
            scale_factor = new_distance / current_mean_dist
            coords[idx_elem] = core_origin + scale_factor * rel_vectors
            print(f"   [OK] Homotecia ejecutada. Factor de escala aplicado: \lambda = {scale_factor:.4f}")
            modified_elements.append(target_elem)
            
        elif mode == '2':
            unit_vectors = rel_vectors / distances[:, np.newaxis]
            coords[idx_elem] = core_origin + new_distance * unit_vectors
            print(f"   [OK] Proyección esférica ejecutada. Radio forzado a {new_distance} Å.")
            modified_elements.append(target_elem)
        else:
            print("   [ERROR] Modo no reconocido.")

    print("-" * 67)
    if modified_elements:
        unique_elems = list(set(modified_elements))
        elem_str = "_".join(unique_elems)
        output_file = f"{input_file.split('.')[0]}_modificado_{elem_str}.xyz"
        
        export_xyz(output_file, num_atoms, comment, symbols, coords)
        print(f"[FINALIZADO] Topología exportada a: '{output_file}'")
    else:
        print("Operación cancelada sin aplicar cambios.")

if __name__ == "__main__":
    main()