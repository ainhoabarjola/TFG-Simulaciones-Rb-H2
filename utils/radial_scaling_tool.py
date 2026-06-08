"""
Herramienta Interactiva de Reescalado Geométrico (Radial Scaling Tool).
Permite la manipulación topológica de arquitecturas moleculares dividiendo 
el sistema en capas de solvatación discretas y aplicando homotecias radiales 
(factores de escala) preservando la simetría angular original del poliedro.
"""

import math
import os
from collections import Counter
from typing import List, Dict, Any

# ==============================================================================
# 1. PARÁMETROS GLOBALES
# ==============================================================================
FILE_INPUT = "N74_escalado_manual.xyz"
FILE_OUTPUT = "N74_escalado_manual_2.xyz"
SHELL_TOLERANCE_ANGSTROMS = 0.5  # Margen radial para agrupar moléculas en una misma capa


# ==============================================================================
# 2. PROCESAMIENTO GEOMÉTRICO Y REESCALADO
# ==============================================================================

def interactive_structure_scaler(input_file: str, output_file: str, tolerance: float = 0.5) -> None:
    """
    Lee un archivo XYZ, segrega el clúster en capas concéntricas (shells) 
    y solicita parámetros de expansión/contracción radial al usuario por consola.
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] No se encontró el archivo de topología: '{input_file}'.")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    
    atoms = []
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) >= 4:
            atoms.append({
                'elem': parts[0],
                'x': float(parts[1]),
                'y': float(parts[2]),
                'z': float(parts[3])
            })
            
    # 1. Identificación del centro de coordenadas (Catión)
    counts = Counter([a['elem'] for a in atoms])
    unique_elements = [e for e, c in counts.items() if c == 1]
    
    if not unique_elements:
        print("[AVISO] Núcleo iónico no detectado. Asumiendo como centro el átomo más cercano al origen (0,0,0).")
        distances_to_origin = [math.hypot(a['x'], a['y'], a['z']) for a in atoms]
        core_index = distances_to_origin.index(min(distances_to_origin))
        core_atom = atoms.pop(core_index)
        core_elem = core_atom['elem']
    else:
        core_elem = unique_elements[0]
        core_atom = next(a for a in atoms if a['elem'] == core_elem)
        atoms = [a for a in atoms if a['elem'] != core_elem]
    
    cx, cy, cz = core_atom['x'], core_atom['y'], core_atom['z']
    
    # 2. Cálculo de vectores unitarios y segregación por capas
    for a in atoms:
        dx = a['x'] - cx
        dy = a['y'] - cy
        dz = a['z'] - cz
        a['original_radius'] = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Conservar el vector direccional para garantizar la simetría angular
        if a['original_radius'] > 0:
            a['u_x'] = dx / a['original_radius']
            a['u_y'] = dy / a['original_radius']
            a['u_z'] = dz / a['original_radius']
        else:
            a['u_x'], a['u_y'], a['u_z'] = 0.0, 0.0, 0.0
            
    atoms.sort(key=lambda a: a['original_radius'])
    
    solvation_shells = []
    current_shell = []
    ref_radius = atoms[0]['original_radius'] if atoms else 0
    
    for a in atoms:
        if abs(a['original_radius'] - ref_radius) <= tolerance:
            current_shell.append(a)
        else:
            solvation_shells.append(current_shell)
            current_shell = [a]
            ref_radius = a['original_radius']
            
    if current_shell:
        solvation_shells.append(current_shell)
        
    print(f"\n=======================================================")
    print(f" ANÁLISIS TOPOLÓGICO: {input_file}")
    print(f"=======================================================")
    print(f" -> Núcleo iónico anclado: {core_elem}")
    print(f" -> Capas de solvatación detectadas: {len(solvation_shells)}\n")
    
    # 3. Interfaz de usuario para reescalado paramétrico
    for i, shell in enumerate(solvation_shells):
        mean_radius = sum(a['original_radius'] for a in shell) / len(shell)
        shell_composition = Counter([a['elem'] for a in shell])
        comp_str = ", ".join([f"{count} {elem}" for elem, count in shell_composition.items()])
        
        print(f" [Capa {i+1}] {len(shell):02d} ligandos [{comp_str}] | Radio medio original: {mean_radius:.3f} Å")
        
        while True:
            try:
                user_input = input(f"   => ¿Nuevo radio de la Capa {i+1} en Å? (Enter para mantener actual): ")
                if user_input.strip() == "":
                    new_radius = mean_radius
                else:
                    new_radius = float(user_input.replace(',', '.'))
                break
            except ValueError:
                print("   [!] Formato numérico inválido. Inténtalo de nuevo.")
                
        # Proyección de las nuevas coordenadas espaciales
        for a in shell:
            a['x_new'] = cx + a['u_x'] * new_radius
            a['y_new'] = cy + a['u_y'] * new_radius
            a['z_new'] = cz + a['u_z'] * new_radius

    # 4. Exportación de la macroestructura ensamblada
    with open(output_file, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(f"Estructura sometida a reescalado radial interactivo ({input_file})\n")
        
        # Núcleo
        f.write(f"{core_atom['elem']:<4} {core_atom['x']:15.5E} {core_atom['y']:15.5E} {core_atom['z']:15.5E}\n")
        
        # Disolvente (ordenado por capas de solvatación)
        for shell in solvation_shells:
            for a in shell:
                f.write(f"{a['elem']:<4} {a['x_new']:15.5E} {a['y_new']:15.5E} {a['z_new']:15.5E}\n")
                
    print(f"\n[OK] Homotecia completada. Macroestructura exportada a '{output_file}'.\n")


# ==============================================================================
# 3. ORQUESTACIÓN DEL SCRIPT
# ==============================================================================

def main():
    interactive_structure_scaler(
        input_file=FILE_INPUT, 
        output_file=FILE_OUTPUT, 
        tolerance=SHELL_TOLERANCE_ANGSTROMS
    )

if __name__ == "__main__":
    main()