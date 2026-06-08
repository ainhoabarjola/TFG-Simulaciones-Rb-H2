"""
Utilidad de Transformación Espacial (Origin Translation).
Lee una topología molecular y aplica un vector de traslación unificado 
a todo el clúster para situar el núcleo de fuerzas (ej. catión Rb+) 
exactamente en el origen del espacio cartesiano (0, 0, 0).
"""

import os
from typing import Optional

# ==============================================================================
# 1. CONFIGURACIÓN DE ARCHIVOS Y PARÁMETROS
# ==============================================================================
FILE_INPUT = "N32 aleatorio.xyz"
FILE_OUTPUT = "N32_aleatorio_centrada.xyz"
TARGET_ELEMENT = "Rb"  # Símbolo del núcleo iónico a centrar


# ==============================================================================
# 2. TRANSFORMACIÓN GEOMÉTRICA
# ==============================================================================

def translate_to_origin(input_file: str, output_file: str, target_elem: str) -> None:
    """
    Identifica el centro de coordenadas deseado y proyecta una traslación global 
    preservando las distancias relativas intermoleculares.
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] No se pudo localizar el archivo de entrada '{input_file}'.")
        return

    with open(input_file, 'r') as f:
        lines = f.readlines()

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        print("[ERROR] El archivo no cumple con el estándar .xyz en su primera línea.")
        return
        
    comment_line = lines[1]
    
    atoms = []
    core_coords: Optional[tuple] = None

    # Extracción de coordenadas y localización del núcleo
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) >= 4:
            element = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            
            atoms.append((element, x, y, z))
            
            # Captura del vector de traslación
            if element == target_elem:
                if core_coords is not None:
                    print(f" [AVISO] Multiplicidad detectada para '{target_elem}'. Se anclará el origen al primer registro.")
                else:
                    core_coords = (x, y, z)

    if core_coords is None:
        print(f"[ERROR] Elemento objetivo '{target_elem}' ausente en la topología.")
        return

    cx, cy, cz = core_coords

    # Exportación aplicando el operador de traslación (T = r - r_core)
    with open(output_file, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write(comment_line)
        
        for element, x, y, z in atoms:
            new_x = x - cx
            new_y = y - cy
            new_z = z - cz
            
            f.write(f"{element:<4} {new_x:15.5E} {new_y:15.5E} {new_z:15.5E}\n")
            
    print(f"[OK] Traslación completada. Origen anclado en '{target_elem}'.")
    print(f" -> Topología exportada a: '{output_file}'")


# ==============================================================================
# 3. EJECUCIÓN
# ==============================================================================

def main():
    print("===================================================================")
    print(" Transformación de Coordenadas (Traslación al Origen)")
    print("===================================================================\n")
    
    translate_to_origin(FILE_INPUT, FILE_OUTPUT, TARGET_ELEMENT)

if __name__ == "__main__":
    main()