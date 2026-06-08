"""
Utilidad de Visualización Topológica (Layer Colorizer).
Analiza topologías moleculares en formato .xyz, segrega las moléculas de disolvente 
en capas de solvatación discretas usando clustering no supervisado (K-Means) 
y exporta un archivo .cml con códigos de color y topología de enlaces intra-capa 
para su renderizado fotorrealista en software de visualización (Avogadro).
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# 1. CONFIGURACIÓN VISUAL Y PARÁMETROS
# ==============================================================================
LAYER_COLORS = [
    "#DAA520",  # Dorado (1ª Capa de Solvatación)
    "#FF0000",  # Rojo (2ª Capa)
    "#00FF00",  # Verde (3ª Capa)
    "#0000FF",  # Azul (4ª Capa)
    "#FFFF00",  # Amarillo (5ª Capa)
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cian
    "#FFA500",  # Naranja
    "#800080",  # Púrpura
    "#40E0D0",  # Turquesa
]


# ==============================================================================
# 2. PROCESAMIENTO GEOMÉTRICO Y CLUSTERING
# ==============================================================================

def load_xyz(filename: str) -> List[Dict]:
    """Carga y parsea el archivo de coordenadas estándar."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("[ERROR] Archivo XYZ incompleto o mal formateado.")

    coordinate_lines = [line.strip() for line in lines[2:] if line.strip()]
    atoms = []
    
    for i, line in enumerate(coordinate_lines):
        parts = line.split()
        if len(parts) == 4:
            atom_info = {
                'id': f"a{i+1}",
                'elementType': parts[0],
                'x': float(parts[1]),
                'y': float(parts[2]),
                'z': float(parts[3]),
            }
            atoms.append(atom_info)
    return atoms


def find_layers_and_bonds(atoms: List[Dict], central_ion_symbol: str, bond_threshold_factor: float = 1.35):
    """
    1. Agrupa el disolvente (H2) en capas concéntricas mediante K-Means sobre r.
    2. Establece la topología de red (enlaces) intra-capa basada en vecindad.
    """
    central_ion = next((a for a in atoms if a['elementType'] == central_ion_symbol), None)
    
    # Tratamiento de los átomos de Hidrógeno (identificados como H en los archivos)
    solvent_atoms = [a for a in atoms if a['elementType'] == 'H'] 

    if not central_ion or not solvent_atoms:
        raise ValueError(f"[ERROR] Ion '{central_ion_symbol}' o disolvente 'H' ausentes en la topología.")

    ion_coord = np.array([central_ion['x'], central_ion['y'], central_ion['z']])
    solv_coords = np.array([[a['x'], a['y'], a['z']] for a in solvent_atoms])
    
    # --- 1. Segregación Radial (K-Means) ---
    distances_to_ion = np.linalg.norm(solv_coords - ion_coord, axis=1)
    x_data = distances_to_ion.reshape(-1, 1)

    sorted_dists = np.sort(distances_to_ion)
    diffs = np.diff(sorted_dists)
    
    layer_jump_threshold = 0.2 
    k_estimated = np.sum(diffs > layer_jump_threshold) + 1
    k_layers = max(1, min(k_estimated, 10)) 

    print(f"   [INFO] Discretizando espacio de solvatación en {k_layers} capas (K-Means)...")
    
    kmeans = KMeans(n_clusters=k_layers, n_init='auto', random_state=42).fit(x_data)
    labels = kmeans.labels_
    
    layer_mean_dists = [(label, np.mean(distances_to_ion[labels == label])) for label in np.unique(labels)]
    layer_mean_dists.sort(key=lambda x: x[1]) 
    
    label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(layer_mean_dists)}
    ordered_labels = np.array([label_mapping[label] for label in labels])
    
    atom_id_to_layer = {solvent_atoms[i]['id']: ordered_labels[i] for i in range(len(solvent_atoms))}
    unique_layers = sorted(np.unique(ordered_labels))
    
    # --- 2. Topología de Red (Enlaces Intra-capa) ---
    bonds = []
    
    for layer_label in unique_layers:
        layer_atoms = [a for a in solvent_atoms if atom_id_to_layer[a['id']] == layer_label]
        num_atoms_in_layer = len(layer_atoms)
        
        if num_atoms_in_layer < 2:
            continue
            
        layer_coords = np.array([[a['x'], a['y'], a['z']] for a in layer_atoms])
        dist_matrix = np.sqrt(np.sum((layer_coords[:, np.newaxis, :] - layer_coords[np.newaxis, :, :]) ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf) 
        
        nn = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(layer_coords)
        distances_to_nn, _ = nn.kneighbors(layer_coords)
        
        mean_nn_dist = np.mean(distances_to_nn[:, 1]) 
        bond_threshold = mean_nn_dist * bond_threshold_factor
        
        for i in range(num_atoms_in_layer):
            for j in range(i + 1, num_atoms_in_layer):
                if dist_matrix[i, j] <= bond_threshold:
                    bonds.append((layer_atoms[i]['id'], layer_atoms[j]['id']))
                    
    return central_ion, solvent_atoms, atom_id_to_layer, unique_layers, bonds


# ==============================================================================
# 3. EXPORTACIÓN A FORMATO AVANZADO (.CML)
# ==============================================================================

def create_cml_file(filename: str, central_ion: Dict, solvent_atoms: List[Dict], 
                    atom_id_to_layer: Dict, unique_layers: List[int], bonds: List[Tuple[str, str]]) -> str:
    """Ensambla el archivo Chemical Markup Language (CML) con soporte de renderizado visual."""
    output_cml = ['<molecule>', '<atomArray>']
    
    layer_color_map = {layer: LAYER_COLORS[i % len(LAYER_COLORS)] 
                       for i, layer in enumerate(unique_layers)}
    
    # Anclaje del núcleo
    output_cml.append(
        f'  <atom id="{central_ion["id"]}" elementType="{central_ion["elementType"]}" '
        f'x3="{central_ion["x"]:.6f}" y3="{central_ion["y"]:.6f}" z3="{central_ion["z"]:.6f}" />'
    )
    
    # Asignación cromática de las moléculas de disolvente
    for atom in solvent_atoms:
        layer_label = atom_id_to_layer[atom['id']]
        color = layer_color_map.get(layer_label, "#FFFFFF")
        
        output_cml.append(
            f'  <atom id="{atom["id"]}" elementType="{atom["elementType"]}" '
            f'x3="{atom["x"]:.6f}" y3="{atom["y"]:.6f}" z3="{atom["z"]:.6f}" '
            f'color="{color}" />'
        )
    
    output_cml.append('</atomArray>')
    
    # Trazado de enlaces
    output_cml.append('<bondArray>')
    for id1, id2 in bonds:
        output_cml.append(f'  <bond atomRefs2="{id1} {id2}" order="1" />')
        
    output_cml.append('</bondArray>')
    output_cml.append('</molecule>')

    output_filename = os.path.splitext(filename)[0] + '_layered.cml'
    with open(output_filename, 'w') as f:
        f.write('\n'.join(output_cml))
        
    print(f"   [OK] Matriz CML ensamblada: {len(unique_layers)} conchas, {len(bonds)} enlaces.")
    print(f"   [OK] Archivo listo para Avogadro: '{output_filename}'")
    
    return output_filename


# ==============================================================================
# 4. ORQUESTACIÓN DEL SCRIPT
# ==============================================================================

def execute_colorizer(filename: str, central_ion_symbol: str):
    """Bucle principal de procesamiento topológico."""
    print(f"\n---> Analizando macroestructura: {filename}")
    try:
        atoms = load_xyz(filename)
        
        central_ion, solvent_atoms, atom_id_to_layer, unique_layers, bonds = find_layers_and_bonds(
            atoms, central_ion_symbol, bond_threshold_factor=1.35 
        )
        
        print("   [INFO] Resumen de empaquetamiento radial:")
        for layer in unique_layers:
            layer_atoms = [a for a in solvent_atoms if atom_id_to_layer[a['id']] == layer]
            count = len(layer_atoms)
            
            layer_coords = np.array([[a['x'], a['y'], a['z']] for a in layer_atoms])
            ion_coord = np.array([central_ion['x'], central_ion['y'], central_ion['z']])
            distances_to_ion = np.linalg.norm(layer_coords - ion_coord, axis=1)
            mean_dist = np.mean(distances_to_ion)
            
            print(f"     * Concha {layer+1}: {count:02d} ligandos | Radio medio: {mean_dist:.4f} Å")
        
        create_cml_file(filename, central_ion, solvent_atoms, atom_id_to_layer, unique_layers, bonds)

    except Exception as e:
        print(f"[ERROR] Fallo crítico durante el procesamiento de {filename}: {e}")

if __name__ == '__main__':
    print("===================================================================")
    print(" Herramienta de Visualización: Generador de Topologías CML ")
    print("===================================================================")
    
    # Define la lista de archivos a procesar
    FILE_CONFIGS = [
        ('N38_Lisa.xyz', 'Rb'),  # Modifica este archivo con la topología que quieras colorear
    ]
    
    for file, ion in FILE_CONFIGS:
        execute_colorizer(file, ion)