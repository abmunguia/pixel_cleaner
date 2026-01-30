#!/usr/bin/env python3
"""
Grid Colorizer v3 - Con paleta de colores reducida.
Analiza una imagen cuadriculada, pinta cada celda con su color predominante,
y reduce la paleta a N colores.

Uso: python pixel_cleaner.py <imagen_entrada> [imagen_salida] [--colors N]
"""

import sys
import numpy as np
from PIL import Image
import argparse
from scipy import ndimage
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from collections import Counter


def find_grid_spacing(image_array, axis, min_spacing=5, max_spacing=50):
    """
    Encuentra el espaciado de la cuadrícula analizando los patrones de líneas oscuras.
    """
    if len(image_array.shape) == 3:
        gray = np.mean(image_array[:, :, :3], axis=2)
    else:
        gray = image_array
    
    if axis == 0:
        profile = np.mean(gray, axis=1)
    else:
        profile = np.mean(gray, axis=0)
    
    profile = 255 - profile
    prominence = max(10, (np.max(profile) - np.min(profile)) * 0.1)
    peaks, properties = find_peaks(profile, distance=min_spacing, prominence=prominence)
    
    if len(peaks) < 2:
        return None, []
    
    spacings = np.diff(peaks)
    valid_spacings = spacings[(spacings >= min_spacing) & (spacings <= max_spacing)]
    
    if len(valid_spacings) == 0:
        return None, peaks
    
    median_spacing = int(np.median(valid_spacings))
    return median_spacing, peaks


def refine_grid_lines(peaks, expected_spacing, tolerance=0.3):
    """
    Refina las líneas de cuadrícula basándose en el espaciado esperado.
    """
    if len(peaks) < 2 or expected_spacing is None:
        return peaks
    
    refined = [peaks[0]]
    
    for i in range(1, len(peaks)):
        gap = peaks[i] - refined[-1]
        expected_gaps = round(gap / expected_spacing)
        
        if expected_gaps == 0:
            continue
        
        if expected_gaps == 1:
            refined.append(peaks[i])
        else:
            for j in range(1, expected_gaps + 1):
                new_pos = refined[-1] + int(j * expected_spacing)
                if abs(new_pos - peaks[i]) < expected_spacing * tolerance or j < expected_gaps:
                    refined.append(int(refined[-1] + expected_spacing))
            if expected_gaps > 0:
                refined[-1] = peaks[i]
    
    return refined


def get_cell_dominant_color(image_array, x1, y1, x2, y2, border_skip=2, black_threshold=25):
    """
    Obtiene el color dominante de una celda, ignorando los bordes.
    """
    x1_inner = x1 + border_skip
    y1_inner = y1 + border_skip
    x2_inner = x2 - border_skip
    y2_inner = y2 - border_skip
    
    if x2_inner <= x1_inner or y2_inner <= y1_inner:
        return None
    
    cell = image_array[y1_inner:y2_inner, x1_inner:x2_inner]
    
    if cell.size == 0:
        return None
    
    pixels = cell.reshape(-1, cell.shape[-1]) if len(cell.shape) == 3 else cell.reshape(-1, 1)
    
    if len(pixels[0]) >= 3:
        brightness = np.mean(pixels[:, :3], axis=1)
    else:
        brightness = pixels[:, 0]
    
    non_black_mask = brightness > black_threshold
    non_black_pixels = pixels[non_black_mask]
    
    if len(non_black_pixels) == 0:
        return (0, 0, 0)
    
    if len(non_black_pixels) < len(pixels) * 0.1:
        return (0, 0, 0)
    
    avg_color = np.mean(non_black_pixels[:, :3], axis=0).astype(int)
    return tuple(avg_color)


def create_color_palette(colors, n_colors=32):
    """
    Crea una paleta reducida de colores usando K-means clustering.
    """
    # Filtrar colores negros/muy oscuros
    non_black_colors = [c for c in colors if sum(c) > 75]  # threshold para no-negro
    
    if len(non_black_colors) < n_colors:
        n_colors = max(1, len(non_black_colors))
    
    if len(non_black_colors) == 0:
        return [(0, 0, 0)]
    
    colors_array = np.array(non_black_colors)
    
    # Usar K-means para encontrar los colores representativos
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(colors_array)
    
    # Obtener los centros de los clusters como la paleta
    palette = kmeans.cluster_centers_.astype(int)
    
    # Agregar negro a la paleta
    palette = np.vstack([[0, 0, 0], palette])
    
    return [tuple(c) for c in palette]


def find_closest_color(color, palette):
    """
    Encuentra el color más cercano en la paleta usando distancia euclidiana.
    """
    if color == (0, 0, 0) or sum(color) < 75:
        return (0, 0, 0)
    
    color_array = np.array(color)
    min_dist = float('inf')
    closest = palette[0]
    
    for p_color in palette:
        if p_color == (0, 0, 0):
            continue
        dist = np.sqrt(np.sum((color_array - np.array(p_color)) ** 2))
        if dist < min_dist:
            min_dist = dist
            closest = p_color
    
    return closest


def process_grid_image(input_path, output_path=None, cell_size=None, border_width=2, n_colors=32):
    """
    Procesa una imagen cuadriculada y genera una nueva con colores uniformes por celda
    y paleta reducida.
    """
    print(f"Cargando imagen: {input_path}")
    img = Image.open(input_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    print(f"Dimensiones: {width}x{height}")
    
    # Detectar espaciado de cuadrícula
    print("Analizando cuadrícula...")
    
    v_spacing, v_peaks = find_grid_spacing(img_array, axis=1)
    h_spacing, h_peaks = find_grid_spacing(img_array, axis=0)
    
    print(f"Espaciado detectado: {v_spacing}x{h_spacing} px")
    
    if cell_size:
        cell_w, cell_h = cell_size
    else:
        cell_w = v_spacing if v_spacing else 20
        cell_h = h_spacing if h_spacing else 20
    
    print(f"Tamaño de celda: {cell_w}x{cell_h}")
    
    # Refinar las líneas de la cuadrícula
    if len(v_peaks) > 0:
        v_lines = refine_grid_lines(v_peaks, cell_w)
    else:
        v_lines = list(range(0, width, cell_w))
    
    if len(h_peaks) > 0:
        h_lines = refine_grid_lines(h_peaks, cell_h)
    else:
        h_lines = list(range(0, height, cell_h))
    
    if v_lines[0] > cell_w // 2:
        v_lines = [0] + v_lines
    if v_lines[-1] < width - cell_w // 2:
        v_lines.append(width)
    
    if h_lines[0] > cell_h // 2:
        h_lines = [0] + h_lines
    if h_lines[-1] < height - cell_h // 2:
        h_lines.append(height)
    
    print(f"Cuadrícula: {len(v_lines)-1} columnas x {len(h_lines)-1} filas")
    
    # Primera pasada: obtener todos los colores dominantes
    print("Extrayendo colores de cada celda...")
    cell_colors = {}
    all_colors = []
    
    for i in range(len(h_lines) - 1):
        y1 = h_lines[i]
        y2 = h_lines[i + 1]
        
        for j in range(len(v_lines) - 1):
            x1 = v_lines[j]
            x2 = v_lines[j + 1]
            
            dominant_color = get_cell_dominant_color(img_array, x1, y1, x2, y2, border_skip=border_width)
            
            if dominant_color:
                cell_colors[(i, j)] = dominant_color
                if sum(dominant_color) > 75:  # No incluir negros para la paleta
                    all_colors.append(dominant_color)
    
    print(f"Colores únicos extraídos: {len(set(all_colors))}")
    
    # Crear paleta reducida
    print(f"Creando paleta de {n_colors} colores...")
    palette = create_color_palette(all_colors, n_colors)
    print(f"Paleta creada con {len(palette)} colores")
    
    # Mostrar la paleta
    print("\nPaleta de colores (RGB):")
    for i, color in enumerate(palette[:10]):  # Mostrar primeros 10
        print(f"  {i+1}. RGB{color}")
    if len(palette) > 10:
        print(f"  ... y {len(palette)-10} colores más")
    
    # Segunda pasada: mapear colores a la paleta y pintar
    print("\nAplicando paleta reducida...")
    output_array = np.zeros_like(img_array)
    
    color_mapping = {}  # Cache para evitar recalcular
    
    for (i, j), original_color in cell_colors.items():
        y1 = h_lines[i]
        y2 = h_lines[i + 1]
        x1 = v_lines[j]
        x2 = v_lines[j + 1]
        
        # Buscar en cache o calcular
        if original_color not in color_mapping:
            color_mapping[original_color] = find_closest_color(original_color, palette)
        
        quantized_color = color_mapping[original_color]
        
        # Pintar el interior de la celda
        inner_x1 = x1 + border_width
        inner_y1 = y1 + border_width
        inner_x2 = x2 - border_width
        inner_y2 = y2 - border_width
        
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            output_array[inner_y1:inner_y2, inner_x1:inner_x2] = quantized_color
    
    # Dibujar las líneas de la cuadrícula
    print("Dibujando líneas de cuadrícula...")
    
    for y in h_lines:
        y_start = max(0, y - border_width // 2)
        y_end = min(height, y + border_width // 2 + border_width % 2)
        output_array[y_start:y_end, :] = [0, 0, 0]
    
    for x in v_lines:
        x_start = max(0, x - border_width // 2)
        x_end = min(width, x + border_width // 2 + border_width % 2)
        output_array[:, x_start:x_end] = [0, 0, 0]
    
    # Guardar imagen
    output_img = Image.fromarray(output_array)
    
    if output_path is None:
        if '.' in input_path:
            base, ext = input_path.rsplit('.', 1)
            output_path = f"{base}_colorized_{n_colors}colors.{ext}"
        else:
            output_path = f"{input_path}_colorized_{n_colors}colors.png"
    
    output_img.save(output_path, quality=95)
    print(f"\nImagen guardada: {output_path}")
    
    # Contar colores finales usados
    unique_colors = len(set(color_mapping.values()))
    print(f"Colores únicos en resultado: {unique_colors}")
    
    return output_path, palette


def main():
    parser = argparse.ArgumentParser(
        description='Procesa una imagen cuadriculada con paleta de colores reducida.'
    )
    parser.add_argument('input', help='Ruta de la imagen de entrada')
    parser.add_argument('output', nargs='?', help='Ruta de la imagen de salida (opcional)')
    parser.add_argument('--colors', type=int, default=7, help='Número de colores en la paleta (default: 32)')
    parser.add_argument('--cell-width', type=int, help='Ancho de celda (si se conoce)')
    parser.add_argument('--cell-height', type=int, help='Alto de celda (si se conoce)')
    parser.add_argument('--border', type=int, default=2, help='Ancho del borde (default: 2)')
    
    args = parser.parse_args()
    
    cell_size = None
    if args.cell_width and args.cell_height:
        cell_size = (args.cell_width, args.cell_height)
    
    try:
        output_path, palette = process_grid_image(
            args.input, 
            args.output, 
            cell_size=cell_size,
            border_width=args.border,
            n_colors=args.colors
        )
        print(f"\n¡Proceso completado exitosamente!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
