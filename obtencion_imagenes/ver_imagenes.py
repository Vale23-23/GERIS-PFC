import os
import numpy as np
import matplotlib.pyplot as plt
import random

OUTPUT_ROOT = "dataset_focos_igeos"
BANDA_PRINCIPAL = "ABI-L1b-Rad-B07"
MASCARA = "ABI-L2-FDCF"

def graficar(nombre_f):
    """ Función interna para realizar el plot de una muestra dada """
    ruta_b7 = os.path.join(OUTPUT_ROOT, BANDA_PRINCIPAL, nombre_f)
    ruta_mask = os.path.join(OUTPUT_ROOT, MASCARA, nombre_f)

    if not os.path.exists(ruta_b7):
        print(f"❌ Error: No existe el archivo de radiancia {nombre_f}")
        return

    # Cargar datos
    rad = np.load(ruta_b7)
    mask = np.load(ruta_mask) if os.path.exists(ruta_mask) else None

    # Graficar
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Muestra: {nombre_f}", fontsize=16)

    # Panel 1: Banda 7 (Radiancia)
    im1 = ax[0].imshow(np.log1p(rad), cmap='magma')
    ax[0].set_title("Banda 7 (Log Radiancia - FIRMA TÉRMICA)")
    plt.colorbar(im1, ax=ax[0], label='log(1 + Rad)')

    # Panel 2: Máscara de Fuego (Basada en tu observación de DQF 0-5)
    if mask is not None:
        # Definimos fuego como los valores de calidad 0 a 5
        fuego_binario = np.isin(mask, [0]).astype(int)
        
        im2 = ax[1].imshow(fuego_binario, cmap='Reds')
        ax[1].set_title(f"Máscara de Fuego (Píxeles detectados: {np.sum(fuego_binario)})")
        plt.colorbar(im2, ax=ax[1])
    else:
        ax[1].text(0.5, 0.5, "Máscara no encontrada", ha='center', va='center', fontsize=12)
        ax[1].set_title("Máscara FDC")

    plt.tight_layout()
    plt.show()

def visualizar_aleatorias(num_muestras=3):
    ruta_folder = os.path.join(OUTPUT_ROOT, BANDA_PRINCIPAL)
    archivos = sorted([f for f in os.listdir(ruta_folder) if f.endswith(".npy")])
    
    if not archivos:
        print("La carpeta está vacía.")
        return

    muestras_a_ver = random.sample(archivos, min(num_muestras, len(archivos)))
    for f in muestras_a_ver:
        graficar(f)

if __name__ == "__main__":
    print("--- Visor de Dataset GOES-19 ---")
    print("1. Ver muestras al azar")
    print("2. Elegir una fecha específica")
    
    opcion = input("\nSelecciona una opción (1 o 2): ")

    if opcion == "1":
        n = int(input("¿Cuántas muestras quieres ver?: "))
        visualizar_aleatorias(n)
    
    elif opcion == "2":
        print("\nArchivos disponibles (últimos 10):")
        archivos = sorted([f for f in os.listdir(os.path.join(OUTPUT_ROOT, BANDA_PRINCIPAL)) if f.endswith(".npy")])
        for a in archivos[-10:]:
            print(f" - {a}")
            
        fecha_input = input("\nEscribe el nombre del archivo (ej: 20250901_1200.npy): ")
        if not fecha_input.endswith(".npy"):
            fecha_input += ".npy"
        
        graficar(fecha_input)
    else:
        print("Opción no válida.")