import os
import numpy as np
import pandas as pd

OUTPUT_ROOT = "dataset_focos_igeos"
MASCARA_DIR = "ABI-L2-FDCF" # Nombre de la carpeta de las etiquetas

def auditar_dataset_con_fuego():
    if not os.path.exists(OUTPUT_ROOT):
        print(f"❌ La carpeta '{OUTPUT_ROOT}' no existe.")
        return

    # 1. Mapear carpetas y archivos
    carpetas = [d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))]
    data_archivos = {}
    
    print(f"🔍 Analizando dataset en: {OUTPUT_ROOT}\n")
    
    todas_las_fechas = set()
    for carpeta in carpetas:
        ruta = os.path.join(OUTPUT_ROOT, carpeta)
        archivos = [f.replace(".npy", "") for f in os.listdir(ruta) if f.endswith(".npy")]
        data_archivos[carpeta] = set(archivos)
        todas_las_fechas.update(archivos)
        print(f"📂 {carpeta.ljust(25)} | Total archivos: {len(archivos)}")

    if not todas_las_fechas:
        print("\n⚠️ No se encontraron archivos .npy.")
        return

    # 2. Contar fuego en las máscaras
    print("\n🔥 Analizando contenido de las máscaras (esto puede demorar un poco)...")
    conteo_fuego = []
    ruta_mascaras = os.path.join(OUTPUT_ROOT, MASCARA_DIR)
    
    muestras_con_fuego = 0
    total_pixeles_fuego = 0

    if os.path.exists(ruta_mascaras):
        for fecha in sorted(list(todas_las_fechas)):
            archivo_path = os.path.join(ruta_mascaras, f"{fecha}.npy")
            
            if os.path.exists(archivo_path):
                mask = np.load(archivo_path)
                # Según GOES: 0 y 1 son fuego (alta y media probabilidad)
                n_pixeles = np.sum(np.isin(mask, [0]))
                
                if n_pixeles > 0:
                    muestras_con_fuego += 1
                    total_pixeles_fuego += n_pixeles
                    conteo_fuego.append({"fecha": fecha, "pixeles_fuego": n_pixeles})
            else:
                # Si falta la máscara para esa fecha, lo anotamos
                pass 

    # 3. Mostrar Resultados
    print("\n" + "="*40)
    print("📊 RESUMEN DE INCENDIOS")
    print("="*40)
    print(f"Total de Timestamps analizados:  {len(todas_las_fechas)}")
    print(f"Muestras CON fuego:             {muestras_con_fuego}")
    print(f"Muestras SIN fuego:             {len(todas_las_fechas) - muestras_con_fuego}")
    
    if muestras_con_fuego > 0:
        print(f"Promedio píxeles fuego/imagen:  {total_pixeles_fuego / muestras_con_fuego:.2f}")
        
        # Mostrar las top 5 imágenes con más fuego
        df_fuego = pd.DataFrame(conteo_fuego).sort_values(by="pixeles_fuego", ascending=False)
        print("\n🔝 Top 5 muestras con mayor actividad:")
        print(df_fuego.head(5).to_string(index=False))
    
    # 4. Alerta de Discrepancias (fechas incompletas)
    discrepancias = []
    for fecha in todas_las_fechas:
        faltan = [c for c in carpetas if fecha not in data_archivos[c]]
        if faltan:
            discrepancias.append({"fecha": fecha, "faltan_en": ", ".join(faltan)})
    
    if discrepancias:
        print(f"\n❌ ATENCIÓN: Hay {len(discrepancias)} fechas incompletas (no se pueden usar para entrenar).")
    else:
        print("\n✅ Integridad referencial: OK (Todas las imágenes tienen su máscara).")

if __name__ == "__main__":
    auditar_dataset_con_fuego()