# GOES-19 Fire Dataset Builder (Uruguay)
Este proyecto permite la construcción de un dataset incremental y multiespectral utilizando datos del satélite GOES-19 para la detección de incendios forestales mediante redes de segmentación (U-Net).

1. Estructura del Dataset
El sistema utiliza un enfoque de almacenamiento desacoplado. Cada producto y banda se guarda en su propia carpeta para permitir la expansión del dataset (agregar nuevas bandas) sin necesidad de descargar o procesar nuevamente los datos existentes.

Plaintext
dataset_focos_igeos/
├── ABI-L1b-Rad-B07/   # Infrarrojo de onda corta (3.9 µm) - Entrada principal
├── ABI-L1b-Rad-B14/   # Infrarrojo de onda larga (11.2 µm) - Contexto térmico
└── ABI-L2-FDCF/       # Fire Detection Product - Máscaras (Ground Truth)
2. Generación del Dataset
El script principal crear_dataset.py permite descargas paralelas e incrementales.

Cómo usarlo:
Configurar el rango: Define las fechas y las bandas deseadas en el bloque if __name__ == "__main__":.

Ejecutar: ```bash
python crear_dataset.py

Actualización: Si ya descargaste la Banda 7 y ahora quieres la 14, simplemente añade ("ABI-L1b-Rad", 14) a la lista de configuración y vuelve a correr el script. El código detectará los archivos existentes y solo descargará lo faltante.

Cuidados y Limitaciones:
Uso de Memoria: El script guarda archivos .npy (NumPy). Son rápidos pero no comprimen tanto como los .npz. Asegúrate de tener espacio en disco (~1-2 MB por imagen recortada).

Cuotas de Red: La descarga es paralela (max_workers). No excedas de 4-8 hilos para evitar bloqueos por parte de los servidores de Amazon S3 (donde residen los datos de GOES).

Alineación Espacial: El script asume que todas las bandas tienen la misma resolución (2 km para B7 y B14). Si agregas bandas visibles (B2), estas tienen 500m y requerirán un re-escalado (resizing) antes de guardarlas.

3. Entrenamiento de la Red (U-Net)
Para entrenar una U-Net con estos datos, sigue este flujo lógico:

Carga de Datos (Data Loader)
No cargues todas las carpetas a la vez. Crea un Dataset de PyTorch que reciba una lista de timestamps y busque los archivos correspondientes en las subcarpetas:

Python
# Ejemplo de lógica interna en __getitem__
b7 = np.load(f"path/ABI-L1b-Rad-B07/{timestamp}.npy")
b14 = np.load(f"path/ABI-L1b-Rad-B14/{timestamp}.npy")
input_tensor = np.stack([b7, b14], axis=0) # Shape: (2, H, W)
Preprocesamiento Crítico:
Normalización: Los valores de radiancia varían mucho entre el día y la noche. Se recomienda convertir a Temperatura de Brillo o normalizar por estadísticas locales (media/desviación).

Máscaras Binarias: El producto FDC tiene varios códigos. Para un entrenamiento binario, transforma la máscara:

Fuego = 1 si el valor original es 0.

No Fuego = 0 para cualquier otro valor.

Estrategia de Entrenamiento:
Pérdida (Loss): Dado que los incendios son eventos "raros" (pocos píxeles positivos), no uses MSE o Accuracy. Usa Dice Loss o Binary Cross Entropy con pesos, dándole mucho más peso a los píxeles de fuego.

Muestreo: Entrena con un mix de 50% imágenes con fuego y 50% imágenes sin fuego para evitar que la red aprenda a predecir siempre "cero".

4. Próximos Pasos
[ ] Implementar normalización Min-Max por banda.

[ ] Agregar scripts de visualización para inspeccionar las muestras .npy.

[ ] Integrar segmentation-models-pytorch para el entrenamiento rápido.

EL MAS ACTUALIZADO ES crear_dataset_imagenes_2.py