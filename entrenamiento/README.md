# Entrenamiento del modelo de detección de incendios

Este módulo entrena una red neuronal para detectar incendios forestales a partir de imágenes satelitales del GOES-19. El modelo aprende a identificar qué píxeles de una imagen corresponden a fuego activo.

---

## ¿Qué hace esto, en términos simples?

Le mostramos al modelo miles de imágenes satelitales junto con su "respuesta correcta" (una máscara que indica dónde hay fuego). El modelo ajusta sus parámetros internos hasta que aprende a predecir esas máscaras por sí solo, sin que le digamos la respuesta.

Una vez entrenado, el modelo puede recibir una imagen nueva que nunca vio y decir: "en estos píxeles hay fuego".

---

## Archivos del módulo

```
entrenamiento/
├── config.yaml    ← Todos los parámetros configurables (sin tocar el código)
├── dataset.py     ← Cómo se cargan y preparan las imágenes
├── train.py       ← El loop de entrenamiento
├── evaluate.py    ← Evaluación y visualización de predicciones
└── checkpoints/   ← Donde se guarda el modelo entrenado (se crea automáticamente)
```

---

## Antes de empezar

Necesitás tener el dataset descargado. Si no lo tenés, seguí las instrucciones en `obtencion_imagenes/README.md`.

Instalá las dependencias desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

Todos los comandos se corren desde dentro de la carpeta `entrenamiento/`:

```bash
cd entrenamiento
```

---

## Uso

### Entrenar el modelo

```bash
python train.py
```

O especificando un archivo de configuración distinto:

```bash
python train.py --config config.yaml
```

Durante el entrenamiento vas a ver algo así:

```
🖥  Usando dispositivo: mps
📊 Dataset: 668 timestamps  |  train: 534  |  val: 134

🚀 Iniciando entrenamiento — 30 épocas

Época   1/30  train loss: 0.8231  train IoU: 0.1042  |  val loss: 0.7654  val IoU: 0.1318
  ✅ Mejor modelo guardado (val IoU: 0.1318)
Época   2/30  train loss: 0.6912  train IoU: 0.1587  |  val loss: 0.6201  val IoU: 0.1892
  ✅ Mejor modelo guardado (val IoU: 0.1892)
...
✔ Entrenamiento completo. Mejor val IoU: 0.4721
  Modelo guardado en: checkpoints/best_model.pth
```

El modelo se guarda automáticamente cada vez que mejora en el conjunto de validación.

### Evaluar el modelo entrenado

```bash
python evaluate.py
```

O apuntando a un checkpoint específico:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Esto imprime las métricas y abre una ventana con visualizaciones comparando la predicción del modelo contra la máscara real:

```
📊 MÉTRICAS DE EVALUACIÓN
===================================
  IoU       : 0.4721
  F1 Score  : 0.6412
  Precision : 0.7103
  Recall    : 0.5851
```

---

## Configuración (`config.yaml`)

Todos los parámetros del entrenamiento se controlan desde `config.yaml`. No hace falta tocar el código Python para cambiar cosas.

```yaml
data:
  dataset_root: ../obtencion_imagenes/dataset/uruguay
  bands:
    - ABI-L1b-Rad-B07       # Banda infrarroja térmica (3.9µm) — señal principal de fuego
  mask_product: ABI-L2-FDCF # Producto de detección de fuego — etiqueta de verdad
  val_split: 0.2            # 20% de los datos para validación
  image_size: 256           # Las imágenes se recortan a 256×256 píxeles

model:
  architecture: Unet
  encoder: resnet34         # Encoder preentrenado en ImageNet
  encoder_weights: imagenet

training:
  epochs: 30
  batch_size: 8
  learning_rate: 0.0001
  device: auto              # Detecta automáticamente MPS (Mac), CUDA (GPU) o CPU

output:
  checkpoint_dir: checkpoints
  best_model: checkpoints/best_model.pth
```

---

## Decisiones técnicas

### Arquitectura: U-Net con encoder ResNet34

Se eligió **U-Net** porque es la arquitectura estándar para segmentación semántica de imágenes. Su estructura en forma de "U" combina dos partes:

- **Encoder (bajada)**: extrae características de la imagen a distintas escalas, desde bordes simples hasta patrones complejos.
- **Decoder (subida)**: reconstruye la imagen píxel a píxel usando esas características para producir la máscara de predicción.
- **Skip connections**: conexiones directas entre encoder y decoder que preservan detalles espaciales finos, importantes para detectar focos pequeños de fuego.

El encoder es un **ResNet34 preentrenado en ImageNet**. Aunque ImageNet contiene fotos de objetos cotidianos (perros, autos, etc.), las características de bajo nivel que aprende (bordes, texturas, gradientes de intensidad) son útiles como punto de partida. Esto se llama *transfer learning* y acelera el entrenamiento considerablemente.

### Función de pérdida: Dice + BCE

El problema principal es el **desbalance de clases**: en la mayoría de las imágenes, los píxeles de fuego son una fracción muy pequeña del total. Si usáramos solo Binary Cross-Entropy (BCE) estándar, el modelo podría aprender a predecir "no hay fuego en ningún lado" y aun así tener una pérdida baja.

Para contrarrestar esto se usan dos pérdidas combinadas:

- **BCE con `pos_weight=10`**: penaliza 10 veces más los falsos negativos (píxeles de fuego que el modelo no detectó) que los falsos positivos.
- **Dice Loss**: mide directamente el solapamiento entre la predicción y la máscara real. Es naturalmente robusta al desbalance porque trabaja con proporciones, no con conteos absolutos.

La pérdida final es la suma de ambas: `loss = BCE + Dice`.

### Métrica principal: IoU (Intersection over Union)

El IoU mide qué tan bien se superpone la predicción del modelo con la máscara real:

```
IoU = área de intersección / área de unión
```

Un IoU de 1.0 significa predicción perfecta. Un IoU de 0.0 significa que no hubo ningún solapamiento. Es la métrica estándar en tareas de segmentación porque es más exigente que la precisión simple.

### Normalización por banda

Cada banda satelital tiene su propia escala de valores (radiancia, temperatura, etc.). Antes de pasarle los datos al modelo, se normalizan restando la media y dividiendo por la desviación estándar, calculadas sobre todo el dataset de entrenamiento. Esto estabiliza el entrenamiento y hace que el modelo no dependa de las unidades físicas de cada sensor.

### Scheduler de learning rate

Se usa `ReduceLROnPlateau`: si la pérdida de validación no mejora durante 5 épocas seguidas, el learning rate se reduce a la mitad. Esto permite que el modelo "afine" sus parámetros con pasos más pequeños cuando ya está cerca de una buena solución.

### Detección de dispositivo automática

El código detecta automáticamente el mejor hardware disponible:
1. **MPS** (Apple Silicon — Mac M1/M2/M3)
2. **CUDA** (GPU NVIDIA)
3. **CPU** (fallback)

Esto se puede sobreescribir en `config.yaml` con `device: cpu`, `device: cuda`, etc.

---

## Datos de entrada y salida

### Entrada

Cada muestra del dataset es un par `(imagen, máscara)`:

- **Imagen**: array de forma `(C, 256, 256)` donde `C` es el número de bandas configuradas. Actualmente se usa solo la Banda 7 (infrarrojo de onda corta, 3.9µm), que es la más sensible a la temperatura de los focos de fuego activo.
- **Máscara**: array binario de forma `(1, 256, 256)` donde `1` indica píxel de fuego y `0` indica no-fuego.

### Etiqueta de verdad (ground truth)

La máscara de fuego se obtiene del producto **ABI-L2-FDCF** del GOES-19, que es el producto oficial de detección de fuego de la NOAA. Este producto incluye un campo `DQF` (Data Quality Flag):

- `DQF = 0` → fuego detectado con alta confianza → se marca como `1` en la máscara
- Cualquier otro valor → no hay fuego o baja confianza → se marca como `0`

### Salida del modelo

El modelo produce un mapa de probabilidades de forma `(1, 256, 256)`. Para convertirlo en una máscara binaria se aplica un umbral de `0.5`: valores mayores se interpretan como fuego.

---

## División train/validación

El dataset se divide aleatoriamente en:
- **80% entrenamiento**: el modelo aprende de estos datos
- **20% validación**: se usan para medir el rendimiento en datos que el modelo nunca vio durante el entrenamiento

Esta división es aleatoria por timestamp (hora), no por región ni por día, para evitar sesgos temporales.

---

## Estructura del dataset esperada

El módulo espera que el dataset esté organizado así (generado por `obtencion_imagenes/`):

```
obtencion_imagenes/dataset/uruguay/
├── ABI-L1b-Rad-B07/
│   ├── 20250901_0000.npy
│   ├── 20250901_0100.npy
│   └── ...
└── ABI-L2-FDCF/
    ├── 20250901_0000.npy
    ├── 20250901_0100.npy
    └── ...
```

Solo se incluyen en el dataset los timestamps donde **todos** los productos requeridos están presentes. Si falta la máscara o la banda para una hora, ese timestamp se descarta automáticamente.
