# Obtención de Imágenes GOES-19

Este módulo descarga imágenes satelitales del satélite GOES-19 y las guarda en tu computadora para usarlas en el entrenamiento de modelos de detección.

---

## ¿Qué hace esto?

El satélite GOES-19 toma fotos de Sudamérica cada hora. Este código descarga esas imágenes, las recorta a la región que nos interesa (por ejemplo, Uruguay), y las guarda ordenadas por carpetas.

Cada imagen se guarda como un archivo `.npy` (formato numérico de Python). Hay un archivo por hora, por producto.

---

## Archivos del proyecto

```
obtencion_imagenes/
├── config.yaml      ← Configuración: qué descargar y de qué región
├── pipeline.py      ← El script principal que vas a usar
├── downloader.py    ← Lógica interna de descarga (no tocar)
├── manifest.py      ← Registro de qué se descargó (no tocar)
```

Los demas archivos son legacy.
---

## Antes de empezar

Asegurate de tener las dependencias instaladas. Usar un evironment. Desde la carpeta raíz del proyecto:

```bash
pip install goes2go pyproj pyyaml numpy
```
o

```bash
pip install -r requirements.txt
```

Todos los comandos se corren desde dentro de la carpeta `obtencion_imagenes/`:

```bash
cd obtencion_imagenes
```

---

## Uso básico

### 1. Ver qué productos están disponibles

```bash
python pipeline.py list-products
```

Esto muestra los productos configurados, por ejemplo:

```
📡 Productos disponibles en config.yaml:

  ABI-L1b-Rad-B07     B07  Shortwave IR 3.9µm - fire thermal signature
  ABI-L1b-Rad-B14     B14  Longwave IR 11.2µm - thermal context
  ABI-L2-FDCF           -  Fire detection mask - ground truth label
```

### 2. Ver qué regiones están disponibles

```bash
python pipeline.py list-regions
```

### 3. Descargar imágenes

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-02 23:00" \
  --products ABI-L1b-Rad-B07 ABI-L2-FDCF
```

Esto descarga la Banda 7 y la máscara de fuego para Uruguay, hora por hora, entre el 1 y el 2 de septiembre de 2025.

Start of operational data: April 7, 2025 

Mientras descarga, vas a ver algo así:

```
🚀 Descargando 48 archivos con 4 workers...

  💾 20250901_0000  ABI-L1b-Rad-B07    downloaded
  💾 20250901_0000  ABI-L2-FDCF        downloaded
  ✅ 20250901_0100  ABI-L1b-Rad-B07    exists
  ...

✔ Descargados: 40  |  Ya existían: 8  |  Errores: 0
```

- 💾 = descargado ahora
- ✅ = ya existía, no se volvió a descargar
- ⚠️ = no había datos para esa hora
- ❌ = error de conexión u otro problema

### 4. Agregar una banda nueva sin re-descargar todo

Si ya tenés la Banda 7 descargada y querés agregar la Banda 14, simplemente corrés el mismo comando con el nuevo producto. El script detecta automáticamente qué ya existe y solo descarga lo que falta:

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-02 23:00" \
  --products ABI-L1b-Rad-B14
```

### 5. Verificar el estado del dataset

```bash
python pipeline.py status \
  --region uruguay \
  --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 ABI-L2-FDCF
```

Esto muestra cuántos archivos se descargaron por producto y si hay timestamps incompletos (horas donde falta algún producto):

```
📦 Estado por producto:
  ABI-L1b-Rad-B07     ✅ 48  ❌ 0
  ABI-L1b-Rad-B14     ✅ 48  ❌ 0
  ABI-L2-FDCF         ✅ 48  ❌ 0

✅ Timestamps completos (todos los productos): 48
```

### 6. Ver estadísticas de fuego

```bash
python pipeline.py fire-stats --region uruguay
```

Muestra cuántos timestamps tienen fuego detectado, el porcentaje, y un ranking de los 10 con más píxeles de fuego:

```
🔥 ESTADÍSTICAS DE FUEGO — uruguay
=============================================
  Total timestamps analizados : 48
  Con fuego detectado         : 12  (25.0%)
  Sin fuego                   : 36  (75.0%)

🔝 Top 10 timestamps con más píxeles de fuego:
  Timestamp              Píxeles fuego
  -----------------------------------
  20250915_1400                    143
  20250912_1600                     87
  ...
```

### 7. Visualizar una imagen y su máscara

```bash
python pipeline.py visualize --region uruguay --timestamp 20250901_1200
```

Abre una ventana con dos paneles: la imagen de la Banda 7 (infrarrojo térmico) a la izquierda y la máscara de fuego a la derecha. Útil para inspeccionar visualmente el dataset antes de entrenar.

Si no sabés qué timestamps tienen fuego, primero corré `fire-stats` para ver el ranking.

---

## ¿Dónde se guardan los archivos?

Los archivos se guardan en una carpeta `dataset/` dentro de `obtencion_imagenes/`, organizada así:

```
dataset/
└── uruguay/
    ├── ABI-L1b-Rad-B07/
    │   ├── 20250901_0000.npy
    │   ├── 20250901_0100.npy
    │   └── ...
    ├── ABI-L1b-Rad-B14/
    │   └── ...
    ├── ABI-L2-FDCF/
    │   └── ...
    └── manifest.json   ← registro de todo lo descargado
```

Cada archivo `.npy` es una imagen recortada a la región elegida, guardada como una matriz numérica.

El archivo `manifest.json` es un registro automático de todo lo que se descargó, con estado y dimensiones. No hace falta abrirlo manualmente.

---

## Agregar un nuevo producto o región

### Nuevo producto (banda)

Abrí `config.yaml` y agregá un bloque nuevo bajo `products`:

```yaml
  - id: ABI-L1b-Rad-B02
    product: ABI-L1b-Rad
    band: 2
    variable: Rad
    dtype: float32
    description: "Visible 0.64µm - luz visible"
```

Después podés descargarlo con `--products ABI-L1b-Rad-B02` sin tocar ningún otro archivo.

### Nueva región

Agregá una entrada bajo `regions` en `config.yaml`:

```yaml
  patagonia:
    lat_min: -55.0
    lat_max: -40.0
    lon_min: -75.0
    lon_max: -60.0
```

Y usala con `--region patagonia`.

---

## Opciones avanzadas

| Opción | Descripción | Default |
|---|---|---|
| `--interval N` | Descargar cada N horas en vez de cada 1 | 1 |
| `--workers N` | Cuántas descargas en paralelo | 4 (config.yaml) |

Ejemplo: descargar cada 3 horas con 6 workers:

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-01 00:00" \
  --end "2025-09-30 23:00" \
  --products ABI-L1b-Rad-B07 ABI-L2-FDCF \
  --interval 3 \
  --workers 6
```

> ⚠️ No uses más de 8 workers. Los servidores de NOAA pueden bloquear conexiones si se hacen demasiadas descargas simultáneas.

---

## Sincronizar el dataset con Hugging Face

El dataset se almacena de forma compartida en Hugging Face para que todo el equipo pueda acceder a él sin necesidad de descargar todo desde cero.

### ¿Cuándo usar este script?

Usá `sync_hf.py` cada vez que descargues datos nuevos con `pipeline.py` y quieras que el resto del equipo los tenga disponibles. El flujo típico es:

1. Descargás datos nuevos con `pipeline.py`
2. Verificás que todo esté bien con `pipeline.py status`
3. Subís los cambios a Hugging Face con `sync_hf.py`

Solo la persona que descargó los datos necesita correr este script. Los demás simplemente descargan desde HF.

### Subir datos a Hugging Face

Desde la raíz del proyecto:

```bash
.venv/bin/python obtencion_imagenes/sync_hf.py
```

Vas a ver algo así:

```
📦 Repositorio listo: https://huggingface.co/datasets/tu-usuario/goes19-uruguay-fires
⬆️  Subiendo dataset/ → tu-usuario/GERIS-Goes19-uruguay-fires ...

✅ Dataset sincronizado correctamente.
   Ver en: https://huggingface.co/datasets/tu-usuario/GERIS-Goes19-uruguay-fires
```

El script solo sube los archivos nuevos o modificados, no vuelve a subir lo que ya estaba.

### Descargar el dataset (para el resto del equipo)

Si sos un compañero que quiere tener el dataset localmente, desde la raíz del proyecto:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='valentina2323/GERIS-Goes19-uruguay-fires',
    repo_type='dataset',
    local_dir='obtencion_imagenes/dataset',
    token='hf_tu_token'
)
"
```

> ⚠️ El token de Hugging Face es personal y privado. No lo compartas ni lo subas a GitHub. Ya está protegido en el archivo `.env` que está en el `.gitignore`.
