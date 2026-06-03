# FDCA Testing — Integración con pipeline GOES-19

Workflow para correr el algoritmo FDCA sobre datos reales descargados
por el pipeline existente (`pipeline.py`), y documentar el efecto de
cada fase con figuras.

---

## Estructura de archivos

```
obtencion_imagenes/          ← ya existía
  pipeline.py
  downloader.py
  config.yaml                ← ACTUALIZADO (agregar B13, B15, B02)
  sync_hf.py
  dataset/
    uruguay/
      ABI-L1b-Rad-B07/
      ABI-L1b-Rad-B14/
      ABI-L2-FDCF/
      ABI-L1b-Rad-B13/       ← nuevo (para FDCA)
      ABI-L1b-Rad-B15/       ← nuevo (para FDCA)
      ABI-L1b-Rad-B02/       ← nuevo (para FDCA)
      manifest.json

fdca/                        ← ya existía (el algoritmo)
  fdca.py, part1.py, part2.py, ...

fdca_adapter.py              ← NUEVO: traduce .npy → FDCAInput
run_fdca.py                  ← NUEVO: corre FDCA y genera figuras
```

---

## Paso 1 — Actualizar config.yaml

Reemplazá el config.yaml existente por la versión provista.
Agrega tres productos nuevos necesarios para el FDCA:

| Nuevo producto    | Banda | Para qué lo usa el FDCA |
|-------------------|-------|--------------------------|
| ABI-L1b-Rad-B13   | 13    | Canal híbrido FPT (ATBD 3.4.2.2) |
| ABI-L1b-Rad-B15   | 15    | Test de nube split-window (ATBD 3.4.2.3) |
| ABI-L1b-Rad-B02   | 2     | Reflectancia visible / glint / nube diurna |

Los productos existentes (B07, B14, FDCF) no cambian.

---

## Paso 2 — Elegir un timestamp con fuego

```bash
cd obtencion_imagenes
python pipeline.py fire-stats --region uruguay
```

Elegí un timestamp del Top 10. Ejemplo: 20250926_1900

---

## Paso 3 — Descargar las bandas adicionales

```bash
python pipeline.py download \
  --region uruguay \
  --start "2025-09-26 19:00" \
  --end   "2025-09-26 19:00" \
  --products ABI-L1b-Rad-B13 ABI-L1b-Rad-B15 ABI-L1b-Rad-B02
```

El pipeline detecta qué ya existe y no vuelve a bajar B07, B14, FDCF.

Verificar completitud:
```bash
python pipeline.py status \
  --region uruguay \
  --products ABI-L1b-Rad-B07 ABI-L1b-Rad-B14 ABI-L1b-Rad-B13 ABI-L1b-Rad-B15 ABI-L1b-Rad-B02
```

---

## Paso 4 — Correr el FDCA

Desde la raíz del proyecto (donde están fdca/ y fdca_adapter.py):

```bash
python run_fdca.py \
  --timestamp 20250926_1900 \
  --region uruguay \
  --dataset-root obtencion_imagenes/dataset \
  --config obtencion_imagenes/config.yaml \
  --save-outputs
```

Genera figuras en figures/20250926_1900/ y outputs en data/20250926_1900/

---

## Figuras generadas

| Figura | Evidencia |
|--------|-----------|
| 00_inputs.png | BT7, BT14, SZA — entradas al algoritmo |
| 01_part1_filters.png | Parte I: filtros, FailChar, candidatos sobre BT7 |
| 02_part2_confirm.png | Parte II: confirmación, categorías, FRP, diferencia P1→P2 |
| 03_fire_map.png | Mapa final georeferenciado en lat/lon |

---

## Paso 5 — Subir a Hugging Face

```bash
# Copiar outputs al dataset antes del sync
cp -r data/20250926_1900/    obtencion_imagenes/dataset/uruguay/FDCA-outputs/
cp -r figures/20250926_1900/ obtencion_imagenes/dataset/uruguay/FDCA-figures/

# Usar el sync_hf.py que ya tienen
python obtencion_imagenes/sync_hf.py
```
