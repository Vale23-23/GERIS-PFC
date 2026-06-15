# GERIS — Pipeline FDCA para GOES-19 (Uruguay)

Objetivo es implementar el **FDCA**
(Fire Detection and Characterization Algorithm) de NOAA — el algoritmo detrás
del producto operacional "Fire/Hot Spot Characterization" de los satélites
GOES — adaptado a la región de Uruguay, usando imágenes ABI del satélite
**GOES-19**.

La implementación sigue la **ATBD v2.7** (Algorithm Theoretical Basis
Document) de NOAA, traduciendo el pseudocódigo a un paquete Python (`fdca/`) modular, 
testeado y corregido pieza por pieza.

> **Estado actual (resumen ejecutivo):** el pipeline corre de punta a punta
> sin crashear, genera figuras y un resumen JSON. El bug histórico de
> temperaturas de brillo (BT) físicamente imposibles está resuelto.
> Queda pendiente investigar una sobre-detección de ~25x respecto al producto
> oficial de NOAA para la misma escena. Ver la sección
> [Estado actual / issues conocidos](#9-estado-actual--issues-conocidos).

---

## 1. Estructura del proyecto

```
GERIS-PFC/
├── implementacion/
│   ├── fdca/                      # paquete principal del algoritmo
│   │   ├── constants.py           # constantes físicas, umbrales, enums
│   │   ├── planck.py              # conversión radiancia ↔ temperatura
│   │   ├── background.py          # estimación de "background" térmico
│   │   ├── dozier.py              # caracterización sub-pixel (Dozier) + FRP
│   │   ├── part1.py                # Parte I: filtros píxel a píxel
│   │   ├── part2.py                # Parte II: confirmación y clasificación
│   │   ├── algorithm.py            # utilidades (ej. _to_epoch)
│   │   ├── fdca_adapter.py          # adaptador: .nc GOES → FDCAInput
│   │   └── downloader.py           # descarga de bandas GOES desde S3
│   │   └── run_fdca.py              # script principal end-to-end
│   ├── descargar_C02.py            # descarga batch de C02 desde server HTTP   
├── data/                            # outputs por timestamp (.npy, .json)
└── figures/                         # figuras generadas por corrida
```

Se reemplazó el config.yaml existente por la versión provista.
Agrega tres productos nuevos necesarios para el FDCA:

| Nuevo producto    | Banda | Para qué lo usa el FDCA |
|-------------------|-------|--------------------------|
| ABI-L1b-Rad-B13   | 13    | Canal híbrido FPT (ATBD 3.4.2.2) |
| ABI-L1b-Rad-B15   | 15    | Test de nube split-window (ATBD 3.4.2.3) |
| ABI-L1b-Rad-B02   | 2     | Reflectancia visible / glint / nube diurna |

Los productos existentes (B07, B14, FDCF) no cambian.

---

## 2. Requisitos e instalación

Dependencias principales (Python 3.10+ recomendado):

- `numpy`
- `xarray`
- `h5py` (lectura de archivos `.nc` de GOES)
- `boto3` (acceso al bucket S3 público de NOAA)
- `Pillow` (`PIL`) — resize entre bandas de distinta resolución
- `matplotlib` (figuras)

```bash
pip install numpy xarray h5py boto3 Pillow matplotlib
```
(Todo esto está en requirements.txt)
---

## 3. Obtención de datos
Elegir un timestamp con fuego

```bash
cd obtencion_imagenes
python pipeline.py fire-stats --region uruguay
```

Hay dos caminos para conseguir las bandas ABI necesarias (B02, B07, B13, B14,
B15):

### 3.1 `fdca/downloader.py` — descarga desde el bucket S3 de NOAA

- Accede al bucket público `noaa-goes19` usando **firma `UNSIGNED`**
  (es un bucket público; pasar `RequestPayer` normal da error de
  autenticación).
- Para cada banda IR descargada (7, 13, 14, 15) genera además un archivo
  `*_planck.json` con los **4 coeficientes de calibración Planck** propios de
  esa escena/banda:
  - `planck_fk1`
  - `planck_fk2`
  - `planck_bc1`
  - `planck_bc2`

  Estos coeficientes son los que después usa `planck.py` para invertir
  radiancia → temperatura de brillo (BT) con la fórmula oficial de NOAA,
  evitando el bug de unidades que afectaba a B07/B14 (ver sección 9).

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

### 3.2 `descargar_C02.py` — descarga batch de B02 (alta resolución)

Script standalone para bajar archivos `.nc` de B02 desde un directorio HTTP
(no S3), con:

- reintentos automáticos,
- soporte de **resume** (no vuelve a bajar lo ya descargado),
- headers HTTP apropiados para evitar errores `403`.

> **Nota:** revisá el bloque `argparse` / variables de configuración al
> principio de cada script para la sintaxis exacta de invocación en tu
> entorno (rutas, rango de fechas, bandas).


---

## 4. Cómo correr el pipeline completo

El punto de entrada es `run_fdca.py`. La función `main()` espera, como
mínimo:

- `args.config` — path a un archivo de configuración (región, parámetros).
- `args.dataset_root` — carpeta raíz donde están los `.nc` descargados.
- `args.save_outputs` — flag para guardar `.npy` + `summary.json` en
  `data/<timestamp>/`.

Invocación típica:

```bash
python -m fdca.run_fdca \
  --timestamp 20250926_1900 \ 
  --region uruguay \
  --config /ruta_a/config.yaml \
  --dataset-root /ruta/a/datos \
  --save-outputs
```

### Qué hace `main()` paso a paso

1. **Carga de inputs** (`load_fdca_input`, en `fdca_adapter.py`): lee las
   bandas `.nc`, aplica calibración Planck por-escena, calcula geometría
   (SZA, LZA, azimuth, ángulo de glint), TPW climatológico, emisividades,
   máscaras de tierra/agua/desierto, etc. Devuelve un objeto `FDCAInput`.

2. **Figura 0** (`fig_inputs`): visualización de todos los inputs crudos
   (BT7, BT14, reflectancia B02, SZA, etc.) → `figures/<ts>/00_inputs.png`.

3. **Parte I** (`run_part1`): filtros píxel a píxel → genera
   `fire_mask_p1`, `fail_char_p1` y una lista de `FireCandidate` (candidatos
   a fuego que pasaron los filtros iniciales).

4. **Figura 1** (`fig_part1`): efecto de los filtros de Parte I →
   `01_part1_filters.png`.

5. **Parte II** (`run_part2`): para cada candidato de Parte I, aplica los
   tests de confirmación/eliminación, re-evaluación de nubes/glint,
   categorización de confianza (alta/media/baja) y filtrado temporal. Devuelve
   `fire_mask_p2`, `fail_char_p2` y la lista final `confirmed`.

6. **Figura 2** (`fig_part2`): comparación Parte I vs Parte II →
   `02_part2_confirm.png`.

7. **Figura 3** (`fig_fire_map`): mapa georreferenciado de detecciones →
   `03_fire_map.png`.

8. **Outputs** (si `--save-outputs`): en `data/<timestamp>/`:
   - `fire_mask.npy` — array 2D con el código final de cada píxel (ver tabla
     en sección 8).
   - `fail_char.npy` — array 2D con el "fail character" de cada píxel (en
     qué test/etapa quedó clasificado, ver sección 8).
   - `summary.json` — resumen numérico de la corrida (ver sección 6.1).

9. **Resumen impreso en consola** — mismos números que `summary.json`, más
   tiempos de ejecución.

---

## 5. ¿Qué devuelve cada parte?

### 5.1 `fdca_adapter.load_fdca_input(...)` → `FDCAInput`

Estructura con (entre otros) los siguientes campos:

| Campo | Significado |
|---|---|
| `bt7`, `bt14`, `bt13`, `bt15` | Temperatura de brillo por banda [K], calculada con `planck_temp_from_coeffs` usando los coeficientes propios de cada `.nc` |
| `rad7`, `rad14`, `rad13`, `rad15` | Radiancias correspondientes [W·m⁻²·sr⁻¹·m⁻¹], regeneradas con `planck_rad(band, bt)` para mantener consistencia |
| `refl2` | Reflectancia de B02 (alta resolución), reescalada a la grilla de las bandas IR |
| `latitudes`, `longitudes` | Grilla de coordenadas |
| `sza`, `lza`, `azimuth`, `glint_angle` | Geometría solar/satelital por píxel |
| `tpw` | Total Precipitable Water (climatológico) |
| `emiss7`, `emiss14` | Emisividades de superficie |
| `land_cover`, `land_mask`, `desert_mask`, `usgs_eco` | Máscaras auxiliares de superficie |
| `data_quality` | Flags de calidad de datos |
| `prev_fire_mask` | Máscara de fuegos de un timestamp anterior (para filtrado temporal), o `None` si no hay |
| `scan_time` | Timestamp del scan actual |

### 5.2 `run_part1(...)` → `(fire_mask_p1, fail_char_p1, candidates)`

- **`fire_mask_p1`**: array 2D, inicializado con un valor "no procesado" /
  sentinel para todos los píxeles. Parte I **no asigna** los códigos finales
  10-15; eso lo hace Parte II. (⚠️ ver sección 9.3 — esto es clave para
  entender el resumen impreso).
- **`fail_char_p1`**: array 2D con el código de "fail character" inicial
  (en qué filtro de Parte I quedó cada píxel: nube, agua, glint, frío, etc.)
- **`candidates`**: lista de `FireCandidate`, uno por cada píxel que **pasó**
  los filtros de descarte de Parte I (candidato a posible fuego). Cada
  `FireCandidate` trae, entre otros: `i, j` (posición), `bt7`, `bt14`,
  `bt7_bkg`, `bt14_bkg`, `bt7_corr`, `rad_diff_sigma`, `refl_pixel`, `reflb`,
  `sza`, `is_day`, `n_passes` (cuántas iteraciones tomó el background),
  `fire_temp`, `fire_frac`, `frp`, `fail_char`, `albedo`, `albedo_bkg`,
  `is_saturated`, `pass_along_scan`, `std_dev_reflb_max`, y un sub-objeto
  `bkg` con estadísticas de background (incluye `std_dev_7_14_diff`).

### 5.3 `run_part2(...)` → `(fire_mask_p2, fail_char_p2, confirmed)`

- **`fire_mask_p2`**: copia de `fire_mask_p1` donde, **solo para los píxeles
  confirmados**, se sobrescribe el código final (10-15, o 30-35 si hay match
  temporal). Los píxeles no confirmados (rechazados por los tests de Parte II,
  o que ni siquiera fueron candidatos) **mantienen el valor sentinel** de
  Parte I.
- **`fail_char_p2`**: igual que `fail_char_p1` pero actualizado para los
  píxeles re-evaluados (ej. reclasificados a `F11` por nube/niebla, o `+20`/
  `+30` por upgrade a media/alta confianza).
- **`confirmed`**: sublista de `candidates` que pasó los tests de Parte II
  (no cayó en `cond1`/`cond2`/`cond3`).

### 5.4 `compute_dozier(...)` → `DozierResult`

Caracterización sub-pixel de cada candidato:

| Campo | Significado |
|---|---|
| `fire_temp` (`Tt`) | Temperatura del componente "fuego" del píxel [K] |
| `fire_frac` (`p`) | Fracción del píxel ocupada por fuego [0-1] |
| `fire_area` | Área del componente fuego [km²] (se completa después con `compute_pixel_area`) |
| `frp` | Fire Radiative Power [MW] |
| `fail_char` | Código de fallo si la solución no es válida |
| `valid` | `True` si Newton convergió con `Tt ≥ MIN_FIRE_TEMP` |

---

## 6. Lógica de cada módulo

### 6.1 `planck.py`

Conversión radiancia ↔ temperatura de brillo usando la **fórmula oficial de
inversión de NOAA**, con los 4 coeficientes propios de cada banda/escena
(`fk1`, `fk2`, `bc1`, `bc2`) leídos del `.nc` original vía
`load_planck_coeffs()` (que mapea `planck_fk1 → fk1`, etc. desde el
`*_planck.json` generado por `downloader.py`).

Funciones clave:

- `planck_rad(band, T)` — temperatura → radiancia (usada para regenerar
  `rad7/rad14/...` de forma consistente con las BT corregidas).
- `planck_temp(band, rad)` — radiancia → temperatura (usa coeficientes
  "genéricos"/por banda; usado dentro de `dozier.py` en la bisección).
- `planck_temp_from_coeffs(rad, fk1, fk2, bc1, bc2)` — **versión correcta**,
  usada por `fdca_adapter.py` para BT7/BT14/BT13/BT15, con los coeficientes
  reales de cada `.nc`.
- `planck_deriv_T(band, T, p)` — derivada de la radiancia respecto a `T`
  (Jacobiano de Dozier).

> Genera warnings de `overflow encountered in exp` cuando se evalúan
> temperaturas extremas (ej. durante la bisección de Dozier probando valores
> de `p` muy chicos). Ver sección 9.4.

### 6.2 `background.py`

Estima, para cada candidato, la **temperatura de background** (BT7/BT14 "sin
fuego") creciendo una ventana espacial alrededor del píxel hasta cumplir
criterios de cantidad/calidad de píxeles válidos (hasta `BKG_MAX_ITER`
iteraciones — `n_passes`). Devuelve además estadísticas como
`std_dev_7_14_diff` y `albedo_bkg`, usadas en los umbrales de Parte II.

### 6.3 `dozier.py` — caracterización sub-pixel y FRP

Implementa el método de **Dozier** (ATBD 3.4.2.10) en dos etapas:

1. **`_solve_bisection`**: bisección logarítmica de 15 iteraciones
   (`DOZIER_BISECT_N`) sobre la fracción de fuego `p`, entre `DOZIER_P_LOWER`
   y `DOZIER_P_UPPER`. En cada paso calcula la radiancia "atribuible al
   fuego" en B07 y B14 (`_dozier_rad_fire`), invierte a temperatura con
   `planck_temp`, y compara signos de `Tt7 - Tt14` para decidir hacia qué
   lado mover el intervalo. Devuelve `(p_mid, Tt)` iniciales.

2. **`_solve_newton`**: refina `(p, Tt)` con Newton-Raphson 2D, resolviendo
   simultáneamente las ecuaciones de balance de radiancia en B07 y B14
   (`f_k = p·L_k(Tt) + (1-p)·L_k(Tb) - A_k = 0`), usando el Jacobiano
   analítico (`planck_deriv_T`). Incluye el guard:

   ```python
   if not np.isfinite(Tt) or Tt <= 0:
       break
   ```

   que corta el loop apenas `Tt` deja de ser un número válido o positivo,
   evitando gastar las `DOZIER_NEWTON_MAX` iteraciones con aritmética
   inválida. Converge cuando `|f7| < tol·|A7|` y `|f14| < tol·|A14|`
   (`DOZIER_NEWTON_TOL`, tolerancia **relativa**).

3. **`compute_dozier`**: orquesta bisección + Newton y clasifica el
   resultado:
   - `Tt ≤ 0` o no convergió → `fire_temp = -999`, `fail_char = F6`
     ("Dozier falló").
   - `Tt < MIN_FIRE_TEMP` → píxel posible-fuego pero sin solución válida;
     `fail_char = F9` (si hay glint) o `F6` (si no).
   - `Tt ≥ MIN_FIRE_TEMP` → solución válida (`valid = True`).

4. **`compute_pixel_area`**: área del píxel en km² usando una caja 4×4
   alrededor de `(i, j)`, distancias great-circle divididas por 4, y fórmula
   de Herón para el área del paralelogramo resultante.

5. **`compute_frp`**: Fire Radiative Power vía aproximación MIR (ATBD eq.
   3.4):

   ```
   FRP = (A_pixel[m²] / FRP_MIR_A) · σ_SB · (L_MIR_obs - L_MIR_bkg)   [W → MW]
   ```

### 6.4 `part1.py` — filtros píxel a píxel (ATBD 3.4.2.x, primeras etapas)

Recorre toda la grilla aplicando, por píxel, los filtros de descarte rápido
de la ATBD: máscaras de agua/nube/desierto, umbrales día/noche de BT7/BT14,
chequeo de saturación, cálculo de background (`background.py`) y geometría de
glint. Los píxeles que **no** son descartados se convierten en
`FireCandidate` y se les corre `compute_dozier` para obtener una primera
estimación de `fire_temp`/`fire_frac`/`frp`. `fire_mask_p1` queda con el
valor sentinel salvo que Parte I marque explícitamente algo (a confirmar
contra tu implementación real).

### 6.5 `part2.py` — confirmación y clasificación (ATBD 3.4.2.14-3.4.2.18)

Para cada candidato de Parte I:

1. **Tests de eliminación de falsas alarmas** (`cond1`, `cond2`, `cond3`):
   combinan `BT7 - Tb3.9` (diferencia contra el background de B07),
   `BT7 - BT14`, geometría día/noche (`sza_cos`), y un test de reflectancia
   (`refl - reflb` vs `std_reflb_p2`) o "along-scan". Si se cumple alguno,
   el píxel se descarta (`continue`, sin tocar `fire_mask`).

2. **Re-evaluación de glint/nube/niebla**: si el `fail_char` original es
   `F9`/`F10` (posible glint o nube), se re-chequea con criterios de
   albedo (`alb`, `alb - albb`) y `bt7_corr`; si corresponde, se reclasifica
   a `F11`.

3. **Categorización inicial** (`_assign_fire_category`):
   - `Tt > MIN_FIRE_TEMP` y válido → `PROCESSED` (10).
   - Saturado sin solución → `SATURATED` (11).
   - `fail_char` en (`F9`, `F10`) → `CLOUD_CONTAM` (12).
   - resto de "posibles fuego" → `LOW_PROB` (15) por defecto.

4. **Upgrade de confianza** (para `fail_char` en `F3, F4, F6, F8` con
   `Tt < 0`): se calculan umbrales dinámicos con `_high_med_thresholds`
   (que dependen de `n_passes`, `bt7_bkg_std`, y la diferencia
   BT7-BT14 de background) y, si `bt7_corr - Tb7` y `Tb7 - Tb14` superan
   esos umbrales (más el test de reflectancia), el píxel sube a
   `MED_PROB` (14, `fail_char += 20`) o `HIGH_PROB` (13, `fail_char += 30`).

5. **FRP**: si `n_passes > BKG_MAX_ITER`, se invalida (`frp = -99`).

6. **Filtrado temporal** (ATBD 3.4.2.16): si `prev_fire_mask` no es `None`,
   se busca en una ventana espacial `±TEMPORAL_PIXEL_RAD` alrededor del
   píxel si hubo fuego en las últimas `TEMPORAL_WINDOW_H` horas
   (`current_epoch - last_t ≤ TEMPORAL_WINDOW_H·3600`). Si sí, se suma
   **+20** al código final (10→30, ..., 15→35). Con `prev_fire_mask = None`
   este bloque se salta por completo y `temporally_filtered` queda en
   `False` para todos los píxeles.

7. Se escribe `fire_mask[i, j] = fire_code` y el candidato pasa a
   `confirmed`.

---

## 7. `constants.py` — qué define (alto nivel)

> Valores exactos a confirmar contra tu archivo real; esta tabla documenta
> el **rol** de cada constante según su uso en `dozier.py` / `part2.py`.

| Constante | Rol |
|---|---|
| `FireMask` (enum) | Códigos finales de máscara: `PROCESSED=10`, `SATURATED=11`, `CLOUD_CONTAM=12`, `HIGH_PROB=13`, `MED_PROB=14`, `LOW_PROB=15` (convención estándar del producto GOES Fire/Hot Spot) |
| `FailChar` (enum) | Códigos de "por qué" un píxel quedó en cierto estado (`F1`...`F11+`), usados internamente para ramificar la lógica de Parte I/II |
| `MIN_FIRE_TEMP` | Umbral de `Tt` [K] para considerar una solución de Dozier "procesada" (≈ 400 K en la ATBD) |
| `MAX_SURF_TEMP` | Umbral de temperatura de superficie "normal" (separa `-999` de `-Tt`) |
| `DOZIER_P_UPPER`, `DOZIER_P_LOWER` | Rango de búsqueda de la fracción de fuego `p` en la bisección |
| `DOZIER_BISECT_N` | Nº de iteraciones de la bisección logarítmica (15 según ATBD) |
| `DOZIER_NEWTON_MAX`, `DOZIER_NEWTON_TOL` | Máx. iteraciones y tolerancia relativa de Newton |
| `SIGMA_SB` | Constante de Stefan-Boltzmann |
| `FRP_MIR_A`, `LAMBDA` | Constantes de la aproximación MIR de FRP |
| `BKG_MAX_ITER` | Tope de iteraciones para el cálculo de background (`n_passes`) |
| `TEMPORAL_WINDOW_H`, `TEMPORAL_PIXEL_RAD` | Ventana temporal [h] y radio espacial [píxeles] para el filtrado temporal |

---

## 8. Códigos de salida

### 8.1 `fire_mask` (final, en `fire_mask_p2`)

| Código | Significado |
|---|---|
| *(sentinel "no procesado")* | Píxel que no fue candidato en Parte I, o que fue descartado en Parte II por `cond1`/`cond2`/`cond3` — **ver sección 9.3** |
| 10 | Fuego procesado (solución de Dozier válida) |
| 11 | Fuego saturado |
| 12 | Contaminado por nube |
| 13 | Alta probabilidad |
| 14 | Media probabilidad |
| 15 | Baja probabilidad |
| 30-35 | Igual que 10-15, pero **además** matchea con un fuego detectado en la ventana temporal previa (`+20`) |

### 8.2 `summary.json`

| Campo | Significado |
|---|---|
| `n_candidates` | Tamaño de `candidates` (salida de Parte I) |
| `n_confirmed` | Tamaño de `confirmed` (salida de Parte II) |
| `n_high` / `n_medium` / `n_low` | Conteo de píxeles con código 13/33, 14/34, 15/35 |
| `n_temporal` | `(fire_mask_p2 >= 30).sum()` — ⚠️ ver sección 9.3, actualmente **no** mide lo que su nombre sugiere |
| `frp_median_mw` / `frp_max_mw` | Estadísticos de FRP sobre `confirmed` con `frp > 0` |
| `run_time_s` | Tiempo total de Parte I + Parte II |

---

## 9. Estado actual / issues conocidos

### 9.1 Bug #10 (RESUELTO) — BT físicamente imposibles

Antes: BT7/BT14 llegaban a 431K/861K (imposible). Causa: error de unidades
(`×1e6` aplicado de más sobre `rad7`/`rad14`) y uso de coeficientes Planck
genéricos en vez de los propios de cada `.nc`.

**Fix aplicado:**
- Eliminado el `×1e6` erróneo.
- `BT7`/`BT14`/`BT13`/`BT15` ahora se calculan con
  `planck_temp_from_coeffs(rad, fk1, fk2, bc1, bc2)` usando los coeficientes
  propios de cada banda/escena (leídos vía `load_planck_coeffs()` desde
  `*_planck.json`).
- `rad7/rad14/rad13/rad15` se regeneran con `planck_rad(band, bt)` para
  mantener consistencia con Dozier/background/FRP.

**Resultado verificado:** BT7 ∈ [288.9, 410.9] K, BT14 ∈ [271.9, 339.2] K —
rangos físicamente sensatos.

### 9.2 Fix de `compute_solar_zenith` (RESUELTO)

`lstm = 15*round(hour_utc)` daba SZA de 115-123° de noche cuando en
realidad era pleno día, lo que hacía que Parte I usara umbrales de
**noche** y tardara 5+ minutos. Fix: `lstm = 0.0`. Verificado: SZA ahora en
rango 53.4°-61.0° con 100% de píxeles diurnos para la escena de prueba.

### 9.3 "Con historial (+20): 64955" — métrica engañosa, NO es un bug de filtrado temporal

**Diagnóstico confirmado en esta sesión.** Para la corrida de prueba:

```
67872 (= 224 × 303, total de píxeles)
  - 2917 (confirmados)
  = 64955  ← coincide EXACTO con "Con historial (+20)"
```

El print en `run_fdca.py`:

```python
print(f"  Con historial (+20): {(fire_mask_p2>=30).sum()}")
```

no cuenta píxeles con offset temporal `+20`. Cuenta **todos los píxeles que
`run_part2` nunca sobrescribió** — es decir, todo lo que quedó con el valor
**sentinel** de `fire_mask_p1` (píxeles que no fueron candidatos en Parte I,
más candidatos descartados por `cond1`/`cond2`/`cond3` en Parte II).

`run_part2.py` ya tiene el guard correcto para `prev_fire_mask = None`
(el bloque de filtrado temporal se salta completo, `temporally_filtered`
queda `False` para todos). El filtrado temporal en sí **probablemente
funciona bien**.

**Acción pendiente — verificación rápida:**

```python
vals, counts = np.unique(fire_mask_p2, return_counts=True)
for v, c in zip(vals, counts):
    print(int(v), int(c))
```

Resultado esperado si el diagnóstico es correcto: un único valor grande
(el sentinel, ej. `100` o `255`) con count `64955`, y **nada** en 30-35.

**Fix recomendado** (en `run_fdca.py`):

```python
n_temporal = int(np.isin(fire_mask_p2, [30, 31, 32, 33, 34, 35]).sum())
print(f"  Con historial (+20): {n_temporal}")
```

y aplicar el mismo criterio en `summary.json["n_temporal"]`.

### 9.4 Warnings de `overflow encountered in exp` en `planck.py`/`dozier.py`

Persisten incluso con el guard de `_solve_newton` aplicado, porque la
**mayoría** viene de `_solve_bisection` (que llama a `planck_temp` sin
ningún guard) al evaluar `p` cerca de `DOZIER_P_LOWER`, generando radiancias
de "fire fraction" extremas. No rompen el pipeline (numpy las maneja como
`inf`/`nan` y el guard de Newton corta después), pero ensucian el log.

**Mitigación opcional:** envolver `_solve_bisection`/`temp_diff_sign` con
`np.errstate(over='ignore', invalid='ignore')`.

### 9.5 Sobre-detección: 2917 confirmados vs 116 píxeles NOAA (~25x) — PENDIENTE

El FDCF oficial de NOAA para el mismo timestamp (`20250926_1900`) marcó 116
píxeles de fuego; esta implementación confirma 2917 (1782 de ellos en
`LOW_PROB`/15). Hipótesis de trabajo:

- Umbrales día/noche de Parte I demasiado permisivos para esta escena
  (BT7 llega a 410.9K, cerca de saturación 411.76K — posible incendio real
  grande en zona de quemas de septiembre Uruguay/Argentina).
- Background térmico local subestimado alrededor del foco real, marcando
  píxeles moderadamente cálidos vecinos como anómalos.

**Siguiente paso sugerido:** una vez aplicado el fix de la sección 9.3,
correr `np.unique` sobre `fail_char_p2` restringido a `candidates` para ver
en qué test de Parte I/II se concentran los 27005→2917, y comparar esa
distribución contra la ubicación geográfica de los 116 píxeles NOAA
(¿están dentro del cluster de 2917, o son completamente distintos?).

---

## 10. Tests

Hay una suite de 17 tests sintéticos sobre `dozier.py`/`constants.py` que
valida, entre otras cosas, los fixes de `DOZIER_NEWTON_TOL` (1e-20 → 1e-6),
la convergencia relativa de Newton, y `FRP_MIR_A` (3e-9 → 3e-3). Correr antes
de cualquier cambio en `dozier.py`:

```bash
pytest implementacion/fdca/tests/ -v
```

---



## 11. Subir a Hugging Face

```bash
# Copiar outputs al dataset antes del sync
cp -r data/20250926_1900/    obtencion_imagenes/dataset/uruguay/FDCA-outputs/
cp -r figures/20250926_1900/ obtencion_imagenes/dataset/uruguay/FDCA-figures/

# Usar el sync_hf.py que ya tienen
python obtencion_imagenes/sync_hf.py


```
