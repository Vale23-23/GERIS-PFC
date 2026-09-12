# Guía rápida para experimentos reproducibles con Decision Tree

Esta guía resume el flujo que estamos usando en el proyecto para experimentar en forma ordenada y reproducible, sin depender de `train_test_split` aleatorio cada vez ni de cambiar la configuración a mano en cada corrida.

La idea central es esta:
- guardar splits fijos una sola vez
- definir feature sets en YAML
- correr experimentos con nombres claros y tags útiles
- registrar todo en Comet de forma consistente

---

## 1. Qué hace este flujo

El proyecto ya tiene una base para experimentos reales de clasificación tabular con árboles de decisión.

Incluye:
- generación de splits fijos con IDs de fila
- definición de feature sets por nombre
- configuración de cada experimento en YAML
- logging simple y reusable a Comet

Los archivos principales son:
- [machine_learning/experiments/generate_fixed_splits.py](../experiments/generate_fixed_splits.py)
- [machine_learning/experiments/decision_tree_real_data.py](../experiments/decision_tree_real_data.py)
- [machine_learning/configs/feature_sets.yaml](../configs/feature_sets.yaml)
- [machine_learning/configs/experiments/dt_baseline_v1.yaml](../configs/experiments/dt_baseline_v1.yaml)
- [machine_learning/tracking.py](../tracking.py)

---

## 2. Entorno y Comet

Lo ideal es dejar en [.env](../../.env) solo lo que no cambia entre corridas:

```dotenv
COMET_ENABLED=true
COMET_API_KEY="tu_api_key_real"
COMET_PROJECT_NAME="GERIS-PFC"
COMET_WORKSPACE="labs"
```

Con eso no hace falta tocar el `.env` cada vez que quieras correr un experimento nuevo.

Solo cambias:
- `--experiment-name`
- `--tags`
- la config del run o el feature set

---

## 3. Activar el entorno

Desde la raíz del proyecto:

```bash
cd /Users/valentinachagas/GERIS-PFC
source geris/bin/activate
```

Si hace falta instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

---

## 4. Generar splits fijos

La primera vez que tenés un dataset, lo más importante es generar los splits una sola vez y guardar los row IDs.

```bash
python machine_learning/experiments/generate_fixed_splits.py \
  --csv /ruta/al/dataset.csv \
  --target target \
  --output-dir machine_learning/splits
```

Esto crea:
- `train_ids.csv`
- `test_ids.csv`

Con eso ya no dependés de un split aleatorio cada vez.

---

## 5. Definir feature sets en YAML

En [machine_learning/configs/feature_sets.yaml](../configs/feature_sets.yaml) se registran los nombres de los feature sets y las columnas que usan.

Ejemplo:

```yaml
goes_baseline:
  - b07
  - b14
  - b07_minus_b14

full_v1:
  - feature_a
  - feature_b
  - feature_c
```

Así después podés correr con un nombre legible como:

```bash
--feature-set goes_baseline
```

o incluso combinar varios sets con `+`:

```bash
--feature-set goes_baseline+contextuales_tabular
```

---

## 6. Configurar cada experimento en YAML

El archivo [machine_learning/configs/experiments/dt_baseline_v1.yaml](../configs/experiments/dt_baseline_v1.yaml) sirve para dejar fijo todo lo que define el experimento:

```yaml
model: DecisionTreeClassifier
feature_set: radiometria_directa
target: target
random_state: 42
max_depth: 6
min_samples_leaf: 5
train_test_split:
  test_size: 0.25
  stratify: true
experiment_name: dt_baseline_v1
tags:
  - baseline
  - fixed_split
  - yaml_config
```

Esto ayuda a que cada comparación tenga la misma lógica y sea reproducible.

---

## 7. Ejecutar un entrenamiento con split fijo

Ahora el pipeline recomendado es:

```bash
python machine_learning/experiments/decision_tree_real_data.py \
  --csv /ruta/al/dataset.csv \
  --target target \
  --config machine_learning/configs/feature_sets.yaml \
  --experiment-config machine_learning/configs/experiments/dt_baseline_v1.yaml \
  --train-split machine_learning/splits/train_ids.csv \
  --test-split machine_learning/splits/test_ids.csv \
  --no-comet
```

Con esto se usan:
- un split fijo guardado en disco
- un feature set llamado por nombre
- hiperparámetros definidos por YAML
- logging de parámetros y métricas en Comet si está habilitado

---

## 8. Cómo nombrar experimentos

La convención recomendada es algo así:

```text
dt_{feature_set}_{split_date}_{version}
```

Por ejemplo:
- `dt_goes_baseline_20251115_v1`
- `dt_radiometria_directa_20251115_v2`
- `dt_full_v1_test42_v3`

También podés usar tags para agrupar:

```bash
--tags "baseline,tree,fixed-split,goes"
```

Esto te permite filtrar runs por set de features, fecha del split o tipo de validación.

---

## 9. Qué se registra en Comet

El script registra normalmente:
- nombre del modelo
- dataset usado
- feature set elegido
- columnas usadas
- target
- tamaño train/test
- random_state
- max_depth
- min_samples_leaf
- métricas: accuracy, precision, recall, F1

Eso permite comparar corridas distintas sin depender de la memoria.

---

## 10. Flujo recomendado para empezar

1. preparar CSV limpio con columnas de features y target
2. generar split fijo
3. definir feature sets en YAML
4. crear una config del experimento
5. correr la experimentación con `decision_tree_real_data.py`
6. comparar resultados en Comet por feature set y split

Esto es mucho más robusto que hacer un `train_test_split` aleatorio desde el script cada vez.

---

## 11. Recomendación práctica

Para trabajo serio, el patrón ideal es:
- un split por dataset o fecha
- un YAML por feature set
- un YAML por versión del experimento
- a lo sumo un par de flags por corrida, como `--experiment-name` y `--tags`

Así la comparación de resultados se vuelve consistente y mucho más fácil de interpretar.

---

## 12. Si aparece un problema

### El split no se ve reproducible
Revisá que estés usando `--train-split` y `--test-split` con los CSV generados previamente.

### El feature set falla
Revisá que las columnas existan en el CSV y que estén escritas exactamente igual que en [machine_learning/configs/feature_sets.yaml](../configs/feature_sets.yaml).

### Comet no sube el run
Revisá que `COMET_ENABLED` esté en `true` y que la API key del `.env` sea válida.

### El run no imprime métricas esperadas
Chequear si el CSV tiene clases balanceadas o si el target tiene pocos ejemplos; si hace falta, conviene revisar el dataset antes de comparar modelos.

---

## 13. Plantilla lista para copiar

Si querés arrancar un nuevo experimento sin reinventar la estructura, podés copiar esta plantilla y dejarla en la carpeta de configs:

```yaml
# machine_learning/configs/experiments/dt_variant_01.yaml
model: DecisionTreeClassifier
feature_set: radiometria_directa
target: target
random_state: 42
max_depth: 6
min_samples_leaf: 5
train_test_split:
  test_size: 0.25
  stratify: true
experiment_name: dt_variant_01
tags:
  - baseline
  - decision_tree
  - fixed_split
```

Luego lo corrés así:

```bash
python machine_learning/experiments/decision_tree_real_data.py \
  --csv /ruta/al/dataset.csv \
  --target target \
  --config machine_learning/configs/feature_sets.yaml \
  --experiment-config machine_learning/configs/experiments/dt_variant_01.yaml \
  --train-split machine_learning/splits/train_ids.csv \
  --test-split machine_learning/splits/test_ids.csv \
  --experiment-name "dt_variant_01" \
  --tags "baseline,decision_tree,fixed_split" \
  --no-comet
```

Regla simple:
- cambiás solo `feature_set`, `max_depth`, `min_samples_leaf`, `experiment_name` y tags
- el split fijo se mantiene igual
- la comparación entre corridas se vuelve limpia y reproducible

---

## 14. Resumen corto

La filosofía del proyecto ya no es “correr un árbol con un dataset chiquito y listo”, sino:
- dejar fija la partición de datos
- declarar los feature sets en YAML
- versionar cada experiencia
- comparar resultados de forma ordenada

Eso hace que cada run sea mucho más útil que una prueba aislada.

### Quiero probar algo sin riesgo
Usa siempre el script toy de ejemplo primero.

Ese script no necesita dataset real ni infraestructura pesada.

---

## 12. Buenas prácticas para principiantes

- no cambies el `.env` cada vez que querés correr un experimento diferente
- usa nombres claros
- usa tags para agrupar
- guarda siempre parámetros y métricas
- prueba primero con un toy dataset
- compará varias configuraciones sin miedo

---

## 13. Flujo recomendado para empezar

1. activar entorno
2. revisar `.env`
3. correr script toy
4. mirar Comet en el navegador
5. cambiar un parámetro de modelo
6. correr otra vez con otro `--experiment-name`
7. comparar resultados en Comet

---

## 14. Resumen rápido

Si querés empezar de cero, basta con esto:

```bash
cd /Users/valentinachagas/GERIS-PFC
source geris/bin/activate
python machine_learning/experiments/decision_tree_simple_comet.py --experiment-name "toy-decision-tree-v1" --tags "toy,baseline"
```

Eso ya crea un run en Comet, guarda parámetros y métricas, y te sirve como base para experimentar con más modelos.

---

## 15. Siguiente paso recomendado

Cuando quieras, puedes pasar a un segundo paso más útil:
- crear un script para comparar varios valores de `max_depth`
- registrar cada una en Comet como un run separado
- comparar resultados en una sola vista
- decidir cuál configuración funciona mejor

Eso es exactamente lo que suele hacerse en una etapa inicial de experimentación con árboles de decisión.
