# Guía para principiantes: cómo crear y registrar experimentos en Comet

Esta guía está pensada para alguien que nunca usó Comet y quiere empezar con un experimento muy simple, sin juntarse con datos reales ni con una infraestructura complicada.

La idea es: crear un script, entrenar un modelo pequeño, registrar parámetros y métricas, y ver todo ordenado en Comet.

---

## 1. Qué es Comet

Comet es una plataforma para registrar experimentos de machine learning.

Te permite guardar:
- nombre del experimento
- parámetros del modelo
- métricas (accuracy, loss, F1, etc.)
- tags para organizar runs
- archivos y artefactos
- gráficos o resultados

En este proyecto ya está integrado para que quede muy simple usarlo.

---

## 2. Qué necesitas antes de empezar

Necesitas:
- tener activado el entorno virtual del proyecto
- tener instalada la dependencia de Comet
- tener tu API key de Comet
- tener un workspace correcto

Puedes ver esto en el archivo [.env](../../.env), por ejemplo:

```dotenv
COMET_ENABLED=true
COMET_API_KEY="tu_api_key_real"
COMET_PROJECT_NAME="GERIS-PFC"
COMET_WORKSPACE="labs"
```

> Si `COMET_ENABLED` está en `true`, el script intenta subir el experimento.

---

## 3. Activar el entorno

Desde la raíz del proyecto:

```bash
cd /Users/valentinachagas/GERIS-PFC
source geris/bin/activate
```

Si todavía no tienes paquetes instalados:

```bash
python -m pip install -r requirements.txt
```

---

## 4. Verificar que el proyecto ya tiene un ejemplo listo

En el proyecto hay un script de ejemplo muy sencillo:

```bash
python machine_learning/experiments/decision_tree_simple_comet.py --experiment-name "toy-decision-tree-v1" --tags "toy,baseline"
```

Ese script:
- crea un dataset sintético pequeño
- entrena un `DecisionTreeClassifier`
- calcula métricas
- las sube a Comet

Si funciona, ya tenés el flujo básico listo para repetirlo con otros modelos o parámetros.

---

## 5. Entender el flujo básico de un experimento

Un experimento en ML normalmente tiene esta estructura:

1. definir el problema
2. preparar los datos
3. elegir el modelo
4. elegir hiperparámetros
5. entrenar
6. evaluar
7. loguear todo en Comet

En este proyecto, la parte de logging está centralizada en:

- [machine_learning/tracking.py](../tracking.py)

Ese archivo define:
- un tracker vacío para cuando Comet está apagado
- una integración con Comet cuando está activado
- funciones para guardar parámetros y métricas

---

## 6. Cómo nombrar experimentos

Cada vez que corrés un experimento, conviene darle un nombre claro.

Buenas prácticas:
- `toy-decision-tree-v1`
- `baseline-tree-depth3`
- `tree-maxdepth5-randomstate42`
- `fire-model-exp-001`

Puedes ponerlo al correr el script:

```bash
python machine_learning/experiments/decision_tree_simple_comet.py --experiment-name "baseline-tree-depth3" --tags "baseline,tree"
```

El nombre te ayuda a distinguir runs en Comet.

---

## 7. Cómo organizar experimentos con tags

Los tags son palabras clave para agrupar runs.

Por ejemplo:

```bash
--tags "toy,baseline,tree"
```

o

```bash
--tags "debug,maxdepth3"
```

Los tags te permiten:
- filtrar experimentos
- comparar distintos modelos
- separar pruebas rápidas de pruebas serias

---

## 8. Qué registrar en un experimento

Lo más útil es guardar:

### Parámetros
- nombre del modelo
- profundidad máxima
- random_state
- tamaño del dataset
- número de features
- porcentaje de test

### Métricas
- `accuracy`
- `precision`
- `recall`
- `f1`

### Extras
- nombre del experiment
- tags
- dataset usado
- fecha o versión

En el script de ejemplo, esto ya está hecho:

```python
tracker.log_parameters({
    "model": "DecisionTreeClassifier",
    "experiment_name": experiment_name,
    "random_state": cfg.random_state,
    "max_depth": cfg.max_depth,
    "dataset_kind": "toy_synthetic_fire_detection",
})

tracker.log_metrics({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
}, step=1)
```

Eso es exactamente lo que Comet visualiza.

---

## 9. Cómo hacer un nuevo experimento

Para crear otro experimento, haz esto:

1. copia el script base
2. cambia los hiperparámetros
3. cambia el nombre del run
4. cambia los tags
5. corrés el script

Ejemplo:

```bash
python machine_learning/experiments/decision_tree_simple_comet.py \
  --experiment-name "tree-depth-5" \
  --tags "baseline,tree,depth5"
```

Luego otro:

```bash
python machine_learning/experiments/decision_tree_simple_comet.py \
  --experiment-name "tree-depth-3" \
  --tags "baseline,tree,depth3"
```

Así comparás distintos modelos en la UI de Comet.

---

## 10. Cómo correr sin cambiar el .env cada vez

Lo ideal es dejar fijo en [.env](../../.env) lo que no cambia:

```dotenv
COMET_ENABLED=true
COMET_API_KEY="tu_api_key_real"
COMET_PROJECT_NAME="GERIS-PFC"
COMET_WORKSPACE="labs"
```

Y para cada run solo cambias:
- `--experiment-name`
- `--tags`

Ejemplo:

```bash
python machine_learning/experiments/decision_tree_simple_comet.py --experiment-name "toy-run-02" --tags "toy,quick-check"
```

---

## 11. Si algo sale mal

### El script no sube nada
Revisá:
- que `COMET_ENABLED=true` esté en el `.env`
- que la API key sea válida
- que el workspace exista
- que el proyecto tenga nombre

### El experimento no se ve
Revisá:
- que el script terminó correctamente
- que tu comando fue ejecutado desde la raíz del proyecto
- que el entorno virtual está activo

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
