# Guía de hiperparámetros y umbral: Decision Tree, Random Forest y XGBoost

Guía práctica para los experimentos de detección de fuego con features por escena.
Para cada modelo dice **qué parámetro configurar, en qué orden y qué esperar**. Los nombres
de flag son los de `decision_tree_features_v1.py` y `random_forest_features_v1.py`.
XGBoost todavía no tiene script; sus parámetros están listados igual para cuando se agregue.

---

## 0. Resumen: qué configurar primero en cada modelo

| Modelo | Tocar primero | Tocar después | Casi no vale la pena |
|---|---|---|---|
| **Decision Tree** | `max-depth`, `min-samples-leaf`, `class-weight` | `ccp-alpha` (alternativa a `max-depth`) | `criterion`, `splitter`, `max-features` |
| **Random Forest** | `min-samples-leaf`, `class-weight`, `max-features` | `max-depth`, `max-samples`, `n-estimators` | `criterion`, `ccp-alpha`, `max-leaf-nodes` |
| **XGBoost** | `learning-rate` + `n-estimators` (con early stopping), `max-depth`, `min-child-weight` | `subsample`, `colsample-bytree`, `scale-pos-weight`, `reg-lambda` | `gamma`, `reg-alpha` |

Idea general: **primero se ajusta la complejidad del modelo** (profundidad, tamaño de hoja) y
**después el punto de operación** (pesos de clase y umbral). El orden de trabajo recomendado es
siempre el mismo: cambiar **un parámetro por vez**, dejar el resto fijo y comparar.

---

## 1. Conceptos que se usan en toda la guía

- **Overfitting**: el modelo memoriza train. Se nota cuando `train_f1` es mucho mayor que el `f1` de test
  (por ejemplo 0.97 contra 0.55). Se combate con modelos más simples: menos profundidad, hojas más grandes.
- **Underfitting**: el modelo es demasiado simple y falla también en train.
- **Punto de operación**: dónde queda el modelo en la curva recall/precisión. Depende del umbral
  (sección 5) y de parámetros que cambian las probabilidades: `class-weight`, `min-samples-leaf`,
  `scale-pos-weight`.
- **AP (average precision)** no depende del umbral: mide qué tan bien **ordena** el modelo los píxeles.
  **recall, precision, f1, sc_meet y FP/scene** sí dependen del umbral: miden **dónde cortó** el modelo.
  Si el AP es bueno pero el recall es malo, el problema suele ser el umbral, no el modelo.
- **Ruido con test chico**: si el test tiene pocos píxeles de fuego (del orden de una decena), un solo
  píxel cambia el recall y el ranking. No hay que sobre-interpretar diferencias chicas.

---

## 2. Decision Tree

Un solo árbol: es interpretable, pero tiende a sobreajustar si crece sin límite.

| Parámetro (flag) | Qué hace | Valores para probar | Efecto |
|---|---|---|---|
| `--max-depth` | Profundidad máxima del árbol | 3, 4, 5, 6, 8, 10 | Más profundo = más complejo y más overfitting. Es el parámetro principal. |
| `--min-samples-leaf` | Mínimo de muestras por hoja | 1, 5, 10, 20, 50 | Más alto = hojas más suaves y menos overfitting. Sube el recall y las falsas alarmas cuando el fuego es raro. |
| `--class-weight` | Pesa las clases | `balanced`, `none` | `balanced` da más peso al fuego (clase rara): sube recall, baja precisión. |
| `--ccp-alpha` | Poda por costo-complejidad | 0.0001, 0.001, 0.01 | Más alto = poda más. Alternativa a limitar `max-depth`: se deja crecer y se poda. |
| `--max-leaf-nodes` | Tope de hojas | 8, 16, 32, 64 | Otra forma de limitar complejidad, más flexible que la profundidad. |
| `--min-impurity-decrease` | Mejora mínima para dividir | 0.0, 0.0001, 0.001 | Más alto = divide menos. Rara vez hace falta. |
| `--criterion` | Medida de impureza | `gini`, `entropy`, `log_loss` | Diferencias mínimas. `entropy` y `log_loss` son equivalentes. |
| `--splitter` | Cómo elige el corte | `best`, `random` | `random` agrega aleatoriedad; casi nunca mejora. |
| `--max-features` | Features por corte | `none` (todas), `sqrt` | En un árbol único conviene dejar todas. |

**Receta para el DT**
1. Fijar `class-weight balanced` y `min-samples-leaf 5`, y barrer `max-depth` en 3, 5, 8, 12.
2. Con la mejor profundidad, barrer `min-samples-leaf` en 1, 5, 20, 50.
3. Si todavía hay overfitting, probar `ccp-alpha` en lugar de (o junto con) `max-depth`.
4. Los demás parámetros se dejan por defecto.

---

## 3. Random Forest

Muchos árboles entrenados con muestras y features al azar, promediados. Reduce mucho el overfitting
del árbol único, por eso los árboles pueden ser más profundos.

| Parámetro (flag) | Qué hace | Valores para probar | Efecto |
|---|---|---|---|
| `--min-samples-leaf` | Mínimo de muestras por hoja | 1, 2, 5, 10, 20 | **La palanca más fuerte del punto de operación.** Más alto: sube el recall y las falsas alarmas, baja el overfitting. |
| `--class-weight` | Pesa las clases | `balanced`, `balanced_subsample`, `none` | `none` suele dar mejor ranking (AP) pero casi no marca nada con umbral 0.5. `balanced_subsample` recalcula los pesos en cada árbol. |
| `--max-features` | Features candidatas por corte | `sqrt`, `log2`, 0.3, 0.5, 1.0 | Más alto = árboles más parecidos entre sí y más lentos. Más bajo = más diversidad. Con pocas features `sqrt` y `log2` pueden dar exactamente lo mismo. |
| `--max-depth` | Profundidad máxima | 8, 12, 20, sin límite | Con hojas de 5 o más, pasar de cierta profundidad no cambia nada. |
| `--max-samples` | Fracción de filas por árbol (con bootstrap) | 0.2, 0.3, 0.5, 0.8 | Más bajo = más diversidad, más rápido y menos memorización de escenas. Útil con píxeles muy correlacionados dentro de una escena. |
| `--n-estimators` | Cantidad de árboles | 100, 300, 500 | Más árboles no sobreajustan; el error se estabiliza. Se ajusta **al final**; el costo crece linealmente. |
| `--no-bootstrap` | Cada árbol usa todo train | flag | Rara vez conviene. Incompatible con `--max-samples` y con `balanced_subsample`. |
| `--criterion`, `--ccp-alpha`, `--max-leaf-nodes`, `--min-impurity-decrease` | Igual que en DT | dejar por defecto | Efecto marginal en un RF. |
| `--n-jobs` | Núcleos | -1 | Solo velocidad; no cambia los resultados. |

**Receta para el RF**
1. Base: `n-estimators 300`, `max-depth 12`, `min-samples-leaf 5`, `class-weight balanced_subsample`, `max-features sqrt`.
2. Barrer `min-samples-leaf` (1, 2, 5, 10, 20) y mirar cómo se mueven recall y falsas alarmas.
3. Comparar `class-weight` (`balanced_subsample`, `balanced`, `none`) mirando **AP** y no solo el recall.
4. Barrer `max-features` (`sqrt`, 0.3, 0.5).
5. Probar `max-samples` (0.3, 0.5) si hay sospecha de overfitting por escena.
6. Subir `n-estimators` al final, solo para estabilizar.

---

## 4. XGBoost

Árboles **chicos y secuenciales**: cada árbol corrige los errores de los anteriores.
Es más sensible a los parámetros que el RF y necesita **early stopping** para saber cuándo parar.
Nombres según la API sklearn de XGBoost (`XGBClassifier`).

| Parámetro | Qué hace | Valores para probar | Efecto |
|---|---|---|---|
| `learning_rate` | Cuánto aporta cada árbol | 0.03, 0.05, 0.1 (default ≈ 0.3) | Más bajo = aprende más lento y generaliza mejor, pero necesita más árboles. |
| `n_estimators` | Cantidad de rondas | 300 a 2000, fijado con early stopping | Va junto con `learning_rate`. Se pone un tope alto y se corta solo. |
| `early_stopping_rounds` | Corta si no mejora | 30 a 100 | Necesita un set de validación. **Debe ser de escenas de train, nunca el test.** |
| `eval_metric` | Métrica de validación | `aucpr` | Con clases desbalanceadas es mejor que `logloss` o `error`. |
| `max_depth` | Profundidad de cada árbol | 3, 4, 6, 8 (default 6) | Árboles chicos; profundo = overfitting. |
| `min_child_weight` | Peso mínimo en una hoja | 1, 5, 10, 20 | Más alto = más conservador. Es el equivalente a `min-samples-leaf`. |
| `subsample` | Fracción de filas por árbol | 0.5, 0.7, 1.0 | Menos = más regularización. |
| `colsample_bytree` | Fracción de features por árbol | 0.5, 0.7, 1.0 | Menos = más regularización. |
| `scale_pos_weight` | Peso de la clase positiva | 1, o negativos/positivos | Sube el recall; distorsiona las probabilidades. Equivale a `class-weight balanced`. |
| `reg_lambda` (L2), `reg_alpha` (L1) | Regularización | lambda 1, 5, 10; alpha 0, 1 | Más alto = modelo más suave. |
| `gamma` | Mejora mínima para dividir | 0, 1, 5 | Más alto = divide menos. Rara vez decisivo. |
| `max_delta_step` | Limita el paso por hoja | 0, 1, 5 | Ayuda con desbalance muy extremo. |
| `tree_method` | Algoritmo de construcción | `hist` | Solo velocidad. |

**Notas**
- XGBoost maneja **NaN de forma nativa**: no hace falta el `SimpleImputer` del pipeline actual.
- `scale_pos_weight` puede combinarse con la elección de umbral, pero conviene usar uno de los dos como
  palanca principal para no doble contar el ajuste.

**Receta para XGBoost**
1. Base: `learning_rate 0.05`, `n_estimators 1000`, `early_stopping_rounds 50`, `eval_metric aucpr`, `max_depth 4`, `min_child_weight 5`, `subsample 0.8`, `colsample_bytree 0.8`.
2. Barrer `max_depth` (3, 4, 6, 8) y `min_child_weight` (1, 5, 10, 20).
3. Barrer `subsample` y `colsample_bytree` (0.5, 0.7, 1.0).
4. Probar `scale_pos_weight` (1 contra negativos/positivos).
5. Al final, bajar `learning_rate` (0.03) para pulir.

---

## 5. El umbral de decisión

### Qué es
Los tres modelos producen una **probabilidad** de fuego por píxel. Para decir "fuego" o "no fuego" hay
que cortar: si `probabilidad ≥ umbral`, el píxel se marca como fuego.
`predict()` usa por defecto **0.5**, que es un valor arbitrario, sobre todo cuando el fuego es raro.

### Qué pasa al moverlo
- **Bajar el umbral** (por ejemplo 0.2): se marcan más píxeles, así que sube el **recall** y suben las falsas alarmas.
- **Subir el umbral** (por ejemplo 0.8): se marcan menos píxeles, así que sube la **precisión** y baja el recall.

El AP no cambia con el umbral. Si el AP es alto pero el recall es bajo, mover el umbral suele ser
más efectivo que seguir cambiando hiperparámetros. En estos experimentos pasó justamente eso: varios
parámetros (`min-samples-leaf`, `class-weight`) solo movían el punto de operación sin mejorar el ranking.

### Cómo elegirlo sin hacer trampa
**Nunca se elige mirando el test.** Si el umbral se ajusta hasta que el test se vea bien, las métricas
de test dejan de ser una estimación honesta. El procedimiento correcto:

1. Con solo las **escenas de train**, obtener probabilidades *out-of-fold*: validación cruzada agrupada
   por escena (`GroupKFold`), para que ninguna escena esté a la vez en entrenamiento y validación.
2. Barrer umbrales sobre esas probabilidades y calcular las métricas por escena.
3. Elegir según el criterio del proyecto. El criterio natural es: **el umbral más alto que todavía cumple
   `scene_recall_target` (0.9)**, es decir, el que tiene menos falsas alarmas entre los que cumplen el recall.
4. Entrenar con todo train, aplicar **ese umbral fijo** al test y reportar. Una sola vez.

### Código de referencia
Se probó con datos sintéticos; hay que adaptarlo a las variables reales del script
(`pipeline`, `train`, `X_train`, `y_train`). Sirve para cualquier modelo sklearn-compatible.

```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

# 1) Probabilidades out-of-fold sobre train, agrupando por escena
oof_proba = cross_val_predict(
    pipeline, X_train, y_train, groups=train["timestamp"],
    cv=GroupKFold(n_splits=5), method="predict_proba",
)[:, 1]


def scene_metrics(df, y, proba, threshold):
    """Recall macro (escenas con fuego) y FP medios por escena a un umbral dado."""
    y = y.to_numpy()
    recalls, false_positives = [], []
    for idx in df.groupby("timestamp").indices.values():
        truth, pred = y[idx], proba[idx] >= threshold
        false_positives.append(int((pred & (truth == 0)).sum()))
        if truth.sum():
            recalls.append(pred[truth == 1].mean())
    return float(np.mean(recalls)), float(np.mean(false_positives))


# 2) Barrido de umbrales
RECALL_TARGET = 0.9
rows = [(t, *scene_metrics(train, y_train, oof_proba, t)) for t in np.linspace(0.05, 0.95, 91)]

# 3) Umbral más alto que cumple el recall objetivo; si ninguno lo cumple, el de mayor recall
ok = [r for r in rows if r[1] >= RECALL_TARGET]
threshold = max(ok, key=lambda r: r[0])[0] if ok else max(rows, key=lambda r: r[1])[0]

# 4) Aplicarlo una sola vez al test
y_pred_test = pipeline.predict_proba(X_test)[:, 1] >= threshold
```

`groupby("timestamp").indices` da posiciones, y funciona porque `train` se arma con
`pd.concat(..., ignore_index=True)` en el script.

### Cuidados
- **Pocos positivos = umbral inestable.** Con pocas escenas de fuego, el umbral elegido puede variar mucho
  entre folds. Conviene mirar la curva completa y no quedarse solo con el "mejor" valor.
- **`class-weight` y umbral hacen trabajos parecidos.** Ambos empujan el recall hacia arriba. Usar los dos
  a la vez es válido, pero el umbral óptimo cambia con cada configuración de pesos: hay que elegirlo
  **después** de fijar el modelo.
- **Cada cambio de modelo requiere re-elegir el umbral.**
- Los scripts actuales **no tienen `--threshold`** todavía; agregarlo implica un cambio en el script.

---

## 6. Diagnóstico: síntoma y qué tocar

| Síntoma | Causa probable | Qué probar |
|---|---|---|
| `train_f1` muy alto, `f1` de test bajo | Overfitting | Bajar `max-depth`, subir `min-samples-leaf` / `min_child_weight`, bajar `max-samples` / `subsample`. |
| Recall bajo, falsas alarmas bajas | Modelo conservador | Bajar el umbral, subir `min-samples-leaf`, usar `class-weight balanced` / `scale_pos_weight`. |
| Muchas falsas alarmas, recall alto | Modelo agresivo | Subir el umbral, `class-weight none`, bajar `min-samples-leaf`. |
| AP bajo | Features o modelo insuficientes | Cambiar de feature set, subir capacidad (`max-depth`) o pasar a RF/XGBoost. El umbral no arregla un ranking malo. |
| AP alto pero recall bajo | Umbral mal puesto | Elegir el umbral con validación agrupada (sección 5). |
| Resultados que cambian mucho entre corridas | Test chico o semilla | Probar varios `random-state` y varios splits antes de decidir. |

---

## 7. Buenas prácticas para no engañarse

1. **Un parámetro por vez**, con el resto fijo; nombres de experimento claros (`rf_hp_leaf2_v1`).
2. **No elegir hiperparámetros ni umbral mirando el test.** Con el test chico esto sobreajusta rápido.
3. **Confirmar en otros splits** (`--split-id` distinto) los 2 o 3 mejores candidatos antes de dar algo por bueno.
4. **Comparar por AP primero** (independiente del umbral) y recién después por recall y falsas alarmas.
5. **Desempatar por simplicidad**: si dos configuraciones rinden igual dentro del ruido, quedarse con la
   más simple (menos features, menos profundidad).