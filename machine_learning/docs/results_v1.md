# Interpretación de las métricas de evaluación

Para entender las métricas, primero hay que definir la **matriz de confusión**. En este problema:

* **Clase 0:** píxel sin fuego.
* **Clase 1:** píxel con fuego.

Con `labels=[0, 1]`, la matriz tiene esta forma:

|            | Predijo 0 | Predijo 1 |
| ---------- | --------: | --------: |
| **Real 0** |        TN |        FP |
| **Real 1** |        FN |        TP |

Donde:

* **TN — True Negative:** píxel sin fuego clasificado correctamente.
* **FP — False Positive:** falsa alarma; píxel sin fuego clasificado como fuego.
* **FN — False Negative:** fuego no detectado.
* **TP — True Positive:** fuego detectado correctamente.

En el test tenemos:

* **2.372 píxeles**
* **11 positivos**
* **2.361 negativos**

Por eso las métricas deben interpretarse con cuidado: hay muchísimos más negativos que positivos.

---

## Accuracy

La **accuracy** es la proporción total de predicciones correctas:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Mide qué porcentaje de todos los píxeles fue clasificado correctamente.

Por ejemplo, para la variante con `min_samples_leaf=10`, aproximadamente:

```text
TN = 2346
FP = 15
FN = 0
TP = 11
```

Entonces:

$$
Accuracy = \frac{2346 + 11}{2372} = 0.9937
$$

Es decir, el modelo acierta el **99,37% de los píxeles**.

### Problema de la accuracy

Como hay muy pocos fuegos, un modelo que predijera todo como negativo obtendría:

$$
Accuracy = \frac{2361}{2372} = 0.9954
$$

Tendría **99,54% de accuracy**, aunque no detectaría ningún incendio:

```text
Recall = 0
```

Por eso, en este problema la **accuracy sola no es suficiente**.

---

## Recall

El **recall**, también llamado sensibilidad o tasa de verdaderos positivos, mide qué proporción de los fuegos reales fue detectada:

$$
Recall = \frac{TP}{TP + FN}
$$

Para `min_samples_leaf=10`:

$$
Recall = \frac{11}{11 + 0} = 1.0
$$

El modelo detectó los **11 positivos** del conjunto de test.

El recall es especialmente importante cuando perder un incendio es peor que generar una falsa alarma.

---

## Precision

La **precision** mide qué proporción de las predicciones positivas realmente era fuego:

$$
Precision = \frac{TP}{TP + FP}
$$

Con el ejemplo:

```text
TP = 11
FP = 15
```

El modelo predijo 26 píxeles como fuego, pero sólo 11 eran realmente fuego:

$$
Precision = \frac{11}{11 + 15} = 0.4231
$$

Es decir, aproximadamente el **42,31% de las alarmas fueron correctas**.

La precision responde a la pregunta:

> Cuando el modelo dice "fuego", ¿qué tan frecuentemente tiene razón?

---

## Precision versus recall

Hay un compromiso habitual entre ambas métricas:

* Si el modelo intenta detectar todos los fuegos, puede generar más falsas alarmas.
* Si el modelo es muy conservador, puede tener buena precision pero perder fuegos.

Por ejemplo, el modelo sin `class_weight=balanced` obtuvo:

```text
precision = 1.0000
recall    = 0.1818
```

Eso significa que las pocas alarmas que produjo fueron correctas, pero sólo encontró **2 de los 11 fuegos**.

Para nuestro caso, no es un buen comportamiento.

---

## Balanced accuracy

La **balanced accuracy** calcula el promedio del desempeño sobre ambas clases:

$$
BalancedAccuracy =
\frac{Recall_{positivo} + Recall_{negativo}}{2}
$$

El recall de la clase negativa es la **especificidad**:

$$
Specificity =
\frac{TN}{TN + FP}
$$

Por lo tanto, en un problema binario:

$$
BalancedAccuracy =
\frac{
\frac{TP}{TP+FN}
+
\frac{TN}{TN+FP}
}{2}
$$

Para el ejemplo:

$$
Recall_{positivo} =
\frac{11}{11+0}=1
$$

y:

$$
Recall_{negativo} =
\frac{2346}{2346+15}
\approx 0.9936
$$

Entonces:

$$
BalancedAccuracy =
\frac{1 + 0.9936}{2}
\approx 0.9968
$$

---

## Accuracy versus balanced accuracy

La diferencia principal es:

| Métrica               | Interpretación                                        |
| --------------------- | ----------------------------------------------------- |
| **Accuracy**          | Pesa cada píxel por igual.                            |
| **Balanced accuracy** | Da el mismo peso a la clase positiva y a la negativa. |

La balanced accuracy es más útil cuando las clases están desbalanceadas, como en nuestro dataset.

Un modelo que predice todo negativo tendría:

```text
accuracy           = 0.9954
balanced_accuracy  = 0.5000
```

¿Por qué?

```text
recall positivo = 0
recall negativo = 1

balanced accuracy = (0 + 1) / 2
                  = 0.5
```

La balanced accuracy deja en evidencia que el modelo **ignora completamente la clase fuego**.

---

## F1-score

El **F1-score** combina precision y recall mediante la media armónica:

$$
F1 =
2 \cdot
\frac{Precision \cdot Recall}
{Precision + Recall}
$$

Para el modelo con `min_samples_leaf=10`:

```text
precision = 0.4231
recall    = 1.0000
```

Entonces:

$$
F1 =
2 \cdot
\frac{0.4231 \cdot 1}
{0.4231 + 1}
\approx 0.5946
$$

La media armónica penaliza los casos en los que una métrica es alta y la otra baja.

Por ejemplo:

```text
precision = 1.0
recall    = 0.1818
```

produce:

```text
F1 = 0.3077
```

Esto refleja que detectar sólo una pequeña parte de los incendios no es suficiente, aunque las alarmas producidas sean correctas.

---

## Average Precision

La **average precision (AP)** es diferente de la precision común.

### Precision común

La precision que estamos reportando se calcula sobre las predicciones finales:

```python
y_pred = pipeline.predict(X_test)
```

Es decir, el árbol asigna a cada píxel una clase final:

```text
0 o 1
```

Luego calcula:

$$
Precision = \frac{TP}{TP+FP}
$$

Es una medida correspondiente a un **único punto de decisión**.

### Average Precision

Para calcular average precision, el runner utiliza las probabilidades:

```python
y_probability = pipeline.predict_proba(X_test)[:, 1]
```

Cada píxel recibe un score, por ejemplo:

```text
0.99
0.87
0.62
0.41
0.18
0.03
```

Después se prueban distintos umbrales sobre esos scores. Para cada umbral se obtiene una pareja:

```text
precision, recall
```

Esto genera una **curva precision-recall**.

La average precision resume el área bajo esa curva:

$$
AP =
\sum_n
(R_n - R_{n-1})P_n
$$

donde:

* $P_n$ es la precision en un punto de la curva.
* $R_n$ es el recall en ese punto.
* $R_n-R_{n-1}$ es cuánto aumentó el recall.

En términos simples:

> La average precision evalúa qué tan bien ordena el modelo los píxeles sospechosos de fuego a través de todos los umbrales posibles.

---

## Precision versus Average Precision

| Métrica               | Qué usa                 | Qué evalúa                                                                        |
| --------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| **Precision**         | Clases finales `0/1`    | Qué proporción de las alarmas actuales es correcta.                               |
| **Average Precision** | Probabilidades o scores | Qué tan bien ordena los positivos frente a los negativos para todos los umbrales. |

Una forma de verlo:

**Precision:**

> "Usando el umbral actual, ¿qué porcentaje de alarmas es correcto?"

**Average precision:**

> "Si cambio el umbral, ¿qué tan buena es la relación precision-recall en general?"

Por eso pueden diferir bastante.

Por ejemplo, el modelo sin ponderación obtuvo:

```text
precision           = 1.0000
recall              = 0.1818
average_precision   = 0.7071
```

Esto puede pasar porque:

* Con el umbral usado por `predict`, sólo clasifica como fuego unos pocos píxeles.
* Esos pocos píxeles son correctos, por eso la precision es 1.
* Pero si se consideran otros umbrales sobre las probabilidades, el modelo puede ordenar razonablemente bien los positivos, por eso la AP es relativamente alta.

Sin embargo, para operar el modelo con el umbral actual, ese modelo no sirve mucho porque pierde la mayoría de los fuegos.

---

# Cómo leer las métricas en nuestro caso

Para detección de incendios, normalmente nos interesan principalmente:

* **Recall:** no perder incendios.
* **Precision:** evitar demasiadas falsas alarmas.
* **F1:** equilibrio entre recall y precision.
* **Balanced accuracy:** comprobar que el modelo no sólo aprende la clase negativa.
* **Average precision:** evaluar la calidad de los scores y del ranking, especialmente si luego queremos modificar el umbral.

La configuración `min_samples_leaf=10` quedó bien balanceada:

```text
accuracy           = 0.9937
balanced accuracy  = 0.9968
precision           = 0.4231
recall              = 1.0000
F1                  = 0.5946
average precision   = 0.6818
```

La interpretación sería:

* Detectó **todos los positivos** del test.
* Generó **15 falsas alarmas**.
* Aproximadamente **42% de sus alarmas fueron fuegos reales**.
* El desempeño está equilibrado entre positivos y negativos.
* Los scores tienen una buena separación general entre fuegos y no fuegos.

Igualmente, el test sólo contiene **11 positivos**. Por lo tanto, cada píxel cambia bastante las métricas: perder un solo fuego haría que el recall pasara de:

$$
1.0 \rightarrow \frac{10}{11}=0.9091
$$

Por eso conviene utilizar estas métricas como una **primera comparación entre configuraciones** y luego validar el comportamiento con más escenas o con varios splits.

```
```
