# Definición de métricas de evaluación — para discutir en equipo

## 1. Problema de fondo

El objetivo original (**precision ≥ 95% / recall ≥ 90%**) no es medible con el test actual:
11 positivos y 213.429 negativos. Llegar a 95% de precision con 11 fuegos detectados
significa permitirse menos de 1 falsa alarma en todo el país durante 10 escenas — no es
un objetivo alcanzable ni medible; el intervalo de confianza es más ancho que la
diferencia que se quiere detectar.

**Contexto de referencia — el propio FDCA (algoritmo que estamos aprendiendo) en estas 46 escenas:**

| Variante | Precision | Recall |
|---|---|---|
| Parte II | 5.08% | 91.94% |
| Parte II (sin cód 15) | 24.56% | 87.50% |
| Nuestro mejor árbol hasta ahora | 36% | 82% |

Cualquier objetivo debe calibrarse contra este piso, no contra un número elegido a priori.

---

## 2. La pregunta clave que reformuló todo: ¿qué es una "unidad de acierto"?

Surgió la duda de si las métricas debían calcularse **por píxel** (todos los píxeles de
todas las escenas juntos) o **por escena/timestamp** ("¿hay fuego ahora, en esta imagen
específica?"). Se decidió que **ambos niveles importan, pero por razones distintas**, y
que además **se necesitan coordenadas**, lo cual cambia el enfoque una vez más.

### 2.1 Lo que se estaba midiendo hasta ahora: micro-promedio (píxel, bolsa única)

Se juntan los 213.440 píxeles de las 10 escenas de test en una sola bolsa y se calcula
precision/recall una única vez. Problema: las escenas con más fuego pesan más, y una
escena que falla completamente queda diluida entre las que funcionan bien.

### 2.2 Ejemplo concreto que motivó la discusión (variante contextual, all-pixels)

| Timestamp | Positivos NOAA | TP | FN | FP | ¿Detectó el fuego? |
|---|---|---|---|---|---|
| 20251115_0850 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |
| 20251118_2250 | 3 | 3 | 0 | 3 | sí |
| 20251122_0220 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |
| 20251123_2010 | 2 | 2 | 0 | 2 | sí |
| 20251126_0720 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |
| 20251128_0530 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |
| 20251128_2220 | 5 | 5 | 0 | 6 | sí |
| 20251204_0830 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |
| 20251204_1330 | 1 | 1 | 0 | 16 | sí, pero con 16 falsas alarmas |
| 20251209_0000 | 0 | 0 | 0 | 0 | sin fuego, sin alarma |

El problema de falsas alarmas está concentrado en **una sola escena** (16 de 27 FP
totales). Comparado con `full_v1`: tiene mejor precision agregada, pero **no detectó
absolutamente nada de fuego** en 20251204_1330. El número agregado de precision=0.36 no
muestra esa falla puntual.

### 2.3 Bug detectado: `zero_division=0`

Las 6 escenas sin fuego hoy reportan `recall=0.0`, `precision=0.0`, `f1=0.0`. Esto es un
artefacto: en una escena sin fuego el recall **no está definido**, no es cero. Si se
promedian esas escenas arrastrando ceros falsos, cualquier macro-métrica queda destruida.
**Corrección necesaria:** devolver `NaN` y excluir esas escenas del promedio de recall.

---

## 3. Decisión final: la unidad de evaluación sigue siendo el píxel — pero mal combinado hoy

Tras discutirlo, el criterio real del proyecto es: **"quiero acertar el píxel y dar
coordenadas"**. Esto descarta usar la detección binaria a nivel escena como criterio
principal. Cambia la recomendación inicial: no reemplazar precision/recall por píxel,
sino **combinarlas correctamente entre escenas**.

### 3.1 Fórmula correcta (macro, no micro)

Recall y precision se calculan **a nivel píxel dentro de cada escena**, y luego se
promedian las escenas con **igual peso**:

```
recall_macro = (1 / N_escenas_con_fuego) * Σ_s recall_s
```

Esto es distinto de:
- el micro-promedio original (una bolsa, escenas con más fuego pesan más), y
- la métrica "¿hubo al menos un píxel acertado en la escena?" (demasiado permisiva:
  puede darse por "detectado" con 1 de 20 píxeles de fuego encontrados).

### 3.2 Falsas alarmas por escena

```
FP_por_escena = (1 / N_total) * Σ_s FP_s
```

Es la métrica que un operador entiende ("cuántas alarmas falsas por imagen") y no
depende de cuántos positivos haya. Reportar también percentiles (mediana, máximo), no
solo promedio — el caso malo importa. Ejemplo: 27 FP / 10 escenas = 2.7 FP/escena en
promedio, pero mediana 0 y máximo 16 (una sola escena concentra el problema).

### 3.3 Fracción de escenas que cumplen el objetivo (criterio de aceptación)

Porcentaje de escenas con fuego donde `recall_s >= 0.9`. Con solo 4 escenas con fuego en
test hoy, esta métrica solo puede tomar los valores 0%, 25%, 50%, 75%, 100% — cualquier
número que se reporte hoy es más una anécdota que un resultado confiable.

---

## 4. Por qué hacen falta coordenadas: nivel foco/cluster, no solo píxel suelto

Recall/precision por píxel solo no alcanza si el objetivo final es **localizar** el
incendio, no solo clasificar píxeles sueltos. Un fuego real casi nunca es un único píxel
aislado — son varios píxeles contiguos.

Dos preguntas abiertas, no resueltas todavía:

1. **¿Cómo se define "acerté" a nivel de foco de fuego (no de píxel)?**
   Agrupar píxeles de fuego contiguos (real y predicho) en clusters/blobs, y considerar
   un cluster real "detectado" si tiene overlap espacial suficiente con algún cluster
   predicho. Con esto, un modelo puede tener recall=0.7 a nivel píxel pero detectar el
   100% de los focos reales si predice "menos denso" alrededor del centro correcto.

2. **¿Qué error de localización es aceptable?**
   Los GOES-ABI usados tienen resolución de ~2 km en el canal de fuego. Un desplazamiento
   de 1 píxel entre lo predicho y lo real puede ser ruido de georreferenciación, no un
   error real del modelo. Hay que decidir el umbral de tolerancia antes de penalizar esos
   casos como FP+FN.

### Propuesta concreta a sumar al pipeline de métricas

- Recall/precision a nivel píxel, **macro-promediado por escena** (no micro).
- Métrica de **detección de foco (cluster-level)**: agrupar píxeles conectados y medir
  qué fracción de los focos reales tiene overlap suficiente con algún cluster predicho.
- **Distancia centroide predicho vs. centroide real** de cada foco detectado (en píxeles
  o km), como métrica descriptiva aparte — no como criterio de pass/fail.
- τ (umbral de decisión) debe ser **el mismo** en todas las métricas y quedar congelado
  en validación antes de evaluar test (ver §6).

---

## 5. Consecuencia sobre el dimensionamiento del dataset

Si la unidad de evaluación relevante es la **escena** (para las métricas macro y de
aceptación), la potencia estadística depende del **número de escenas con fuego**, no del
número de píxeles de fuego.

Hoy: 4 escenas con fuego en test. Cada una vale el 25% de cualquier macro-métrica — no se
puede distinguir un modelo con recall 0.90 de uno con 0.75 con esa resolución. Esto
también explica por qué un barrido de `min_samples_leaf` dio seis matrices de confusión
idénticas: no es que los hiperparámetros no importen, es que el test no tiene resolución
para detectar la diferencia.

**Reformulación de la meta del dataset (foco en escenas, no solo en positivos totales):**

| Fase | Escenas con fuego | Escenas sin fuego | Resolución esperada |
|---|---|---|---|
| B | ~40 | ~40 | ±15% |
| C | ~150 | ~150 | ±5% |

Nota: se necesitan escenas **sin** fuego en cantidad comparable, porque son las únicas
que miden falsas alarmas. Hoy hay solo 6 y aportan 0 FP, lo que hace parecer la tasa de
falsas alarmas mejor de lo que probablemente es en la realidad.

---

## 6. Cambios concretos propuestos al runner/pipeline

- [ ] Devolver `NaN` en lugar de `0.0` para métricas indefinidas por escena (recall en
      escenas sin fuego).
- [ ] Agregar las métricas de la sección 3 (recall macro, FP/escena, fracción de escenas
      que cumplen objetivo), tanto local como en Comet.
- [ ] Agregar métrica de detección de foco (cluster-level) y distancia de centroide,
      sección 4.
- [ ] Reportar tabla por escena en el output, con TP/FN/FP absolutos.
- [ ] Separar el resumen en escenas con fuego y escenas sin fuego.
- [ ] Fijar un único τ (umbral) para todas las métricas, elegido en validación y
      congelado antes de evaluar test — no usar `predict()` con 0.5 implícito.
- [ ] Etiquetar como "exploratorios" los experimentos ya corridos en Comet con el dataset
      actual (46 escenas / 4 con fuego): sirven para comparación relativa, no para
      números finales reportables.

---

## 7. Preguntas abiertas para discutir con el equipo

1. ¿Qué overlap mínimo entre cluster real y cluster predicho se considera "detección
   válida" a nivel foco?
2. ¿Qué distancia de centroide (en píxeles o km) se considera aceptable dado el error de
   georreferenciación esperado de GOES-ABI?
3. ¿El criterio de éxito final del proyecto pondera más la detección de eventos (alarma
   sí/no por escena) o la localización precisa (coordenadas)? La respuesta ya dada fue
   "acertar el píxel y dar coordenadas" — esto debería quedar documentado como decisión
   de diseño, no solo como conversación.
4. Dado el atraso general del cronograma, ¿el dataset Fase B (~40+40 escenas) es
   alcanzable en el tiempo que queda, o hay que definir un criterio de aceptación
   provisorio con el dataset actual mientras se amplía?