````markdown
# Descripción del modelo de aprendizaje automático

## 1. Inputs del GOES

### 1.1 Bandas del ABI

Las variables de entrada del modelo se obtienen principalmente a partir de las observaciones del instrumento ABI (Advanced Baseline Imager) a bordo del satélite GOES. Se consideran tanto mediciones radiométricas directas como variables derivadas y estadísticas del entorno espacial de cada píxel.

Las variables seleccionadas son:

- **Banda 7 (`B07`)**: temperatura de brillo correspondiente a la banda infrarroja de onda corta, utilizada como una de las principales fuentes de información para la detección de anomalías térmicas.

- **Banda 14 (`B14`)**: temperatura de brillo correspondiente al infrarrojo térmico. Es una de las bandas principales utilizadas por el algoritmo de detección de incendios de la NOAA.

- **Banda 15 (`B15`)**: temperatura de brillo utilizada, entre otras funciones, para la identificación de nubes opacas.

- **Albedo y reflectividad**: se utilizan variables asociadas a la reflectividad de la superficie y a productos derivados de la banda de onda corta.

- **Diferencia entre bandas**:

  $$
  \Delta T_{7-14}=T_7-T_{14}
  $$

  Esta variable permite caracterizar diferencias térmicas entre las bandas 7 y 14.

- **Información espacial del entorno**: el algoritmo de la NOAA calcula estadísticas sobre una región alrededor de cada píxel mediante ventanas de fondo. Estas estadísticas son incorporadas como variables de entrada del modelo.

Además de las mediciones radiométricas, se incorporan variables geométricas y auxiliares:

- **SZA**: ángulo cenital solar.
- **LZA**: ángulo cenital de observación del satélite.
- **Glint angle**: ángulo asociado al resplandor solar.
- **TPW**: agua precipitable total.
- **Emisividad superficial**: emisividades correspondientes a las bandas 7 y 14.
- **Tipo de ecosistema**: código de ecosistema o cobertura de suelo asociado al píxel.

### 1.2 Correcciones radiométricas

También se consideran las temperaturas corregidas utilizadas durante el procesamiento del algoritmo:

- **`bt7_corr`**: temperatura de brillo corregida de la banda 7.
- **`bt14_corr`**: temperatura de brillo corregida de la banda 14.

Estas correcciones incorporan información asociada a factores físicos como la transmitancia atmosférica, la emisividad superficial y el agua precipitable total.

Debe tenerse en cuenta que estas variables pueden presentar valores faltantes en determinadas situaciones, por ejemplo cuando el píxel se encuentra saturado o cuando el procesamiento supera el máximo de iteraciones. Estos valores faltantes no necesariamente representan una ausencia aleatoria de información, ya que la saturación puede ser en sí misma un indicio de temperaturas extremadamente altas.

### 1.3 Máscara de fuego de la NOAA

Además de la detección de incendios, la máscara producida por el algoritmo de la NOAA contiene información sobre diferentes estados de los píxeles, incluyendo píxeles procesados, incendios de distintas probabilidades, nubes, agua y píxeles no procesados.

La máscara se utiliza como referencia para construir el *target* del modelo. En particular, se define una variable binaria `ref_fire`, que indica si el píxel corresponde a una categoría considerada como fuego según los códigos de la máscara de referencia.

---

## 2. Variables contextuales

El algoritmo de la NOAA no funciona estrictamente de manera píxel-a-píxel, sino que utiliza información estadística del entorno de cada píxel. Por este motivo, las estadísticas de fondo constituyen una parte importante de las entradas del modelo tabular.

Para cada píxel se consideran, entre otras, las siguientes variables:

- Media y desviación estándar de la temperatura de brillo de la banda 7:

  $$
  \mathrm{bt7\_bkg}, \qquad
  \mathrm{bt7\_bkg\_std}
  $$

- Media y desviación estándar de la temperatura de brillo de la banda 14:

  $$
  \mathrm{bt14\_bkg}, \qquad
  \mathrm{bt14\_bkg\_std}
  $$

- Estadísticas del fondo de reflectividad:

  $$
  \mathrm{albedo\_bkg}, \qquad
  \mathrm{reflb}, \qquad
  \mathrm{std\_reflb}
  $$

- Diferencia entre el píxel y su fondo:

  $$
  T_7-T_{7,\mathrm{bkg}},
  \qquad
  T_{14}-T_{14,\mathrm{bkg}}
  $$

  y la diferencia entre la reflectividad del píxel y la reflectividad de fondo.

- **Número de pasadas de la ventana de fondo (`n_passes`)**: cantidad de expansiones realizadas durante la búsqueda de un fondo válido.

- **Desviación asociada a la diferencia radiométrica (`rad_diff_sigma`)**: medida de dispersión asociada a la diferencia entre las bandas térmicas.

Estas variables permiten que el modelo tenga acceso no solamente a la temperatura absoluta del píxel, sino también a qué tan diferente es el píxel respecto de su entorno.

---

## 3. Selección del enfoque de arquitectura

Existen dos alternativas principales para procesar la información contextual.

### 3.1 Modelo tabular píxel a píxel

En primera instancia se utilizará un modelo basado en árboles de decisión, utilizando algoritmos como XGBoost, LightGBM o CatBoost.

Cada muestra corresponde a un píxel y se representa mediante un vector de características que contiene sus mediciones radiométricas, variables geométricas, variables auxiliares y estadísticas previamente calculadas sobre su entorno.

Este enfoque presenta como principales ventajas un bajo requerimiento computacional, entrenamiento relativamente rápido e interpretabilidad mediante la importancia de las variables.

La principal desventaja es que las estadísticas espaciales deben ser calculadas previamente mediante código tradicional.

### 3.2 Red neuronal convolucional

Como alternativa futura se considera el uso de una red convolucional (CNN), por ejemplo sobre parches de $33\times33$ o $65\times65$ píxeles.

En este caso la red recibiría directamente los datos espaciales y podría aprender automáticamente características asociadas al entorno del píxel, gradientes térmicos y estructuras espaciales.

Sin embargo, este enfoque requiere mayor cantidad de datos y recursos computacionales. Por este motivo, se considera como una posible extensión posterior del modelo tabular.

---

## 4. Estructura del vector de entrada

El conjunto de variables seleccionado para el modelo tabular se resume a continuación:

| Grupo | Variables |
|---|---|
| Radiometría directa | `bt7`, `bt14`, `bt15` |
| Reflectividad | `albedo`, `refl`, `refl2` |
| Variables visibles | `vis_brightness` |
| Geometría | `sza`, `lza`, `glint_angle` |
| Variables atmosféricas | `tpw` |
| Emisividad | `emiss7`, `emiss14` |
| Superficie | `eco_code` |
| Diferencia térmica | `diff_bt7_bt14` |
| Fondo térmico | `bt7_bkg`, `bt7_bkg_std`, `bt14_bkg`, `bt14_bkg_std` |
| Fondo de reflectividad | `albedo_bkg`, `reflb`, `std_reflb` |
| Diferencias respecto al fondo | `bt7_minus_bkg7`, `bt14_minus_bkg14`, `refl_minus_reflb` |
| Contexto de ventana | `n_passes`, `rad_diff_sigma` |
| Temperaturas corregidas | `bt7_corr`, `bt14_corr` |

El conjunto completo de variables es:

```text
bt7
bt14
bt15
albedo
refl
refl2
vis_brightness
tpw
sza
lza
glint_angle
emiss7
emiss14
eco_code
diff_bt7_bt14
bt7_bkg
bt7_bkg_std
bt14_bkg
bt14_bkg_std
albedo_bkg
reflb
std_reflb
bt7_minus_bkg7
bt14_minus_bkg14
refl_minus_reflb
n_passes
rad_diff_sigma
bt7_corr
bt14_corr
````

---

## 5. Variables excluidas

Para evitar fuga de información (*data leakage*), no se utilizan como entradas variables que correspondan directamente al resultado del algoritmo que se pretende aprender.

### 5.1 Variables que representan el target o resultados comparables

Se excluyen:

```text
ref_code
pred_code
ref_base_code
pred_base_code
base_code
final_code
upgraded_code
verdict
```

Estas variables representan el *target* o resultados directamente comparables con él.

### 5.2 Variables correspondientes al rastro de decisión interno

También se excluyen:

```text
stage
stage_killed_by
p1_code
fail_char_*
reached_candidate
confirmed_part2
eliminated
elim_reason
detection_policy
policy_*
conf_thr1_*
conf_thr2_*
conf_bkg7_minus_bkg14
conf_bt7c_minus_bkg7
p2_cond1
p2_cond2
p2_cond3
p2_*_thr
p2_refl_ok
mg_*
```

Estas variables representan el rastro de decisión interno del algoritmo existente.

Utilizarlas como entradas implicaría enseñarle al modelo a reproducir las decisiones de otro algoritmo en lugar de aprender directamente la relación entre las mediciones físicas y la presencia de fuego.

### 5.3 Resultados de Dozier

Se excluyen:

```text
fire_frac
fire_temp
frp
dozier_valid
```

Estas variables son resultados de la subrutina de Dozier y solamente tienen sentido una vez que el píxel ha sido clasificado como fuego.

### 5.4 Umbrales de decisión

Se excluyen:

```text
bt7_min_thr
bt7_refl_thr
```

debido a que representan umbrales de decisión y no mediciones físicas.

### 5.5 Resultados de tests internos

Se excluyen:

```text
is_cloudy
sat_flag
```

`is_cloudy` es el resultado de un conjunto de pruebas basadas en umbrales sobre variables que ya se encuentran disponibles como entradas.

`sat_flag` presenta el mismo problema: representa el resultado de un test de umbral basado en las variables radiométricas.

### 5.6 Variables redundantes o de filtrado

Se excluye:

```text
bt14_eff
```

debido a que representa una selección entre `bt13` y `bt14` según disponibilidad y, por lo tanto, no agrega información independiente a las bandas originales.

También se excluye:

```text
in_roi
```

ya que representa una máscara geográfica fija utilizada para filtrar la región de interés y no una característica física predictiva del fuego.

---

## 6. Definición del target

El objetivo del modelo es predecir la variable binaria:

```text
ref_fire
```

Esta variable se construye a partir de la máscara de referencia de la NOAA y representa si el píxel pertenece o no a una categoría de fuego según los códigos definidos para el problema.

Por lo tanto, el problema principal se formula como una clasificación binaria:

$$
X \longrightarrow \texttt{ref\_fire}
$$

donde $X$ corresponde al vector de características definido en la sección anterior.

---

## 7. Estrategia de entrenamiento

### 7.1 Desbalance de clases

La detección de incendios presenta un fuerte desbalance entre píxeles de fuego y píxeles que corresponden a superficie limpia, nubes u otras categorías.

Por este motivo, el entrenamiento deberá considerar estrategias para evitar que el modelo aprenda simplemente a clasificar la mayoría de los píxeles como no fuego.

Para modelos basados en árboles se podrán utilizar pesos de clase, mientras que en redes neuronales se podría utilizar una función de pérdida como *Focal Loss*.

También puede utilizarse submuestreo de píxeles de fondo, manteniendo una proporción mayor de los píxeles que presentan anomalías térmicas o que pertenecen a regiones relevantes para la detección.

### 7.2 División de los datos

La división entre entrenamiento, validación y prueba deberá realizarse por eventos temporales o días completos y, cuando corresponda, por regiones geográficas disjuntas.

No se deben mezclar cuadros consecutivos del mismo evento entre entrenamiento y prueba, ya que esto podría producir fuga de información debido a la fuerte correlación espacial y temporal entre observaciones.

### 7.3 Evaluación

El modelo se evaluará comparando la predicción de `ref_fire` contra la referencia construida a partir de la máscara de la NOAA.

Las métricas principales serán:

* precisión (*precision*);
* recall;
* F1-score;
* matriz de confusión.

Una evaluación posterior podrá incorporar fuentes independientes de mayor resolución espacial, como VIIRS, para analizar el comportamiento del modelo respecto de observaciones externas a la máscara utilizada como target.

---

## 8. Resumen del modelo

El modelo inicial será un clasificador basado en árboles que recibirá mediciones físicas del ABI, información geométrica y atmosférica, características de superficie y estadísticas del entorno espacial del píxel.

El conjunto de entrada se ha diseñado evitando variables que representen directamente decisiones internas del algoritmo de la NOAA o resultados posteriores de su procesamiento.

De esta manera, el modelo deberá aprender la relación entre las mediciones físicas y contextuales del píxel y la clasificación de fuego representada por `ref_fire`.

Como extensiones futuras se considera evaluar arquitecturas convolucionales que permitan aprender directamente características espaciales y, posteriormente, explotar la alta frecuencia temporal de las observaciones del GOES mediante modelos capaces de incorporar dependencias temporales.
