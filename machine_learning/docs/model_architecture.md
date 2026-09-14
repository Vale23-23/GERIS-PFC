# Selección del Enfoque de Arquitectura

Existen dos maneras principales de abordar este problema según cómo decidas procesar la información contextual:

## Enfoque A: Modelo Tabular Píxel a Píxel (XGBoost / LightGBM / CatBoost)

- **Cómo funciona:** Se extrae un vector de características (*feature vector*) por cada píxel individual, incluyendo tanto sus mediciones propias como las estadísticas de su entorno espacial (precalculadas mediante una ventana flotante).
- **Ventajas:** Excelente rendimiento computacional, bajo consumo de memoria, entrenamiento rápido e interpretabilidad directa de la importancia de los umbrales.
- **Desventajas:** Requiere calcular previamente las variables contextuales de fondo ($\mu, \sigma$, pases de ventana) mediante código tradicional.

## Enfoque B: Red Neuronal Convolucional (CNN 2D / U-Net / ResNet sobre Parches)

- **Cómo funciona:** La red recibe directamente tensores multi-canal correspondientes a parches de imagen (por ejemplo, parches de $33 \times 33$ o $65 \times 65$ píxeles centrados en el píxel de evaluación, o imágenes completas para segmentación semántica)36.
- **Ventajas:** **Elimina la necesidad de calcular manualmente las estadísticas de fondo** y las desviaciones estándar4; la convolución aprende automáticamente las variaciones locales del terreno, gradientes térmicos y bordes de nubes78.
- **Desventajas:** Mayor requerimiento computacional y necesidad de mayor volumen de datos para entrenar.

---

# Estructura Detallada del Tensor/Vector de Entrada (Inputs)

Si optas por el **Enfoque A (Tabular)**, cada muestra es un vector de atributos. Si optas por el **Enfoque B (CNN)**, cada canal de entrada es una matriz $H \times W$:

| Grupo de Entrada | Variables / Atributos | Descripción Física |
|---|---|---|
| **Radiometría Directa (ABI L1b)** | $T_7, T_{14}, T_2, T_{13}, T_{15}$910 | Temperaturas de brillo crudas del infrarrojo (3.9, 11.2, 10.35, 12.3 µm) y albedo visible (0.64 µm)11.
| **Diferencias y Productos** | $T_7 - T_{14}$, $\text{Refl}$ (Fog Product)1213 | Diferencia térmica básica y producto de reflectividad en espacio de $3.9\,\mu\text{m}$1314.
| **Geometría Solar y Visión** | $\text{SZA}, \text{LZA}, \text{Glint Angle}, \cos(\text{SZA})$1516 | Ángulo cenital solar, ángulo cenital local del satélite y ángulo de resplandor para identificar reflejos16.
| **Variables Auxiliares** | $\text{TPW}$, $\epsilon_7, \epsilon_{14}$, Tipo de Ecosistema1017 | Agua precipitable total (GFS)18, emisividad superficial (CAMEL/SEEBOR)17 y clase de uso de suelo UMD17.
| **Atributos Contextuales** *(Solo Enfoque Tabular)* | $\text{Temp4\_Bkg\_Mean}, \text{Temp4\_Bkg\_StdDev}$, $\text{Temp11\_Bkg\_Mean}, \text{StdDev\_4Mu\_11Mu\_Temp\_Diff}$, $\text{Reflb}, \text{Bkg\_Passes}$5more_horiz | Media y desviación estándar de la ventana de fondo en Ch7, Ch14, diferencia de bandas y producto de reflectividad4more_horiz.
| **Memoria Temporal** | $\Delta t_{\text{fuego\_anterior}}$1017 | Tiempo transcurrido (en segundos) desde la última detección en esa coordenada para emular el filtro temporal de 12 horas1022.

---

# Definición de Targets / Salidas del Modelo (Outputs)

El modelo puede diseñarse como una **red multitarea (Multi-Head)** con una cabeza principal de clasificación y cabezales secundarios de regresión:

**Cabezal Principal: Clasificación de la Máscara de Fuego (Fire Mask)**2324

Mapeo de las salidas de la Tabla 3.11 del ATBD en clases de entrenamiento24:

1. **Categorías de Fuego Activo Confirmado:** `Clase 0`: Processed Fire (Códigos `10` / `30`)2425.
2. `Clase 1`: Saturated Fire (Códigos `11` / `31`)2425.
3. `Clase 2`: Cloud Contaminated Fire (Códigos `12` / `32`)2425.
4. `Clase 3`: High Probability Fire (Códigos `13` / `33`)2426.
5. `Clase 4`: Medium Probability Fire (Códigos `14` / `34`)2426.
6. `Clase 5`: Low Probability Fire (Códigos `15` / `35`)2426.
7. **Fondo y Superficie Limpia:** `Clase 6`: Fire-Free Ground / Processed (Código `100`)2728.
8. **Nubes y Exclusiones:** `Clase 7`: Opaque Cloud / Cloud Edge (Códigos `200` al `245`)29more_horiz.
9. `Clase 8`: Block-outs / Non-processed (Espacio `40`, LZA `50`, Glint `60`, Agua/Invalido `150-153`)24more_horiz.

**Cabezales Secundarios de Regresión (Solo activos si la predicción es Fuego Procesado - Código 10/30)**23:

- **Temperatura de Fuego ($T_t$):** En Kelvin (rango $400\text{ K} - 1500\text{ K}$)2332.
- **Área o Proporción del Fuego ($p$):** En $\text{km}^2$ o porcentaje del píxel2332.
- **Potencia Radiativa ($FRP$):** En Megavatios (MW)3233.

---

# Estrategia de Entrenamiento y Pérdida (Loss Function)

1. **Manejo del Desbalance Extremo de Clases:** En una escena completa del GOES, el **99.9% de los píxeles son tierra limpia (100) o nubes (200+)**16more_horiz, mientras que los incendios ocupan un porcentaje mínimo. *Solución:* Utilizar **Focal Loss** en redes neuronales o ajustar los pesos de clase (`class_weight='balanced'`) en árboles de decisión.
2. *Submuestreo:* Submuestrear drásticamente los píxeles fríos de fondo (Clase 100) y mantener el 100% de los píxeles con anomalías térmicas y nubes de borde.
3. **División Estricta de Datos (Train/Val/Test):** Se debe realizar una división por **eventos temporales/días completos** o regiones geográficas disjuntas, evitando mezclar cuadros consecutivos de la misma hora para prevenir el fugado de información (*data leakage*).
4. **Validación Cruzada:** Validar las matrices de confusión comparando las predicciones contra las máscaras oficiales de la NOAA2331 y contra verdades de terreno (*truth data*) de resolución fina como Landsat-8 OLI / VIIRS3435.
