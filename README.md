# Predicción de eventos sísmicos de alta energía en una mina subterránea de carbón

Proyecto de machine learning para predecir, a partir de las mediciones de un turno de 8 horas, si en el turno siguiente se registrará un golpe sísmico de energía mayor a 10⁴ J.

El foco del proyecto está en la **metodología**: datos muy desbalanceados, evaluación sin fuga de información, comparación honesta de alternativas y reporte claro de las limitaciones. El modelo supera al azar, pero **no es apto para uso operativo** (ver [Conclusiones](#conclusiones-y-limitaciones)).

## Problema

En minería subterránea, los golpes sísmicos de alta energía son difíciles de predecir y raros frente a los de baja energía. El dataset define como **estado peligroso** (`class = 1`) un turno tras el cual ocurre al menos un golpe de más de 10⁴ J. Con esa definición operativa, el problema es una clasificación binaria con una clase positiva muy escasa: solo 170 de 2.584 turnos (≈ 6,6 %) son positivos.

Por eso la exactitud (*accuracy*) no sirve como métrica: un modelo que siempre responda "sin evento" acertaría más del 93 % y sería inútil. Se usan PR-AUC (métrica principal), ROC-AUC, recall y precisión.

## Datos

Dataset *seismic-bumps*, del [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

- **Autores y año:** Marek Sikora y Lukasz Wrobel 2010
- **Enlace al dataset:** https://archive.ics.uci.edu/dataset/266/seismic+bumps
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

El archivo incluido en `data/` no tiene modificaciones. Todas las transformaciones (codificación de categóricas, selección de columnas, variables nuevas) se hacen en memoria dentro del notebook.

Cada fila resume un turno de 8 horas. Las columnas principales usadas en el análisis:

| Columna | Significado |
|---|---|
| `seismic` | Evaluación del peligro sísmico del turno obtenida con el método sísmico: `a` sin peligro, `b` peligro bajo, `c` peligro alto, `d` estado de peligro |
| `seismoacoustic` | Misma evaluación, obtenida con el método sismoacústico |
| `shift` | Tipo de turno: `W` extracción de carbón, `N` preparación |
| `genergy` | Energía sísmica registrada en el turno anterior por el geófono más activo (GMax) de los que monitorean el frente de explotación |
| `gpuls` | Número de pulsos registrados por GMax en el turno anterior |
| `gdenergy` | Desviación de la energía de GMax en el turno anterior respecto al promedio de los ocho turnos previos |
| `gdpuls` | Desviación del número de pulsos de GMax en el turno anterior respecto al promedio de los ocho turnos previos |
| `ghazard` | Evaluación de peligro sismoacústica basada solo en los registros de GMax |
| `nbumps` | Número de golpes sísmicos registrados en el turno anterior |
| `nbumps2` | Golpes con energía en [10², 10³) J |
| `nbumps3` | Golpes con energía en [10³, 10⁴) J |
| `nbumps4` | Golpes con energía en [10⁴, 10⁵) J |
| `nbumps5` | Golpes con energía en [10⁵, 10⁶) J |
| `nbumps6` | Golpes con energía en [10⁶, 10⁷) J (siempre 0 en los datos) |
| `nbumps7` | Golpes con energía en [10⁷, 10⁸) J (siempre 0 en los datos) |
| `nbumps89` | Golpes con energía en [10⁸, 10¹⁰) J (siempre 0 en los datos) |
| `energy` | Energía total de los golpes sísmicos registrados en el turno anterior |
| `maxenergy` | Energía del golpe más fuerte registrado en el turno anterior |
| `class` | **Objetivo:** `1` si en el turno siguiente ocurre un golpe de alta energía (estado peligroso), `0` si no |

## Metodología

Todo el flujo está en [`prediccion_peligro.ipynb`](prediccion_peligro.ipynb):

1. **Validación de datos:** nulos, duplicados, columnas constantes (`nbumps6`, `nbumps7` y `nbumps89` valen 0 siempre y se descartan), consistencia entre columnas y balance de clases.
2. **Codificación** de variables categóricas con `OrdinalEncoder`.
3. **Split train/test (75 % / 25 %)** estratificado. El test no se toca hasta la evaluación final.
4. **Evaluación de columnas con el train:** correlación de Spearman (redundancia), AUC univariado e información mutua (relación con el objetivo).
5. **Selección:** 5 variables (`nbumps`, `energy`, `genergy`, `nbumps2`, `gpuls`). De `energy` y `maxenergy` (correlación ~1) se conserva `energy`.
6. **Ingeniería de características:** dos variables nuevas (`energia_por_pulso` y `genergy_vs_energy`), calculadas fila a fila para evitar fuga de información.
7. **Validación cruzada repetida** (5 folds × 5 repeticiones) para comparar random forest y gradient boosting, con y sin las variables nuevas.
8. **Elección del umbral de decisión** maximizando F2 sobre predicciones de validación cruzada del train.
9. **Evaluación final única** en el test.

## Resultados

### Validación cruzada (train)

| Modelo | Columnas | PR-AUC | ROC-AUC |
|---|---|---|---|
| Random forest | base (5) | 0,225 ± 0,049 | 0,747 ± 0,056 |
| Random forest | base + nuevas (7) | 0,229 ± 0,060 | 0,749 ± 0,050 |
| Gradient boosting | base (5) | 0,204 ± 0,051 | 0,719 ± 0,048 |
| Gradient boosting | base + nuevas (7) | 0,201 ± 0,061 | 0,713 ± 0,046 |

Referencia de azar: PR-AUC ≈ 0,066 · ROC-AUC = 0,5.

### Test (evaluación única)

Modelo final: random forest con las 5 variables base y umbral 0,563 (elegido con el train).

| Métrica | Valor |
|---|---|
| PR-AUC | 0,276 (azar: 0,065) |
| ROC-AUC | 0,812 |
| Recall | 0,43 (18 de 42 eventos detectados) |
| Precisión | 0,23 |
| F2 | 0,37 |
| Falsas alarmas | 59 |
| Eventos no avisados | 24 |
| `nbumps` sola (PR-AUC) | 0,172 (diferencia con el modelo: 0,11; IC 95 % [0,008; 0,229]) |

Matriz de confusión (filas = real, columnas = predicho):

|  | Predicho: sin evento | Predicho: evento |
|---|---|---|
| **Real: sin evento** | 545 | 59 |
| **Real: evento** | 24 | 18 |

## Conclusiones y limitaciones

**Qué funcionó.** El modelo supera claramente al azar (PR-AUC unas cuatro veces mayor). Frente a la regla más simple, basada solo en `nbumps`, la mejora es modesta: +0,04 de PR-AUC en validación cruzada (no concluyente) y +0,11 en el test, con un intervalo que apenas excluye el 0. En ROC-AUC no hay diferencia (0,747 contra 0,744). La señal se concentra en los golpes sísmicos del turno y, en menor medida, en la actividad del geófono.

**Qué no funcionó.** Las dos variables nuevas no mejoraron el rendimiento ni con random forest ni con gradient boosting: las diferencias son menores que la desviación estándar. Gradient boosting tampoco superó al random forest. Que dos algoritmos distintos den la misma conclusión sugiere que el límite está en la información del dataset y no en el modelo.

**Limitaciones.**

- Pocos eventos positivos (170 en total, 42 en el test), por lo que todas las métricas son ruidosas. El PR-AUC del test (0,276) es algo mayor que el de validación cruzada (0,225), pero con tan pocos eventos esa diferencia es compatible con el azar.
- Cada fila describe un solo turno, sin historia previa.
- Los datos provienen de una sola mina.
- Con el umbral elegido, el modelo avisa menos de la mitad de los eventos y tres de cada cuatro alarmas son falsas. **No es apto como herramienta única de alerta.**
- El modelo aporta poco sobre una regla de una sola variable (`nbumps`): la ventaja en PR-AUC es pequeña y su significancia no es clara con tan pocos eventos.

## Siguientes pasos

- Probar variables temporales (promedios móviles y rezagos), si se confirma que las filas están en orden cronológico, con una validación que respete el tiempo.
- Empaquetar el modelo en una aplicación de demostración.

## Cómo reproducirlo

```bash
git clone https://github.com/MatiRivass/mine-seismic-hazard-prediction.git
cd mine-seismic-hazard-prediction
pip install -r requirements.txt
```

Abre `prediccion_peligro.ipynb` desde la carpeta del proyecto y ejecuta todas las celdas en orden. Todo lo aleatorio usa `random_state` fijo, por lo que los resultados son reproducibles.

Probado con Python 3.13.3.

## Licencia

- Datos (`data/`): CC BY 4.0, de sus autores originales (ver sección Datos).
