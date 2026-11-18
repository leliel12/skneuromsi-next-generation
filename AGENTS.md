# AGENTS.md — Guía del proyecto `skneuromsi-next-generation`

Documento de contexto para quien (persona o agente) vaya a **refactorizar**
este repositorio. Describe el objetivo, la arquitectura actual, los
contratos entre piezas, las convenciones numéricas, lo que está muerto o roto
y las decisiones abiertas. Está escrito a partir del código y del paper; donde
algo no se pudo verificar, se dice.

Última revisión del contenido: 2026-11-18 (rama `master`, commit `7c09427`).

---

## 1. Qué es este proyecto y por qué existe

Es la **siguiente generación de Scikit-NeuroMSI**, el framework en Python
para modelar integración multisensorial. El paquete original (paper:
Paredes, Cabral & Seriès, *Scikit-NeuroMSI: A Generalized Framework for
Modeling Multisensory Integration*, Neuroinformatics 2025, 23:40,
doi:10.1007/s12021-025-09738-1; PDF en `draft/`) implementa cada modelo como
una clase monolítica (`ModelABC`) que recibe los parámetros del estímulo
**dentro** de `run()`.

### 1.1 El pendiente que este repo implementa

La sección **"Availability and Future Directions"** del paper (p. 12–13)
propone descomponer cada modelo en dos entidades:

- Un componente de **procesamiento de estímulos** que maneja cada entrada
  sensorial individual.
- Un componente de **integración** que consolida los resultados en una
  modalidad unificada.

Formalmente (Eq. 1 del paper):

```
E : S × I → R
```

- `S`: las fuentes de estímulo, de una o varias modalidades.
- `I`: el integrador que las combina.
- `R`: el espacio de resultados.

El paper muestra la API deseada en el **Code 6** (composición):

```python
m0 = Model(stimulus.Auditory(...), stimulus.Visual(...),
           integrators.SomeIntegrator(...))     # un sonido y un estímulo visual
m1 = Model(stimulus.Auditory(...), stimulus.Auditory(...),
           integrators.SomeIntegrator(...))     # dos sonidos
```

Beneficios buscados (textual del paper): poder implementar y probar nuevos
modelos de integración **sin modificar el código de estímulos**; extender a
más modalidades o tipos de estímulo (interfaces cerebro-computadora,
robótica); más reutilización y mantenibilidad del código.

### 1.2 Cómo se tradujo al código

| Paper | Este repo |
|---|---|
| `S` (fuentes de estímulo) | `Stimulus`, `Visual`, `Audio` ([stimulus.py](neuromsi/stimulus.py)) |
| `I` (integrador) | `Integrator` y sus subclases: `Cuppini2017`, `Kording2007`, `AlaisBurr2004` |
| `E : S × I → R` | `Backend(stimuli, integrator, ...).run()` → `NDResult` |
| (no está en el paper) | `NeuralSignal` / `UnisensorySignal` ([signal.py](neuromsi/signal.py)): el contrato entre `S` e `I` |
| `R` | `NDResult` ([core/ndresult/](neuromsi/core/ndresult/)) |

La API de uso real hoy es:

```python
from neuromsi import Audio, Visual, Backend, Cuppini2017

result = Backend(
    stimuli=[Audio(position=60), Visual(position=70)],
    integrator=Cuppini2017(),
    time_range=(0, 150),
    time_res=0.5,
    seed=42,
).run()
```

Nota: el paper propone `Model(stim, stim, integrator)`; aquí esa composición
la hace `Backend`, que además valida la grilla, reparte el generador
aleatorio y empaqueta el resultado.

---

## 2. Estado actual

Hecho:

- Separación estímulo / señal / integrador / backend funcionando.
- Tres modelos portados: `Cuppini2017` (neural), `Kording2007` (bayesiano) y
  `AlaisBurr2004` (MLE). Los tres consumen los mismos objetos `Stimulus`.
- Ejemplos de comparación contra la implementación original en
  `testcode_*/` (ver §9).
- Docstrings estilo numpy y comentarios de algoritmo en estímulos, señales,
  integrador, backend, Cuppini2017 y Kording2007.
- Codificación de señales sin `if` en el `Backend`: `SignalDispatcher` (tabla
  por `model_type`) + codificadores públicos en `Stimulus` (ver §3 y §11).
- Métodos internos privados (`_x`) en `Stimulus`, `Backend` e `Integrator`.

Pendiente / no portado (existe en el paquete original, no acá):

- **`Paredes2025`** (el "Multisensory Spatiotemporal Causal Inference
  Network" del paper, Apéndice C.4). No está implementado.
- **`ParameterSweep`** (`sweep.py` del original) y `testing.py`.
- Cuppini2017 original tenía parámetros por estímulo en `run()`; acá viven en
  el `Stimulus` (ver §5.1).
- Tests automáticos: **no existe** el directorio `tests/` (ver §9).
- Documentación de usuario, README real (el actual es una sola línea) y
  empaquetado correcto (ver §8).

---

## 3. Arquitectura y flujo de datos

```
 Stimulus (Visual/Audio)                       Integrator
  "qué pasó en el mundo"                        "cómo lo procesa un modelo"
        │                                             ▲
        │ generate_neural_signal(...)                 │ execute(signals, **grids)
        │ generate_unisensory_signal(...)             │   └─ integrate(...)
        ▼                                             │
   NeuralSignal / UnisensorySignal  ──── Backend ─────┘
   (contrato, inmutable, validado)    SignalDispatcher
                                            │
                                            ▼
                         (response, extra) → _calculate_causes → NDResult
```

Paso a paso de `Backend.run()` ([backend.py](neuromsi/backend.py)):

1. `integrator.check_stimuli_compatible(modalities)` — falla temprano si las
   modalidades no son las que el modelo espera.
2. `dispatcher = SignalDispatcher(self)` y, en un bucle de `run()`,
   `signals[stim.modality] = dispatcher.encode(stim)`. `SignalDispatcher` es una
   clase de [backend.py](neuromsi/backend.py) que recibe el propio `Backend` y
   se crea dentro de `run()` (así ve siempre los valores actuales). Valida
   `integrator.model_type` al construirse y despacha con su tabla `_ENCODERS`
   (`model_type` → adaptador). El adaptador (`_neural_signal` o
   `_statistical_signal`) lee de `backend.integrator`, `time_range`,
   `time_res` y `random` los argumentos y llama al codificador **público** del
   estímulo, que no conoce al `Backend` ni al integrador (ver tabla abajo).
3. `integrator.execute(signals, time_range, time_res, position_range,
   position_res, random)` — valida las señales y delega en
   `integrator.integrate(...)`, que devuelve `(response, extra)`.
4. `extra["stim_position"] = [...]` — se agrega acá, fuera de las señales.
5. `_make_ndresult(response, extra, stimuli)` — agrega los parámetros de
   causas, calcula las causas con `integrator.calculate_causes(...)`, arma
   `run_parameters` (incluye `integrator.model_parameters`) y construye el
   `NDResult`.

### 3.1 `model_type` decide la codificación

El `Backend` **no conoce** los modelos concretos: lee el atributo de clase
`model_type` del integrador; el `SignalDispatcher` elige el adaptador con su
tabla `_ENCODERS`. Conoce, en cambio, los dos codificadores públicos del
estímulo (cada adaptador llama a uno). El cableado (qué parámetros del
integrador necesita cada familia) vive en el `SignalDispatcher`; `Stimulus`
recibe argumentos explícitos y queda independiente del `Backend`.

| `model_type` | Señal | Productor en `Stimulus` | Modelos |
|---|---|---|---|
| `"Neural"` | `NeuralSignal` (`values`: matriz `(tiempo, neuronas)`) | `generate_neural_signal(neurons, simulation_length, time_res)`; adaptador `SignalDispatcher._neural_signal` (lee `integrator.neurons`, `time_range[1]`, `time_res`) | `Cuppini2017` |
| `"Bayesian"` | `UnisensorySignal` (`values`: `{"var", "input"}`) | `generate_unisensory_signal(n, noise, random)`; adaptador `SignalDispatcher._statistical_signal` (lee `integrator.n_sim` y `integrator.noise`, con defaults 1 y `False`, y el `random` de la corrida) | `Kording2007` |
| `"MLE"` | `UnisensorySignal` (ídem) | ídem | `AlaisBurr2004` |

Un `model_type` desconocido lanza `ValueError` en
`SignalDispatcher.__init__`. **Para soportar una familia de modelos nueva:**
escribir su codificador público en `Stimulus`, un adaptador en
`SignalDispatcher` y registrarlo en `_ENCODERS` (el `Backend` no cambia).
Agregar un modelo *dentro* de
una familia existente no toca ninguno de los dos.

`NDResult.mtype` guarda ese `model_type` como string (no se valida contra una
lista).

---

## 4. Mapa de módulos

Todo está bajo [neuromsi/](neuromsi/). El paquete se importa como `neuromsi`
(el original es `skneuromsi`).

### 4.1 Núcleo de la arquitectura nueva (lo que importa para el refactor)

| Archivo | Contenido |
|---|---|
| [`__init__.py`](neuromsi/__init__.py) | API pública: `Stimulus`, `Visual`, `Audio`, `NeuralSignal`, `UnisensorySignal`, `Integrator`, `Cuppini2017`, `Kording2007`, `AlaisBurr2004`, `Backend`. |
| [`stimulus.py`](neuromsi/stimulus.py) | `Stimulus` (base), `Visual`, `Audio`. Describe posición, intensidad, sigma, onset, duración, `stim_n`, `soa`. Sabe codificarse a señal. |
| [`signal.py`](neuromsi/signal.py) | `NeuralSignal`, `UnisensorySignal`: dataclasses inmutables (`frozen=True`) que validan en `__post_init__`. |
| [`integrator.py`](neuromsi/integrator.py) | `Integrator`: base común. `execute()`, helpers de sinapsis (privados), contrato abstracto. |
| [`backend.py`](neuromsi/backend.py) | `Backend`: orquesta, valida grilla, empaqueta `NDResult`. Único método público: `run()`. Incluye `SignalDispatcher`, que codifica los estímulos según el `model_type` del integrador. |
| [`neural/cuppini2017.py`](neuromsi/neural/cuppini2017.py) | Red neuronal de 3 capas (Cuppini et al., 2017). |
| [`bayesian/kording2007.py`](neuromsi/bayesian/kording2007.py) | Inferencia causal bayesiana (Körding et al., 2007). |
| [`mle/alais_burr2004.py`](neuromsi/mle/alais_burr2004.py) | Integrador óptimo bimodal (Alais & Burr, 2004). Sin inferencia causal. |

### 4.2 Utilidades de modelos

| Archivo | Contenido |
|---|---|
| [`utils/neural_tools.py`](neuromsi/utils/neural_tools.py) | `calculate_neural_distance` (distancia en anillo, **usada**) y copias libres de `calculate_lateral_synapses`, `calculate_inter_areal_synapses`, `calculate_stimuli_input`, `create_unimodal_stimuli_matrix` (**duplicadas** respecto de `Integrator`/`Stimulus`; ver §8). |
| [`utils/readout_tools.py`](neuromsi/utils/readout_tools.py) | Lectura de causas por picos (`scipy.signal.find_peaks`): `calculate_causes_from_peaks`, `calculate_spatiotemporal_causes_from_peaks`, probabilidades de picos. Lo usa `Cuppini2017.calculate_causes`. |
| [`utils/bunch.py`](neuromsi/utils/bunch.py), [`utils/ddtype_tools.py`](neuromsi/utils/ddtype_tools.py) | `Bunch` y herramientas de dtypes que usa `NDResult`. |

### 4.3 Resultados (heredado del paquete original, funciona)

| Carpeta | Contenido |
|---|---|
| [`core/ndresult/`](neuromsi/core/ndresult/) | `NDResult` (envuelve un `xarray.DataArray` de dims `modes, times, positions, positions_coordinates`), accesores `plot` y `stats`, compresión. |
| [`core/constants.py`](neuromsi/core/constants.py) | `NAME`, `VERSION`, nombres de dimensiones (`D_MODES`, `D_TIMES`, ...). |

### 4.4 Heredado, muerto o roto

Ver §8. Resumen: `core/modelabc.py` (muerto), `io.py` y `ndcollection/`
(**no se pueden importar**).

---

## 5. Contratos entre piezas

### 5.1 `Stimulus` — describe el mundo, no el modelo

```python
Stimulus(position, intensity, sigma, onset=0, duration=None, stim_n=1, soa=None)
Visual(position, *, intensity=27, sigma=4, onset=0, duration=100, **kwargs)
Audio(position, *, intensity=28, sigma=32, onset=0, duration=100, **kwargs)
```

- `modality` es atributo de **clase** (`"visual"`, `"auditory"`, `"generic"`);
  el `Backend` lo usa como clave para enrutar la señal.
- Unidades: posición en grados; tiempos en ms.
- `duration=None` ⇒ dura toda la simulación.
- `stim_n` repeticiones separadas por `soa` (onset a onset; el hueco real es
  `soa - duration`).
- API pública de codificación: `generate_neural_signal(neurons,
  simulation_length, time_res)` y `generate_unisensory_signal(n, noise,
  random)`, con argumentos explícitos (el estímulo no conoce al `Backend`).
  Todo lo demás es privado:
  `_calculate_stimuli_input` (gaussiana espacial), `_create_unimodal_stimuli_matrix`
  (línea de tiempo), `_draw_measurement` (medición para modelos estadísticos).
- `_draw_measurement`: con `noise=False` devuelve la **posición verdadera**
  (observador perfecto; se eliminó un sesgo antiguo de `+sigma`); con
  `noise=True` devuelve `position + sigma * N(0,1)` con `n` muestras.

### 5.2 Señales — el contrato

- `NeuralSignal(modality, values)`: `values` es `np.ndarray` 2D
  `(time_steps, neurons)`. Valida `ndim == 2` y `modality` no vacía.
- `UnisensorySignal(modality, values)`: `values` es `dict` con claves
  `"var"` (escalar > 0, varianza = `sigma**2`) y `"input"` (escalar o array
  de `n_sim` valores). Copia el dict al construirse.
- Ambas son inmutables; para cambiar el dato convertido usan
  `object.__setattr__` dentro de `__post_init__`.
- Históricamente `UnisensorySignal` se llamaba `BayesianSignal`; se renombró al
  agregar el modelo MLE. Si encontrás el nombre viejo, es un resto.

### 5.3 `Integrator` — qué debe cumplir una subclase

Base: [integrator.py](neuromsi/integrator.py). Una subclase **debe** aportar:

| Miembro | Descripción |
|---|---|
| `model_type` | `"Neural"`, `"Bayesian"` o `"MLE"` (atributo de clase). El de la base es `"Neural"`. |
| `check_stimuli_compatible(modalities)` | Lanza `ValueError` si las modalidades no sirven. La base lanza `NotImplementedError`. |
| `integrate(signals, **kwargs)` → `(response, extra)` | El modelo en sí. |
| `calculate_causes(**kwargs)` | Lectura del número de causas (recibe los `extra` del run). |
| `model_parameters` (propiedad) | Dict con todos los parámetros que definen el modelo; el `Backend` lo guarda en `run_parameters["model"]` para reproducibilidad. |

Lo que la base ya da:

- `execute(signals, **kwargs)`: valida que `signals` sea `Mapping`, que cada
  clave coincida con `signal.modality`, llama `check_stimuli_compatible` y
  luego `integrate`.
- `_calculate_lateral_synapses` (sombrero mexicano) y
  `_calculate_inter_areal_synapses` (gaussiana): privados, los usa solo
  `Cuppini2017`.

Nota: `integrate` y `calculate_causes` **no están declarados** en la base;
el contrato es solo por convención y docstring. No hay clase abstracta (`abc`).

`integrate` recibe por kwargs: `time_range`, `time_res`, `position_range`,
`position_res`, `random`. Cada modelo toma solo los que necesita
(`Cuppini2017.integrate` ignora los de posición vía `**kwargs`).

`response` es un dict `{mode0: array, mode1: array, "multi": array}`, cada
array de forma `(n_times, n_positions)`.

### 5.4 `Backend`

```python
Backend(stimuli, integrator, *, seed=None, time_range=(0, 100), time_res=0.01,
        position_range=(0, 180), position_res=1, causes_kind="count",
        causes_dim="space", causes_peak_threshold=0.15,
        causes_peak_distance=None)
```

- Guarda `self.stimuli = tuple(stimuli)` y crea **un único**
  `np.random.default_rng(seed)` que reparte a codificación e integración (así
  toda la corrida es reproducible con una semilla).
- `_check_position_range()` (en `__init__`): para integradores con atributo
  `neurons`, exige que `position_range`/`position_res` describan exactamente
  esa cantidad de posiciones. Los modelos sin neuronas se saltan el chequeo.
- Los parámetros `causes_*` son del **Backend** (política de lectura), no del
  integrador.
- Métodos: `_check_position_range`, `_make_ndresult`,
  `_calculate_causes` (privados, van primero) y `run` (público, va al final).
- `SignalDispatcher` (mismo archivo, definida antes de `Backend`) depende de
  cuatro atributos del `Backend`: `integrator`, `time_range`, `time_res` y
  `random`. Si se renombra alguno, hay que tocarla.

---

## 6. Modelos implementados

Todos requieren **exactamente 2 estímulos**, uno por cada `mode0`/`mode1`
(por defecto `"auditory"` y `"visual"`).

### 6.1 `Cuppini2017` (`model_type = "Neural"`)

- Tres capas en anillo de `neurons` unidades (por defecto 180): `mode0`,
  `mode1` y una capa multisensorial. La capa multisensorial no recibe
  entrada externa, solo *feedforward* de las unisensoriales.
- Dinámica: `tau * dy/dt = -y + F(u)`, `F(u) = 1/(1+exp(-s(u-theta)))`,
  integrada por **Euler sincrónico** (todas las entradas de un paso se
  calculan con la actividad del paso anterior).
- `u` = externa (señal + ruido opcional) + cruzada + lateral (unisensoriales);
  feedforward + lateral (multisensorial).
- Ruido: uniforme en `[-A, +A]` por neurona y paso, con
  `A = noise_level * max(señal de esa capa)`.
- Parámetros por defecto: `tau=(3, 15, 1)`, `s=0.3`, `theta=20`,
  `cross_modal_weight=1.4`, `cross_modal_sigma=5`, `feedforward_weight=18`,
  `feedforward_sigma=0.5`, `noise_level=0.40`; sombrero mexicano por capa en
  `lateral_params`.
- `get_connectivity()` es **público** y devuelve un dict con
  `lateral`, `cross_modal`, `feedforward`; las matrices son `[post, pre]`.
- Causas: cuenta picos de la capa multisensorial
  (`calculate_spatiotemporal_causes_from_peaks`); en `"time"` lee la neurona
  `int(mean(stim_position))`.
- Sin diferencia de latencia (`Δt_feed`, `Δt_cross`) ni retroalimentación: eso
  es de `Paredes2025`, no implementado.

### 6.2 `Kording2007` (`model_type = "Bayesian"`)

- Inferencia causal bayesiana. Parámetros: `n_sim=1`, `p_common=0.5`,
  `prior_sigma=20.0`, `prior_mu=0`, `strategy` (`"selection"`,
  `"averaging"`, `"matching"`), `noise=True`, `dimension` (`"space"` o
  `"time"`).
- Simula `n_sim` observadores independientes; el resultado es la
  **distribución** de estimaciones (histograma normalizado) sobre la grilla.
- El resultado es estático: ocupa **una sola fila** (`dimension="space"`) o
  **una sola columna** (`"time"`) de la matriz `(tiempo, posición)`; el resto
  queda en cero.
- La distribución `"multi"` es la estimación fusionada y **no depende de la
  estrategia**; solo las de cada modalidad dependen.
- `extra` incluye `mean_p_common_cause` y `p_common_cause`.
- `calculate_causes`: `"count"` → 1 si `mean_p_common_cause > 0.5`, si no 2;
  `"prob"` → `mean_p_common_cause`.
- `unisensory_estimator` y `multisensory_estimator` son **públicos** hoy.

### 6.3 `AlaisBurr2004` (`model_type = "MLE"`)

- Combinación lineal ponderada por varianzas inversas; sin inferencia
  causal: siempre fusiona.
- Parámetros: `mode0`, `mode1`, `noise=False`.
- Evalúa gaussianas sobre la grilla de posiciones; si `input` es un array
  (ruido con `n_sim` > 1) promedia las densidades (mezcla de gaussianas).
- Estático: llena la primera fila temporal; `extra` trae
  `"<mode0>_weight"` y `"<mode1>_weight"`.
- `calculate_causes` es constante: `1` (`"count"`) o `1.0` (`"prob"`).
- Notar que `Backend` solo le pasa `n_sim=1` (no tiene el atributo) y lee su
  `noise`.

---

## 7. Convenciones numéricas y supuestos implícitos

Estos son los puntos donde un refactor puede romper cosas sin que nadie se
entere. Están verificados en el código salvo que se indique lo contrario.

1. **Posición ≡ índice de neurona.** `Stimulus._calculate_stimuli_input` y
   `Cuppini2017.calculate_causes` usan la posición directamente como índice
   de neurona. Solo es correcto con `position_res=1` y `position_range`
   comenzando en 0. El `Backend` valida la *cantidad* de posiciones, no esta
   equivalencia.
2. **`time_range[0]` debe ser 0 para modelos neuronales.** El `Backend` pasa
   `simulation_length=time_range[1]` al estímulo, mientras `Cuppini2017`
   crea `np.arange(*time_range, time_res)` pasos. Solo coinciden con inicio
   en 0.
3. **Matriz temporal del estímulo.** Se construye con una fila por unidad de
   tiempo (ms), con `onset`, `duration`, `stim_n` y `soa` convertidos a
   enteros, y al final se refina con `np.repeat(..., 1/time_res)`. Implica que
   `1/time_res` debería ser entero.
4. **Distancia en anillo.** `calculate_neural_distance(neurons, j, k)` envuelve
   el espacio: la neurona 0 está al lado de la `neurons-1`.
5. **Errores del estímulo:** `ValueError` si la duración total (con SOAs) no
   cabe, o si `soa < duration`. Si el `onset` deja sin lugar al estímulo,
   NumPy falla con *"negative dimensions are not allowed"* (mensaje poco
   claro, pendiente).
6. **Un solo RNG.** Todo el azar sale de `Backend.random`. Los integradores
   reciben `random` y no deben crear generadores propios.
7. **`noise` pertenece al integrador.** El `Backend` lo lee del integrador
   para decidir si la señal es ruidosa. En `Cuppini2017` el ruido se agrega
   dentro de `integrate`, no en la señal.
8. **Resultados estáticos vs dinámicos.** Los modelos estadísticos rellenan
   una sola fila/columna; no hay tiempo real. `Cuppini2017` sí produce
   dinámica.

---

## 8. Código muerto, roto y deuda técnica (verificado)

- **`core/modelabc.py` (~740 líneas): no se usa.** Ningún módulo lo importa;
  `core/__init__.py` solo exporta `NAME`, `NDResult`, `VERSION`. Contiene
  `SKNMSIMethodABC`, `SKNMSIRunConfig`, `ParameterAliasTemplate`: el diseño
  monolítico que este repo reemplaza. Candidato a borrar.
- **`io.py` y `ndcollection/` no se pueden importar.**
  `from neuromsi.ndcollection import ...` falla con
  `ImportError: cannot import name 'dict_cmp' from 'neuromsi.utils'`;
  `io.py` además importa `utils.custom_json`, que tampoco existe. Consecuencia:
  `NDResult.to_ndr` (guardado) probablemente no funciona (no se probó).
  Además `io.py` hace `".".join(map(str, core.VERSION))` sobre un `VERSION` que
  es un string, lo que daría `"0.0.0+unknown"` con puntos entre caracteres
  (no verificado; el módulo no importa).
- **Duplicación en `utils/neural_tools.py`:** `calculate_lateral_synapses`,
  `calculate_inter_areal_synapses`, `calculate_stimuli_input` y
  `create_unimodal_stimuli_matrix` existen como funciones libres y también
  como métodos de `Integrator`/`Stimulus`. Los métodos son los que se usan; las
  funciones libres no las llama nadie (`calculate_neural_distance` sí).
- **`pyproject.toml` desactualizado:**
  - `[tool.setuptools.packages.find] include = ["skneuromsi", ...]` — el
    paquete es `neuromsi`. Funciona hoy solo por instalación editable.
  - `license = {file = "LICENSE.txt"}` — el archivo es `LICENSE`.
  - `testpaths = ["tests"]` — no existe.
  - Dependencias que el código nuevo no usa (`brainpy`, `jax`, `jaxlib`, ...)
    y posibles faltantes (`matplotlib` no figura). Hay que auditarlas.
- **`README.md`** tiene una sola línea. **`scikit_neuromsi.egg-info/`** está
  en el árbol de trabajo (revisar si debe ignorarse).
- **Mensaje fijo en `Cuppini2017.check_stimuli_compatible`:** dice
  "one visual and one auditory" aunque `mode0`/`mode1` sean otros.
- **`Visual` y `Audio` duplican el `__init__`**, difieren solo en defaults.
  Idea: atributos de clase (`default_intensity`, `default_sigma`) y un único
  `__init__` en la base.
- **Convenciones de visibilidad inconsistentes:** `Cuppini2017.get_connectivity`,
  `Cuppini2017.sigmoid` y `Kording2007.unisensory_estimator` /
  `multisensory_estimator`, `AlaisBurr2004.weight_calculator` /
  `unisensory_estimator` / `multisensory_estimator` son públicos aunque
  parecen internos. Se evaluó hacerlos privados; no se hizo.

---

## 9. Cómo correr y cómo verificar

**No hay tests automáticos.** La verificación actual es:

1. **Smoke test de cada modelo** (rápido, sin gráficos):

   ```python
   from neuromsi import Audio, Visual, Backend, Cuppini2017, Kording2007, AlaisBurr2004

   Backend(stimuli=[Audio(60), Visual(70)], integrator=Cuppini2017(noise=True),
           seed=1, time_range=(0, 150), time_res=0.5).run()

   Backend(stimuli=[Audio(-20, intensity=1, sigma=5), Visual(25, intensity=1, sigma=5)],
           integrator=Kording2007(n_sim=2000, p_common=0.3, prior_sigma=30,
                                  strategy="selection", noise=True),
           seed=42, position_range=(-60, 60), position_res=1,
           time_range=(0, 1), time_res=1, causes_kind="prob").run()

   Backend(stimuli=[Audio(-5, intensity=1, sigma=3), Visual(5, intensity=1, sigma=3)],
           integrator=AlaisBurr2004(), position_range=(-20, 20),
           position_res=0.1, time_range=(0, 1), time_res=1).run()
   ```

   Ojo: los estímulos duran 100 ms por defecto; con `time_range` menor a 100
   falla con "Stimuli total duration exceeds simulation length."

2. **Comparación contra el original.** Carpetas `testcode_cuppini/`,
   `testcode_kording/`, `testcode_alais_burr/`: cada `exampleN/` trae
   `sktest_og*.py` (implementación original, `skneuromsi`) y `sktest_new*.py`
   (esta), más `compare*.py`. Los `_og` y `compare` requieren el paquete
   original, que **no está instalado** en este entorno. El repo original vive en
   `/home/juan/misc/proyectos/skneuromsi/` (paquete en `src/skneuromsi`) y los
   scripts de comparación lo buscan en `<raíz>/../scikit-neuromsi`, ruta que
   no existe: hay que ajustarla o instalar el original. Usan `matplotlib`.

3. **Línea base numérica antes de refactorizar** (muy útil, se usó en los
   refactors del `SignalDispatcher`): correr los tres modelos con semilla
   fija, guardar `np.asarray(result.to_xarray())` y `result.causes_` con
   `np.savez`, y tras el cambio comparar con `np.array_equal`. Debe dar
   idéntico bit a bit si solo cambió la estructura.

4. **Criterio de equivalencia**: la idea de los ejemplos es que, con los
   mismos parámetros y semilla, la nueva arquitectura reproduzca al original.
   Cualquier refactor debería conservar ese comportamiento (idealmente
   convirtiendo estas comparaciones en tests).

Herramientas declaradas en `requirements_dev.txt`: `tox`, `pytest`,
`pytest-randomly`, `black` (línea de 79), `flake8` (+ import-order, black,
builtins), `check-manifest`, `coverage`, `pydocstyle`.

---

## 10. Convenciones de código

- **Idioma del código y docstrings: inglés.** Documentación de proceso en
  español está bien.
- Docstrings **numpy** (`Parameters`, `Returns`, `Raises`, `Notes`,
  `Attributes`). Fórmulas en LaTeX con `.. math::`. Si el docstring contiene
  `\`, debe ser **raw** (`r"""..."""`) o duplicar las barras; ya hubo un bug por
  esto (`\t` se volvía tab).
- `black` con `line-length = 79`.
- **Orden de métodos:** `__init__`, `__repr__`, privados (`_x`), públicos al
  final.
- Visibilidad: lo que no es contrato va con guion bajo. Antes de hacer
  privado un método, buscar usos con `grep` **real**.
- Parámetros opcionales de los modelos y del `Backend`: **solo por keyword**
  (`*`). En `Stimulus`, `position` es el único posicional de `Visual`/`Audio`.
- Cada integrador valida que `mode0 != mode1`.
- Cabecera de licencia (BSD-3, "Scikit-NeuroMSI Project, Renato Paredes;
  Cabral, Juan") en los archivos que la tenían; respetarla.

---

## 11. Decisiones de diseño

### 11.1 Ya tomadas por el autor (no revertir sin consultar)

- **Se mantiene el concepto del `Backend`:** conecta una colección de
  estímulos con un integrador; el **estímulo emite señales**. No se rediseña
  el contrato estímulo/señal/integrador de golpe (un boceto con `spec` y
  registro de codificadores se juzgó demasiado complejo).
- **Las señales se mantienen como concepto.** Se acepta que a futuro deberían
  ser **de un solo tipo** (hoy hay dos, `NeuralSignal` y `UnisensorySignal`);
  no está hecho ni definido cómo.
- **`Backend` es un mal nombre**, pero no se cambia por ahora. Candidato:
  `Experiment` (alternativa `Simulation`; evitar `Model`, que en el paper es el
  integrador).
- **Escala esperada:** *muchos* modelos pero a lo sumo ~3 familias nuevas
  (hoy: `Neural`, `Bayesian`, `MLE`). Por eso el costo de tocar `Stimulus` y
  `SignalDispatcher` al agregar una familia se acepta.
- **Estilo:** el autor prefiere bucles explícitos en `run()` a
  comprensiones, y soluciones simples a abstracciones elaboradas.

### 11.2 Abiertas

Preguntas que el código actual no resuelve; conviene decidirlas antes de
mover piezas:

1. **¿Dónde vive la codificación?** *Resuelta en parte.* El despacho por
   `model_type` vive en `SignalDispatcher` (clase de `backend.py`) como una
   tabla (`_ENCODERS`) y no como un `if`; `Stimulus` expone codificadores con
   argumentos explícitos y no depende del `Backend`. (Se probó antes que el
   dispatcher viviera en `Stimulus` con `generate_signal(backend)`; se
   descartó porque acoplaba el estímulo al `Backend`. Pasar al dispatcher el
   `Backend` entero, y no valores sueltos, fue una decisión del autor.) Quedan
   dos costos: agregar una *familia* de modelos exige tocar `Stimulus` y
   `SignalDispatcher` (se asume que serán pocas; los modelos nuevos dentro de
   una familia no tocan nada), y los adaptadores leen atributos del
   integrador (`neurons`, `n_sim`, `noise`) por convención, sin contrato
   formal. Alternativa futura: que el integrador declare qué señal necesita.
2. **`model_type` como string.** Tres valores literales repetidos en
   `Backend`, integradores y `NDResult`. ¿Enum o constante?
3. **Contrato del `Integrator`.** ¿Convertirlo en clase abstracta (`abc`) con
   `integrate`, `calculate_causes`, `model_type` y `model_parameters`?
4. **API de composición del paper (Code 6).** ¿Mantener `Backend(stimuli,
   integrator)` o ofrecer también `Model(stim, stim, integrator)`?
5. **Más de dos estímulos / modalidades distintas.** Todos los modelos
   exigen exactamente dos. El paper menciona táctil y otras modalidades
   (`mode0="tactile"` ya funciona por nombre, pero no hay `Tactile`) y el caso
   dos sonidos (Code 6, `m1`). Un `Stimulus` genérico ya sirve, pero no hay
   clase para otras modalidades.
6. **Estímulos 2D/3D.** El paper habla de 1D a 3D; hoy todo es 1D (y la
   dimensión `positions_coordinates` del `NDResult` existe pero se usa con
   una coordenada).
7. **Resultado estático en matriz tiempo×posición.** Kording y Alais-Burr
   rellenan una fila y dejan el resto en cero para ajustarse a `NDResult`.
   ¿Es la mejor representación?
8. **`ParameterSweep`.** Falta portarlo; depende del contrato final del
   `Backend`/integrador (en el original dependía de `ModelABC`, y por eso la
   "inversión de dependencias" era el argumento central del paper).
9. **`io`/`ndcollection`.** ¿Repararlos (recuperar `dict_cmp`,
   `custom_json`) o excluirlos de esta versión?
10. **Tests.** Convertir los `compare*.py` en tests de regresión numérica contra
   el original (con tolerancia) y agregar tests de contrato por integrador.

---

## 12. Reglas de trabajo para agentes

- **No hacer commit ni push sin que se pida.** Antes de acciones difíciles de
  revertir, confirmar. El autor commitea con `git aicommit` (script propio en
  `/usr/local/bin/git-aicommit` que genera el mensaje con Claude a partir de
  lo que está *staged*; con `-a` hace `git add` de todo, sin `-a` solo
  commitea lo ya staged) y hace el push él mismo o lo pide explícitamente.
- **Antes de borrar o renombrar algo público, buscar usos con el `grep`
  real.** En algunos entornos `grep` es un wrapper que respeta `.gitignore` y
  puede devolver vacío en falso; usar `command grep -rn ... --exclude-dir=.git`
  (y excluir `lib/`, `bin/`, `share/`, que son el venv dentro de la carpeta
  padre).
- Si se cambia comportamiento numérico (no solo estructura), contrastarlo con
  los ejemplos `testcode_*` o con el original.
- Verificar siempre con un `Backend(...).run()` real de **los tres** modelos
  después de un cambio en `Stimulus`, `Integrator`, `Backend` o señales.
- Se trabaja sobre un repo con cambios del usuario sin commitear: leer el
  archivo de nuevo antes de editarlo y respetar lo que haya cambiado.
- Ante la duda sobre el diseño, la referencia es la sección *Availability and
  Future Directions* del paper (`draft/`) y la Eq. 1: **estímulo e integración
  deben poder variar de forma independiente**.

---

## 13. Referencias y datos de contexto

- Paper: Paredes, R., Cabral, J. B., & Seriès, P. (2025). *Scikit-NeuroMSI: A
  Generalized Framework for Modeling Multisensory Integration.*
  Neuroinformatics, 23:40. En `draft/s12021-025-09738-1.pdf` (versionado). Secciones útiles:
  *Availability and Future Directions* (p. 12), Apéndice A (estandarización
  formal), B (clases), C (modelos), D (lecturas de causas).
- Modelos: Alais & Burr (2004); Körding et al. (2007); Cuppini et al. (2017).
  Implementación de Matlab que inspira a Kording2007: BCI Toolbox (Zhu,
  Beierholm & Shams, 2024).
- Remoto git: `github.com/leliel12/skneuromsi-next-generation`. Raíz del git:
  `.../sknmsi_tng/src` (el directorio padre `sknmsi_tng` contiene además un
  venv: `bin/`, `lib/`, `share/`, `pyvenv.cfg`).
- Python del entorno: 3.13; `requires-python >= 3.10`.
- Autoría en `pyproject.toml`: Renato Paredes y Juan B. Cabral.
