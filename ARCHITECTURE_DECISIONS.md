# 🛡️ Architecture Decision Record (ADR) & Ensayo de Defensa Técnica

Este documento recoge y justifica las decisiones de diseño arquitectónico y matemático que asientan la madurez de este proyecto frente a escrutinios técnicos exigentes (entrevistas, auditorías de código, MLOps reviews).

## 1. ¿Por qué priorizar radicalmente el Recall sobre la Precision?
*(Ataque potencial: "Tu modelo genera Falsos Positivos, estás enviando mecánicos a revisar motores sanos y costando dinero a la empresa").*

**Defensa:** En entornos de mantenimiento crítico aeroespacial o industrial, la matriz de costes es altamente asimétrica. Un Falso Positivo cuesta ~\$2,000 (horas de inspección y pérdida menor de operatividad). Un Falso Negativo (un motor colapsando en pleno vuelo porque el modelo dijo que estaba "Sano") cuesta decenas de millones de dólares por pérdida de hardware, compensaciones y daño irreparable a la reputación de la marca. 

Por ello, entrenamos a XGBoost con `scale_pos_weight` buscando interceptar >90% de los fallos, asumiendo conscientemente el repunte de Falsas Alarmas; porque revisar un motor extra siempre será órdenes de magnitud más barato que un siniestro en pista.

## 2. ¿Por qué usar GroupShuffleSplit (`unit_id`) y no un TimeSeriesSplit aleatorio clásico?
*(Ataque potencial: "Deberías haber usado K-Fold clásico o un split temporal puro sobre todo el dataframe").*

**Defensa:** Aplicar un `train_test_split` aleatorio puro sobre filas de series temporales industriales destruye la validación en IoT. Si un motor `nº 44` tiene 200 filas de vuelo y caen aleatoriamente 180 al dataset de Entrenamiento y 20 al de Validación, el modelo "memorizará" la firma termodinámica de ese motor `nº 44`. Cuando intentemos evaluarlo en esas 20 filas ocultas, la predicción será artificialmente alta por culpa del *Data Leakage*. 

Al usar un split agrupado estrictamente por `unit_id`, garantizamos que si el motor `nº 50` cae en Validación, el algoritmo **jamás** ha visto ni un milisegundo de su telemetría. Esto certifica que nuestras métricas de inferencia representan verdaderamente la capacidad de generalización sobre hardware totalmente nuevo.

## 3. ¿Por qué forzar un K-Means Clustering antes de procesar las series temporales en FD002?
*(Ataque potencial: "En FD002 podrías haberle dado los datos crudos a XGBoost, al ser basado en árboles, internamente habría encontrado las divisiones por régimen climático él solo").*

**Defensa:** Es cierto que XGBoost divide hiperplanos excepcionalmente, pero en FD002 la degradación sutil del motor queda sepultada por cambios masivos en la altitud y los números Mach. Si leemos esto ingenuamente con un *Rolling Window* estándar, sacaremos medias cruzadas entre la Temperatura a Nivel del Mar y a 30.000 pies de altura, inyectando un ruido aniquilador en la serie de tiempo de desgaste.

Al anticiparnos e inyectar *K-Means* como Inteligencia No-Supervisada, particionamos dictatorialmente la atmósfera en 6 "burbujas" operacionales (*Flight Regimes*). Forzando a cada burbuja a ejecutarse bajo un sub-sistema `StandardScaler` independiente, neutralizamos por completo el impacto del clima antes de que ensucie la serie. Sobre esa señal químicamente pura, ahora sí calculamos medias móviles. Ese es el verdadero responsable de que el *Recall* en el test multirégimen alcanzase el 97%.

## 4. ¿Por qué enfocar el target como Clasificación Binaria en vez de Regresión Numérica pura (RUL)?
*(Ataque potencial: "El target original de la NASA es encontrar el Remaining Useful Life (RUL) exacto, ¿por qué lo binarizaste a 'Fallo inminente'?").*

**Defensa:** Desde el punto de vista puramente algorítmico, intentar predecir que a un turbofán le quedan "123.4 vuelos" frente a "125 vuelos" en su ecuador de vida es ruido matemático sin valor de negocio. El desgaste físico de un motor no es lineal al principio de su ciclo. 
Lo que un departamento de Operaciones MRO (Maintenance, Repair, Overhaul) necesita es accionar un Protocolo Operativo. Reenfocar la señal cruzando un umbral binario de urgencia técnica (A los 30 ciclos) alinea el algoritmo con la realidad física. Resulta estadísticamente más robusto y logísticamente más útil decirle a los ingenieros "Luz Roja: Peligro de colapso crítico inmediato", en lugar de devolver predicciones numéricas con alta varianza inútil en las primeras etapas sanas del aparato.

## 5. ¿Por qué exponer el modelo con FastAPI Asíncrono en lugar de un proceso Batch Nocturno (Cron/Airflow)?
*(Ataque potencial: "Para rutinas de mantenimiento pesadas suele bastar con correr los modelos masivamente una vez a la noche. ¿Por qué crear una REST API de microservicio?").*

**Defensa:** El despliegue Batch clásico sirve para sustituir correas dentadas de coche periódicamente, pero resulta inaceptable si una turbina de 4 millones de dólares manifiesta una vibración anómala en pleno aterrizaje y debemos dictaminar la viabilidad de su despegue a la mañana siguiente.
Levantar y "sembrar" el pipeline completo Híbrido (K-Means + XGBoost) dentro de la memoria RAM de un contenedor asíncrono con FastAPI nos traslada al paradigma "Event-Driven Maintenance" (Mantenimiento reactivo de ultra-baja latencia). La ráfaga de telemetría de vuelo puede enviarse directamente contra el Endpoint tras tocar tierra e interceptar el desastre mecácnico en 40 milisegundos.

## 6. ¿Por qué usar Teoría de Juegos (Calculo SHAP) en vez del "Feature Importance" predeterminado del XGboost?
*(Ataque potencial: "Calcular valores SHAP requiere peso extra. ¿Por qué no sacaste la gráfica nativa de Importance del árbol?").*

**Defensa:** El "Feature Importance" nativo (*Gini / Weight*) promedia la capacidad destructiva del árbol pero es mudo: no indica dirección causal (es decir, no te dice si la presión alta rompe el motor o lo cura) y sobre todo, explica el bosque y no el árbol. 
Nuestros usuarios finales asumen grandes responsabilidades de riesgo. Cuando al ingeniero le salta nuestra Alarma Roja en Streamlit, recordarle que "El Sensor_14 suele ser muy importante para tu flota" no le sirve. Él debe saber qué fue mal *hoy* y en *este avión* para desmontar sus piezas. SHAP (*SHapley Additive exPlanations*) nos provee Explicabilidad Local, destrozando la caja negra del modelo para mostrarle qué sensor térmico concreto acorraló al motor hacia el desastre aquel jueves por la tarde. Esto, bajo estricta ley de auditoría internacional (ej. Europea), blinda legalmente la adopción de Big Data.
