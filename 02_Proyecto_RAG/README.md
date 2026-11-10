# Asistente RAG · NébulaLab (Misión 10)

Asistente que responde preguntas sobre las políticas internas de NébulaLab (empresa ficticia) a partir de sus
documentos, citando la fuente de cada dato, rechazando lo que no consta y distinguiendo los documentos vigentes
de los obsoletos.

## Requisitos e instalación
Python 3.10–3.12.

```bash
cd 02_Proyecto_RAG
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso sin API (TF-IDF local)
```bash
python -m src.rag index
python -m src.rag ask "¿Cuántos días de teletrabajo se permiten?"
python -m src.rag ask "¿Cuántos días de teletrabajo permitía el borrador anterior?"   # activa el modo histórico
python -m src.rag evaluate --file ../03_Pruebas/preguntas_ampliadas.csv
```
Sin `.env` el sistema usa TF-IDF y la respuesta es extractiva (muestra los fragmentos y su fuente).

## Uso con Gemini
1. Copiar `.env.example` a `.env` y rellenar `GEMINI_API_KEY` (no subir `.env` al repositorio).
2. `python -m src.rag index` (hay que repetirlo cada vez que se cambie `EMBEDDING_BACKEND`).
3. `python -m src.rag ask "¿Cómo solicito vacaciones?" --gemini`

Si el modelo configurado ya no existe, `python -m src.rag modelos` lista los disponibles para la clave.
`GEMINI_MODEL` admite varios modelos separados por comas y prueba el siguiente si uno falla.

## Opciones de recuperación (`ask` y `evaluate`)
| Opción | Por defecto | Qué hace |
|---|---|---|
| `-k` | 3 | Número máximo de fragmentos recuperados |
| `--modo auto\|vigente\|historico` | `auto` | `vigente` filtra los documentos obsoletos; `historico` los incluye; `auto` decide según la pregunta |
| `--penal-secundarias` | 0.1 | Distancia que se suma a la FAQ (fuente secundaria); `0` la desactiva |
| `--umbral` | desactivado | Si el mejor fragmento supera esa distancia, la pregunta se rechaza sin llamar al modelo |
| `--margen` | desactivado | Solo pasan al modelo los fragmentos a esa distancia del mejor (k dinámico) |
| `ask --prompt estricto\|mejorado\|original` | `estricto` | Prompt de generación |
| `index --size --overlap --sin-titulo` | 650 / 100 | Fragmentación |

`UMBRAL_RECHAZO` y `MARGEN_CONTEXTO` se pueden fijar en `.env`.
Configuración equivalente al kit original: `evaluate --modo historico --penal-secundarias 0`.

## Experimentos y resultados
```bash
python ../03_Pruebas/experimentos.py                    # recuperación con TF-IDF (sin API)
python ../03_Pruebas/experimentos.py --backend gemini   # recuperación con embeddings de Gemini (2 llamadas)
python ../03_Pruebas/respuestas_gemini.py               # respuestas generadas y evaluación automática (4 llamadas)
python ../03_Pruebas/respuestas_gemini.py --prompts estricto --margen 0.1   # lo mismo con recorte del contexto
```
Los resultados se guardan en `03_Pruebas/resultados/`:
- `retrieval_tfidf.txt` y `retrieval_gemini.txt`: métricas de recuperación antes/después, por tipo de pregunta,
  detector de preguntas históricas, recorte por margen y umbral de rechazo.
- `respuestas_gemini_<prompt>.md`: las 44 respuestas con sus fuentes y la comprobación automática de cada una.
- `resumen_generacion_gemini.txt`: exactitud, cita correcta, rechazo y fidelidad por prompt.

Las llamadas a la API van en lote y se guardan en `.cache/` (embeddings y respuestas, sin la clave). El repositorio
incluye la caché de nuestra ejecución, así que los tres comandos anteriores reproducen los resultados sin clave y sin
consumir cuota. `--exigir-modelo MODELO` repite los lotes que respondió un modelo distinto.

## Conjunto de pruebas
- `03_Pruebas/preguntas_prueba.csv`: las 20 preguntas del kit.
- `03_Pruebas/preguntas_ampliadas.csv`: 44 preguntas (las 20 del kit y 24 nuevas) con tipo de pregunta
  (factual, detalle, paráfrasis, comparación, histórica, conflicto y trampa), todas las fuentes relevantes
  (`fuentes_relevantes`) y una expresión regular con el dato esperado (`clave`).

## Archivos
- `src/rag.py`: carga, fragmentación, metadatos, embeddings, índice Chroma, recuperación, generación y evaluación.
- `data/`: documentación ficticia de NébulaLab.
- `.env.example`: plantilla de configuración.

## Seguridad
Documentos ficticios, sin datos personales. La clave de la API solo está en `.env`, excluido del repositorio.
Una cita `[F1]` indica de dónde sale el dato; no garantiza por sí sola que sea correcto ni vigente.
