# Asistente RAG · NébulaLab

Misión 10 («Cuando la IA conoce nuestros datos») de Aplicaciones y Tendencias del Análisis de Datos,
Universidad Europea de Madrid.

**Equipo:** Guillermo Angulo · Eva Nieto · Sergio Gama

Asistente que responde preguntas sobre las políticas internas de NébulaLab (empresa ficticia) a partir de sus
propios documentos: recupera los fragmentos relevantes con Chroma, genera la respuesta con Gemini citando la
fuente de cada dato, rechaza lo que no consta en el corpus y distingue los documentos vigentes de los obsoletos.

## Estructura
| Carpeta | Contenido |
|---|---|
| `02_Proyecto_RAG/` | Código (`src/rag.py`), documentos (`data/`), configuración y [README de uso](02_Proyecto_RAG/README.md) |
| `03_Pruebas/` | Conjunto de pruebas (44 preguntas), scripts de experimentos y [resultados](03_Pruebas/resultados/) |
| `01_Guias/`, `04_Evaluacion/`, `05_Plantillas/`, `06_Recursos/` | Material del kit de la misión |

## Puesta en marcha
```bash
cd 02_Proyecto_RAG
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.rag index
python -m src.rag ask "¿Cuántos días de teletrabajo se permiten?"
```
Sin clave funciona en local con TF-IDF. Para usar Gemini, copiar `.env.example` a `.env` y rellenar
`GEMINI_API_KEY`. Los experimentos se pueden reproducir sin clave porque el repositorio incluye la caché de
embeddings y respuestas de nuestra ejecución. Todas las opciones están en el
[README de uso](02_Proyecto_RAG/README.md).

## Resultados principales
Conjunto ampliado: 36 preguntas con respuesta y 8 preguntas trampa. Embeddings `gemini-embedding-001`, k = 3.

| Configuración de recuperación | Hit@1 | Hit@3 | MRR@3 |
|---|---|---|---|
| Kit original | 86,1 % | 97,2 % | 0,917 |
| Filtro de obsoletos + jerarquía de fuentes | 86,1 % | 88,9 % | 0,870 |
| + modo de vigencia automático | 97,2 % | 100 % | 0,981 |

Con el recorte del contexto por margen (0,1), la precisión de la recuperación pasa de 0,42 a 0,75 sin perder
recall (0,93). Con la configuración final y el prompt estricto, Gemini responde bien las 36 preguntas con
respuesta y rechaza las 8 preguntas trampa. El detalle está en `03_Pruebas/resultados/`.
