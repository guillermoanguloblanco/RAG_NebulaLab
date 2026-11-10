# Evaluación de RAG
## Dos planos que no deben confundirse
- **Recuperación:** Hit@k y MRR@k miden si el documento esperado aparece en los primeros k resultados. El script `evaluate` calcula estas métricas sobre preguntas con fuente conocida.
- **Generación:** exactitud factual, completitud, cita correcta, rechazo fundamentado y detección de contradicciones requieren revisar las respuestas (evaluación humana o protocolo independiente).

## Experimento antes/después
1. Indexa con `--size 650 --overlap 100`, ejecuta `evaluate` y anota métricas.
2. Cambia el tamaño de chunk (por ejemplo 300) y repite; compara.
3. Prueba k=1 frente a k=3; compara.
4. Añade metadatos o mejora los documentos y documenta el cambio.
5. Formula preguntas ausentes y comprueba que Gemini no inventa información.
6. Usa el borrador de teletrabajo obsoleto para evaluar cómo gestiona contradicciones.

**Importante:** el pequeño conjunto de prueba es didáctico y no constituye un benchmark científico; evita optimizar solo para estas preguntas.
