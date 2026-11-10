"""Genera con Gemini las respuestas de todo el conjunto de prueba y las evalúa automáticamente.

Para gastar el mínimo de cuota, las preguntas se envían en lotes (una llamada por lote, cada pregunta con su
propio contexto) y todo queda en caché: si se corta por cuota, al relanzarlo continúa donde se quedó.

Uso (desde 02_Proyecto_RAG, con GEMINI_API_KEY en .env):
    python ../03_Pruebas/respuestas_gemini.py                      # prompts mejorado y estricto, lotes de 22
    python ../03_Pruebas/respuestas_gemini.py --prompts estricto --margen 0.1   # con recorte del contexto

Escribe 03_Pruebas/resultados/respuestas_<backend>_<prompt>.md y resumen_generacion_<backend>.txt
"""
import argparse, os, re, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_Proyecto_RAG'))

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['tfidf', 'gemini'], default='gemini', help='embeddings para la recuperación')
p.add_argument('--prompts', default='mejorado,estricto')
p.add_argument('--lote', type=int, default=22, help='preguntas por llamada')
p.add_argument('-k', type=int, default=3)
p.add_argument('--margen', type=float, help='recorte del contexto: solo pasan los fragmentos a esta distancia del mejor')
p.add_argument('--umbral', type=float, help='umbral de rechazo en la recuperación')
p.add_argument('--exigir-modelo', help='usa solo este modelo y repite los lotes que respondió otro')
args = p.parse_args()
os.environ['EMBEDDING_BACKEND'] = args.backend
if args.exigir_modelo: os.environ['GEMINI_EXIGIR_MODELO'] = args.exigir_modelo
from src import rag

HERE = Path(__file__).parent
OUT = HERE / 'resultados'; OUT.mkdir(exist_ok=True)
ROWS = rag.load_questions(HERE / 'preguntas_ampliadas.csv')
OPTS = rag.defaults()
if args.margen is not None: OPTS['margen'] = args.margen
if args.umbral is not None: OPTS['umbral'] = args.umbral
SUFFIX = (f"_margen{OPTS['margen']}" if OPTS['margen'] is not None else '') + (f"_umbral{OPTS['umbral']}" if OPTS['umbral'] is not None else '')

def plain(text):
    text = unicodedata.normalize('NFD', text.lower())
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

def check(row, sources, answer):
    """Comprobaciones automáticas de una respuesta (no sustituyen a la revisión manual)."""
    rejected = plain(rag.RECHAZO.rstrip('.')) in plain(answer)
    cited = sorted({int(n) for n in re.findall(r'\[F(\d+)', answer)})
    cited_sources = [sources[n - 1]['source'] for n in cited if 0 < n <= len(sources)]
    context = ' '.join(s['text'] + ' ' + ' '.join(str(v) for v in s['meta'].values()) for s in sources)
    numbers = re.findall(r'\d+', re.sub(r'\[F\d+(?:,\s*F\d+)*\]', '', answer))
    result = {'rechazo': rejected, 'citas': cited_sources,
              # fidelidad: todas las cifras de la respuesta aparecen en los fragmentos recibidos
              'cifras_respaldadas': all(n in context for n in numbers),
              'cita_inexistente': any(n < 1 or n > len(sources) for n in cited)}
    if row['relevantes']:
        result['correcta'] = (not rejected) and bool(re.search(row['clave'], answer, re.I))
        result['cita_correcta'] = bool(cited_sources) and all(any(rel in s for rel in row['relevantes']) for s in cited_sources)
    else:
        result['correcta'] = rejected
        result['texto_anadido'] = rejected and len(answer.strip()) > len(rag.RECHAZO) + 5
    return result

rag.index(650, 100, quiet=True)   # con caché: no gasta llamadas si los documentos ya se indexaron
rag.warm_up(r['pregunta'] for r in ROWS)
retrieved = {r['n']: rag.retrieve(r['pregunta'], args.k, **OPTS) for r in ROWS}
summary = [f'Backend: {rag.backend()} | k={args.k} | opciones de recuperación: {OPTS} | lotes de {args.lote} preguntas',
           f'{len(ROWS)} preguntas: {sum(bool(r["relevantes"]) for r in ROWS)} con respuesta y {sum(not r["relevantes"] for r in ROWS)} trampa', '']

for version in args.prompts.split(','):
    answers, models = {}, []
    for i in range(0, len(ROWS), args.lote):
        items = [(r['n'], r['pregunta'], retrieved[r['n']]) for r in ROWS[i:i + args.lote]]
        try:
            answers.update(rag.gemini_answer_batch(items, version))
            used = rag.batch_model(items, version)
            if used: models.append(f'preguntas {items[0][0]}-{items[-1][0]}: {used}')
        except Exception as ex:
            print(f'ERROR en el lote {i // args.lote + 1} ({version}): {type(ex).__name__}: {str(ex)[:300]}', flush=True)
            print('Lo ya respondido está en caché: vuelve a lanzar el script más tarde para continuar.')
            break
    lines = [f'# Respuestas de Gemini · prompt {version} · embeddings {rag.backend()}', '']
    checks = {}
    for r in ROWS:
        n, sources = r['n'], retrieved[r['n']]
        if n not in answers: continue
        answers[n] = answers[n].strip()
        c = checks[n] = check(r, sources, answers[n])
        flags = ['correcta' if c['correcta'] else 'REVISAR']
        if r['relevantes']: flags.append('cita correcta' if c['cita_correcta'] else 'cita a revisar')
        if not c['cifras_respaldadas']: flags.append('CIFRA NO RESPALDADA')
        if c.get('texto_anadido'): flags.append('texto añadido al rechazo')
        lines += [f"## {n}. {r['pregunta']}", f"**Tipo:** {r['tipo']} · **Esperado:** {r['respuesta_esperada']} ({'; '.join(x[:2] for x in r['relevantes']) or 'sin fuente'})",
                  '**Recuperado:** ' + (' · '.join(f"[{s['id']}] {s['source'][:2]} ({s['distance']})" for s in sources) + f" · modo {sources[0]['modo']}" if sources else 'nada (rechazada por el umbral)'),
                  f"**Comprobación automática:** {' · '.join(flags)}", '', answers[n], '']
    lines[1:1] = ['', 'Modelo: ' + ' · '.join(models)]
    (OUT / f'respuestas_{rag.backend()}_{version}{SUFFIX}.md').write_text('\n'.join(lines), encoding='utf-8')

    ans = [r for r in ROWS if r['relevantes'] and r['n'] in checks]
    traps = [r for r in ROWS if not r['relevantes'] and r['n'] in checks]
    rate = lambda rows, key: f"{sum(bool(checks[r['n']].get(key)) for r in rows)}/{len(rows)}"
    words = lambda rows: sum(len(answers[r['n']].split()) for r in rows) / (len(rows) or 1)
    summary += [f'## Prompt {version} ({len(checks)}/{len(ROWS)} preguntas respondidas)',
                'Modelo: ' + ' · '.join(models),
                f"Fragmentos de contexto por pregunta (media): {sum(len(retrieved[r['n']]) for r in ROWS) / len(ROWS):.2f}",
                f'Longitud media de la respuesta (palabras) · con respuesta: {words(ans):.1f} · trampa: {words(traps):.1f}',
                f'Preguntas con respuesta · exactitud (contiene el dato esperado): {rate(ans, "correcta")}',
                f'Preguntas con respuesta · cita correcta (todas las citas son fuentes relevantes): {rate(ans, "cita_correcta")}',
                f'Preguntas con respuesta · rechazadas por error: {rate(ans, "rechazo")}',
                f'Preguntas trampa · rechazo correcto: {rate(traps, "correcta")}',
                f'Preguntas trampa · rechazo con texto añadido: {rate(traps, "texto_anadido")}',
                f'Fidelidad · respuestas con todas las cifras respaldadas por el contexto: {rate(ans + traps, "cifras_respaldadas")}',
                f'Citas a fragmentos inexistentes: {rate(ans + traps, "cita_inexistente")}',
                'Exactitud por tipo: ' + ' · '.join(f"{t} {rate([r for r in ans + traps if r['tipo'] == t], 'correcta')}" for t in dict.fromkeys(r['tipo'] for r in ROWS)),
                'A revisar a mano: ' + (', '.join(str(r['n']) for r in ans + traps if not checks[r['n']]['correcta'] or (r['relevantes'] and not checks[r['n']]['cita_correcta']) or not checks[r['n']]['cifras_respaldadas']) or 'ninguna'), '']

text = '\n'.join(summary)
(OUT / f'resumen_generacion_{rag.backend()}{SUFFIX}.txt').write_text(text + '\n', encoding='utf-8')
print('\n' + text)
