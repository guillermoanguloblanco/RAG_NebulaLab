"""Experimentos de recuperación (antes/después) sobre el conjunto de prueba ampliado.

Uso (desde 02_Proyecto_RAG):
    python ../03_Pruebas/experimentos.py                    # TF-IDF, sin API
    python ../03_Pruebas/experimentos.py --backend gemini   # embeddings de Gemini (2 llamadas en lote, con caché)

Escribe los resultados en 03_Pruebas/resultados/retrieval_<backend>.txt
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '02_Proyecto_RAG'))

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['tfidf', 'gemini'], default='tfidf')
args = p.parse_args()
os.environ['EMBEDDING_BACKEND'] = args.backend
from src import rag

HERE = Path(__file__).parent
ROWS = rag.load_questions(HERE / 'preguntas_ampliadas.csv')
KIT = [r for r in ROWS if r['origen'] == 'kit']
OUT = []

def say(text=''):
    print(text, flush=True); OUT.append(text)

def line(name, rows, **opts):
    m1 = rag.retrieval_metrics(rows, 1, **opts)
    m3 = rag.retrieval_metrics(rows, 3, **opts)
    say(f"{name:44} {m1['hit']:7.1%} {m3['hit']:7.1%} {m3['mrr']:7.3f} {m3['precision']:7.3f} {m3['recall']:7.3f} {m3['f1']:7.3f} {m3['contexto']:6.2f}")
    return m3

HEAD = f"{'Configuración':44} {'Hit@1':>7} {'Hit@3':>7} {'MRR@3':>7} {'P@3':>7} {'R@3':>7} {'F1@3':>7} {'ctx':>6}"
KIT_BASE = dict(modo='historico', secondary_penalty=0.0)     # código del kit: sin filtro ni jerarquía
PARTIDA = dict(modo='vigente', secondary_penalty=0.1)         # filtro de obsoletos + jerarquía de fuentes
AUTO = dict(modo='auto', secondary_penalty=0.1)               # + modo de vigencia automático

rag.index(650, 100, quiet=True)
rag.warm_up(r['pregunta'] for r in ROWS)
n_ans = sum(bool(r['relevantes']) for r in ROWS)
say(f'Backend de embeddings: {args.backend} | fragmentos 650/100 | {len(ROWS)} preguntas ({n_ans} con respuesta, {len(ROWS) - n_ans} trampa)')
say('ctx = número medio de fragmentos que se pasan al modelo')

say('\n## 1. Conjunto del kit (17 preguntas con fuente)')
say(HEAD)
line('Base (kit)', KIT, **KIT_BASE)
line('Filtro de obsoletos', KIT, modo='vigente', secondary_penalty=0.0)
line('Jerarquía de fuentes (FAQ +0,1)', KIT, modo='historico', secondary_penalty=0.1)
line('Filtro + jerarquía', KIT, **PARTIDA)
line('Filtro + jerarquía + modo automático', KIT, **AUTO)

say(f'\n## 2. Conjunto ampliado ({n_ans} preguntas con respuesta)')
say(HEAD)
line('Base (kit)', ROWS, **KIT_BASE)
line('Filtro + jerarquía', ROWS, **PARTIDA)
final = line('Filtro + jerarquía + modo automático', ROWS, **AUTO)

say('\n## 3. Hit@1 y Recall@3 por tipo de pregunta (conjunto ampliado; entre paréntesis, número de preguntas)')
tipos = list(dict.fromkeys(r['tipo'] for r in ROWS if r['relevantes']))
say(f"{'Configuración':44} " + ' '.join(f"{t + ' (' + str(sum(r['tipo'] == t for r in ROWS)) + ')':>16}" for t in tipos))
for name, opts in [('Base (kit)', KIT_BASE), ('Filtro + jerarquía', PARTIDA), ('Filtro + jerarquía + modo automático', AUTO)]:
    cells = []
    for t in tipos:
        sub = [r for r in ROWS if r['tipo'] == t]
        cells.append(f"{rag.retrieval_metrics(sub, 1, **opts)['hit']:.0%} / {rag.retrieval_metrics(sub, 3, **opts)['recall']:.2f}".rjust(16))
    say(f'{name:44} ' + ' '.join(cells))

say('\n## 4. Detector de preguntas históricas (44 preguntas)')
metas = [rag.parse_header(text) for _, text in rag.load_docs()]
tp = fp = fn = tn = 0
for r in ROWS:
    pred, real = rag.is_historical(r['pregunta'], metas), r['tipo'] == 'historica'
    tp += pred and real; fp += pred and not real; fn += real and not pred; tn += not pred and not real
    if pred != real: say(f"  ERROR ({'falso positivo' if pred else 'falso negativo'}): {r['pregunta']}")
say(f'Históricas detectadas: {tp}/{tp + fn} | falsos positivos: {fp}/{fp + tn}')

say('\n## 5. Recorte del contexto por margen de distancia (k=3, configuración con modo automático)')
say(f"{'margen':>8} {'P@3':>7} {'R@3':>7} {'F1@3':>7} {'Hit@3':>7} {'ctx':>6}")
for margen in [None, 0.2, 0.15, 0.1, 0.075, 0.05, 0.03, 0.02]:
    m = rag.retrieval_metrics(ROWS, 3, margen=margen, **AUTO)
    say(f"{str(margen):>8} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['hit']:7.1%} {m['contexto']:6.2f}")

say('\n## 6. Umbral de rechazo: distancia del mejor resultado (configuración con modo automático)')
dist = [(rag.retrieve(r['pregunta'], 1, **AUTO)[0]['distance'], bool(r['relevantes']), r) for r in ROWS]
yes = sorted(d for d, ok, _ in dist if ok); no = sorted(d for d, ok, _ in dist if not ok)
say(f'con respuesta: mín {yes[0]:.3f} · mediana {yes[len(yes) // 2]:.3f} · máx {yes[-1]:.3f}')
say(f'sin respuesta: mín {no[0]:.3f} · mediana {no[len(no) // 2]:.3f} · máx {no[-1]:.3f}')
say(f"{'umbral':>8} {'trampas rechazadas':>20} {'respuestas perdidas':>20}")
steps = sorted({round(x, 3) for x in [no[0] - 0.02, no[0] - 0.005, (yes[-1] + no[0]) / 2, yes[-1] + 0.005, no[len(no) // 2], no[-1] + 0.005]})
for u in steps:
    say(f'{u:8.3f} {sum(d > u for d in no):>14}/{len(no):<5} {sum(d > u for d in yes):>14}/{len(yes):<5}')
say(f'Umbral más bajo que no pierde ninguna respuesta ({yes[-1]:.3f}): rechaza {sum(d > yes[-1] for d in no)}/{len(no)} trampas; '
    f'{sum(d <= yes[-1] for d in no)} trampas quedan dentro del rango de las preguntas con respuesta')
say('Preguntas sin respuesta y la más lejana con respuesta:')
for d, ok, r in sorted(dist, key=lambda x: -x[0])[:len(no) + 3]:
    say(f"  {d:.3f} | {'con respuesta' if ok else 'TRAMPA       '} | {r['pregunta']}")

say('\n## 7. Preguntas cuya fuente principal no sale la primera (configuración con modo automático, k=3)')
for d in final['detalle']:
    if d['rank'] != 1: say(f"  {d['n']:>2}. {d['pregunta']} | rango={d['rank']} | recuperado={d['fuentes']} | tipo={d['tipo']}")

if args.backend == 'tfidf':
    say('\n## 8. Tamaño de fragmento (TF-IDF, configuración con modo automático)')
    say(f"{'size/overlap':14} {'frag.':>5} " + HEAD[45:])
    for size, overlap in [(200, 40), (300, 50), (450, 80), (650, 100), (1000, 150)]:
        n = rag.index(size, overlap, quiet=True)
        m1 = rag.retrieval_metrics(ROWS, 1, **AUTO); m3 = rag.retrieval_metrics(ROWS, 3, **AUTO)
        say(f"{size:>4}/{overlap:<9} {n:5} {m1['hit']:7.1%} {m3['hit']:7.1%} {m3['mrr']:7.3f} {m3['precision']:7.3f} {m3['recall']:7.3f} {m3['f1']:7.3f} {m3['contexto']:6.2f}")
    rag.index(650, 100, quiet=True)

out = HERE / 'resultados'
out.mkdir(exist_ok=True)
(out / f'retrieval_{args.backend}.txt').write_text('\n'.join(OUT) + '\n', encoding='utf-8')
print(f'\nGuardado en {out / f"retrieval_{args.backend}.txt"}')
