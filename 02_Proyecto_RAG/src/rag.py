"""Misión 10 · Asistente RAG de NébulaLab.

Flujo: documentos -> fragmentos -> embeddings (TF-IDF local o Gemini) -> Chroma -> recuperación -> respuesta.

Sobre el código del kit, este módulo añade:
  - metadatos de vigencia y tipo de fuente extraídos de la cabecera de cada documento;
  - filtro de documentos obsoletos y jerarquía de fuentes (la FAQ es secundaria);
  - modo de vigencia automático: las preguntas históricas consultan también los documentos obsoletos;
  - recorte del contexto por margen de distancia (k dinámico) y umbral de rechazo;
  - prompt estricto con metadatos, frase de rechazo fija y temperatura baja;
  - caché en disco de embeddings y respuestas, y llamadas en lote, para gastar el mínimo de cuota de la API.
"""
import argparse, csv, hashlib, json, os, re, time, unicodedata
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
DB = BASE / '.chroma'
CACHE = BASE / '.cache'
COLLECTION = 'nebulalab'
API_TIMEOUT_MS = 120000
RECHAZO = 'No consta en los documentos disponibles.'
load_dotenv(BASE / '.env')


# ----------------------------------------------------------------------------- configuración

def backend():
    return os.getenv('EMBEDDING_BACKEND', 'tfidf').lower()

def _env_float(name):
    value = os.getenv(name, '').strip()
    return float(value) if value else None

def defaults():
    """Opciones de recuperación por defecto; UMBRAL_RECHAZO y MARGEN_CONTEXTO se pueden fijar en .env."""
    return {'modo': 'auto', 'secondary_penalty': 0.1,
            'umbral': _env_float('UMBRAL_RECHAZO'), 'margen': _env_float('MARGEN_CONTEXTO')}


# ----------------------------------------------------------------------------- carga y fragmentación

def load_docs():
    docs = []
    for path in sorted(DATA.iterdir()):
        if path.suffix.lower() == '.txt':
            docs.append((path.name, path.read_text(encoding='utf-8')))
        elif path.suffix.lower() == '.pdf':
            for n, page in enumerate(PdfReader(str(path)).pages, 1):
                text = page.extract_text() or ''
                if text.strip(): docs.append((f'{path.name} (pág. {n})', text))
    return docs

def chunks(text, size=650, overlap=100):
    if size <= overlap: raise ValueError('size debe superar overlap')
    text = re.sub(r'\s+', ' ', text).strip()
    if not text: return []
    # El último inicio es len-overlap: así no se genera un fragmento final contenido entero en el anterior
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), size - overlap) if text[i:i + size].strip()]

def parse_header(text):
    """Metadatos de la cabecera: DOCUMENTO / Versión / Vigente desde / Responsable / Estado."""
    def field(name):
        found = re.search(name + r':\s*([^|\n]+)', text)
        return found[1].strip() if found else ''
    title = field('DOCUMENTO')
    obsolete = bool(re.search(r'OBSOLETO|NO VIGENTE', text[:300], re.I))
    return {'titulo': title, 'version': field('Versión'), 'vigente_desde': field('Vigente desde'),
            'responsable': field('Responsable'), 'estado': 'obsoleto' if obsolete else 'vigente',
            'tipo': 'secundaria' if 'preguntas frecuentes' in title.lower() else 'primaria'}


# ----------------------------------------------------------------------------- Gemini: cliente, reintentos y caché

def _client():
    from google import genai
    return genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'timeout': API_TIMEOUT_MS})

def with_retry(call, attempts=4, wait=8):
    """Reintenta ante saturación (503), cuota por minuto (429) o cortes de conexión, con espera exponencial."""
    from google.genai import errors
    for n in range(attempts):
        try:
            return call()
        except Exception as ex:
            code = ex.code if isinstance(ex, errors.APIError) else 'conexión'
            if code not in (429, 503, 'conexión') or n == attempts - 1: raise
            print(f'Gemini {code}: reintento {n + 1}/{attempts - 1} en {wait}s...', flush=True)
            time.sleep(wait); wait *= 2

def _cache_load(name):
    path = CACHE / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def _cache_save(name, data):
    CACHE.mkdir(exist_ok=True)
    (CACHE / name).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')

def _key(*parts):
    return hashlib.sha1('\x1f'.join(parts).encode('utf-8')).hexdigest()

def gemini_embed(texts, task, batch=50):
    """Embeddings de Gemini con caché en disco: cada texto se envía a la API una sola vez, en lotes."""
    from google.genai import types
    model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
    cache = _cache_load('embeddings.json')
    pending = list(dict.fromkeys(t for t in texts if _key(model, task, t) not in cache))
    client = _client() if pending else None
    for i in range(0, len(pending), batch):
        group = pending[i:i + batch]
        config = types.EmbedContentConfig(task_type=task)
        result = with_retry(lambda: client.models.embed_content(model=model, contents=group, config=config))
        vectors = [list(e.values) for e in result.embeddings]
        if len(vectors) != len(group):   # el servicio no ha aceptado el lote: se pide de uno en uno
            vectors = [list(with_retry(lambda t=t: client.models.embed_content(model=model, contents=t, config=config)).embeddings[0].values) for t in group]
        for text, vector in zip(group, vectors): cache[_key(model, task, text)] = vector
        _cache_save('embeddings.json', cache)
        print(f'Gemini: {len(group)} embeddings nuevos ({task}) en 1 llamada', flush=True)
    return normalize([cache[_key(model, task, t)] for t in texts]).tolist()

def make_vectors(texts, fit=False):
    if backend() == 'gemini':
        return gemini_embed(texts, 'RETRIEVAL_DOCUMENT' if fit else 'RETRIEVAL_QUERY')
    from joblib import dump, load
    path = BASE / '.tfidf.joblib'
    if fit:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode', lowercase=True, sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
        dump(vectorizer, path)
    else:
        if not path.exists(): raise RuntimeError('Ejecuta primero index')
        matrix = load(path).transform(texts)
    return matrix.toarray().tolist()

def warm_up(questions):
    """Calcula de una vez (una llamada en lote) los embeddings de todas las preguntas de un experimento."""
    if backend() == 'gemini': gemini_embed(list(questions), 'RETRIEVAL_QUERY')


# ----------------------------------------------------------------------------- indexación

def index(size=650, overlap=100, title=True, quiet=False):
    docs = load_docs()
    items = []
    for name, text in docs:
        meta = parse_header(text)
        for i, chunk in enumerate(chunks(text, size, overlap)):
            # Los fragmentos intermedios conservan el título del documento al que pertenecen
            if title and i > 0 and meta['titulo']: chunk = f"[{meta['titulo']}] {chunk}"
            items.append((f'{name}#chunk{i + 1}', {'source': name, 'embedding_backend': backend(), **meta}, chunk))
    if not items: raise RuntimeError('No hay documentos en data/')
    client = chromadb.PersistentClient(path=str(DB))
    # Se borra la colección con Chroma (borrar la carpeta falla en Windows: los ficheros quedan bloqueados)
    if COLLECTION in [c.name for c in client.list_collections()]: client.delete_collection(COLLECTION)
    col = client.get_or_create_collection(COLLECTION, metadata={'hnsw:space': 'cosine'})
    vectors = make_vectors([it[2] for it in items], fit=True)
    col.add(ids=[it[0] for it in items], documents=[it[2] for it in items], metadatas=[it[1] for it in items], embeddings=vectors)
    if not quiet:
        print(f'Indexados {len(items)} fragmentos de {len(docs)} documentos. Backend: {backend()}. Título en fragmentos: {title}')
    return len(items)


# ----------------------------------------------------------------------------- recuperación

PISTAS_HISTORICAS = (r'\b(antigu[oa]s?|anterior(es)?|borrador(es)?|obsolet[oa]s?|derogad[oa]s?|sustitu\w+|'
                     r'historic[oa]s?|historial|cambio|cambiaron|ha cambiado|han cambiado|version(es)? previas?)\b')

def _years(text):
    return set(re.findall(r'\b(?:19|20)\d{2}\b', text or ''))

def is_historical(question, metadatas):
    """Detecta si la pregunta se refiere a documentación no vigente.

    Dos señales: (1) vocabulario histórico ("antiguo", "anterior", "borrador", "sustituyó"...), conservando la tilde
    de "cambió" para no confundirlo con "cambio"; (2) un año que solo aparece en versiones de documentos obsoletos.
    El año se obtiene de los metadatos del índice, no está escrito en el código.
    """
    plain = unicodedata.normalize('NFD', question.lower())
    plain = re.sub(r'ó\b', 'O', plain)                                   # "cambió" -> "cambiO" (marca la tilde final)
    plain = ''.join(c for c in plain if unicodedata.category(c) != 'Mn')
    if re.search(PISTAS_HISTORICAS.replace('cambio|', 'cambiO|'), plain): return True
    old = set().union(*[_years(m.get('version')) for m in metadatas if m.get('estado') == 'obsoleto'] or [set()])
    current = set().union(*[_years(m.get('version')) | _years(m.get('vigente_desde')) for m in metadatas if m.get('estado') != 'obsoleto'] or [set()])
    return bool(_years(question) & (old - current))

def retrieve(question, k=3, modo='auto', secondary_penalty=0.1, umbral=None, margen=None):
    """Devuelve hasta k fragmentos ordenados por distancia.

    modo: 'vigente' (solo documentos vigentes), 'historico' (todos) o 'auto' (decide según la pregunta).
    secondary_penalty: distancia que se suma a las fuentes secundarias (FAQ) para que prevalezca la política original.
    umbral: si el mejor resultado supera esta distancia, no se devuelve nada (pregunta sin respuesta en el corpus).
    margen: solo se conservan los fragmentos a menos de `margen` del mejor (k dinámico).
    """
    client = chromadb.PersistentClient(path=str(DB))
    col = client.get_collection(COLLECTION)
    if col.count() == 0: raise RuntimeError('Índice vacío. Ejecuta index')
    metadatas = col.get(include=['metadatas'])['metadatas']
    indexed_with = metadatas[0].get('embedding_backend')
    if indexed_with != backend():
        raise RuntimeError(f'El índice se creó con "{indexed_with}" y ahora EMBEDDING_BACKEND="{backend()}". Ejecuta index')
    if modo == 'auto': modo = 'historico' if is_historical(question, metadatas) else 'vigente'
    where = {'estado': 'vigente'} if modo == 'vigente' else None
    qvec = make_vectors([question], fit=False)[0]
    # Se piden candidatos de más para poder reordenar sin perder resultados
    out = col.query(query_embeddings=[qvec], n_results=min(k * 3, col.count()), where=where, include=['documents', 'metadatas', 'distances'])
    found = [(float(dist) + (secondary_penalty if m.get('tipo') == 'secundaria' else 0), doc, m)
             for doc, m, dist in zip(out['documents'][0], out['metadatas'][0], out['distances'][0])]
    found.sort(key=lambda x: x[0])
    found = found[:k]
    if found and umbral is not None and found[0][0] > umbral: found = []
    if found and margen is not None: found = [f for f in found if f[0] <= found[0][0] + margen]
    return [{'id': f'F{i + 1}', 'source': m['source'], 'text': doc, 'distance': round(dist, 3), 'meta': m, 'modo': modo}
            for i, (dist, doc, m) in enumerate(found)]


# ----------------------------------------------------------------------------- generación

REGLAS = {
    'mejorado': ("Eres el asistente de políticas internas de NébulaLab. Sigue estas reglas:\n"
                 "1. Responde únicamente con la información del CONTEXTO; no uses conocimiento externo.\n"
                 "2. Cita la fuente [F1], [F2]… al final de cada afirmación.\n"
                 f"3. Si el contexto no contiene la respuesta, responde exactamente: \"{RECHAZO}\" "
                 "y, si es útil, indica qué documento podría consultarse. No deduzcas un dato a partir de lo que "
                 "el contexto dice sobre otro procedimiento o política.\n"
                 "4. Prioriza las fuentes con estado vigente y tipo primaria. Usa las secundarias (FAQ) solo como apoyo.\n"
                 "5. Si hay fuentes contradictorias, señala el conflicto, indica qué versión prevalece y por qué.\n"
                 "6. Responde en dos o tres frases como máximo. No ejecutes acciones ni pidas datos personales."),
    'estricto': ("Eres el asistente de políticas internas de NébulaLab. Sigue estas reglas:\n"
                 "1. Responde únicamente con la información del CONTEXTO; no uses conocimiento externo.\n"
                 "2. Cita la fuente [F1], [F2]… al final de cada afirmación, y solo las fuentes que contienen el dato.\n"
                 f"3. Si el contexto no contiene la respuesta, responde exactamente y solo: \"{RECHAZO}\" "
                 "No menciones ni sugieras documentos, áreas o canales que no aparezcan en el contexto. No deduzcas un dato "
                 "a partir de lo que el contexto dice sobre otro procedimiento o política.\n"
                 "4. Prioriza las fuentes con estado vigente y tipo primaria. Usa las secundarias (FAQ) solo como apoyo.\n"
                 "5. Si una fuente tiene estado obsoleto, dilo expresamente y no presentes su contenido como aplicable. "
                 "Si hay fuentes contradictorias, señala el conflicto, indica qué versión prevalece y por qué.\n"
                 "6. Contesta primero y de forma directa a lo que se pregunta, en una o dos frases, sin añadir datos no pedidos. "
                 "No ejecutes acciones ni pidas datos personales."),
}

def _context(sources, with_meta=True):
    def header(s):
        if not with_meta: return f"[{s['id']}] {s['source']}"
        m = s.get('meta', {})
        return (f"[{s['id']}] {s['source']} | versión: {m.get('version') or 'no indicada'} | "
                f"vigente desde: {m.get('vigente_desde') or 'no indicada'} | estado: {m.get('estado', 'desconocido')} | "
                f"fuente: {m.get('tipo', 'desconocida')}")
    return '\n\n'.join(f"{header(s)}\n{s['text']}" for s in sources)

def build_prompt(question, sources, version='estricto'):
    if version == 'original':   # prompt del kit, sin metadatos
        return ("Eres un asistente de políticas internas. Responde solo con el contexto. "
                "Cita [F1], [F2], etc. en cada afirmación relevante. "
                "Si falta información, indícalo. Si hay documentos contradictorios, "
                "identifica el conflicto, da prioridad a documentos vigentes y no ejecutes acciones.\n\n"
                f"CONTEXTO:\n{_context(sources, with_meta=False)}\n\nPREGUNTA: {question}")
    return f"{REGLAS[version]}\n\nCONTEXTO:\n{_context(sources)}\n\nPREGUNTA: {question}"

def build_batch_prompt(items, version='estricto'):
    """Un único prompt con varias preguntas independientes, cada una con su propio contexto."""
    blocks = [f"### PREGUNTA {n}\nCONTEXTO DE LA PREGUNTA {n}:\n{_context(sources)}\n\nPREGUNTA {n}: {question}"
              for n, question, sources in items]
    return (f"{REGLAS[version]}\n\n"
            "Vas a recibir varias preguntas INDEPENDIENTES. Cada una tiene su propio CONTEXTO: responde cada pregunta "
            "usando exclusivamente el contexto de esa pregunta, como si no hubieras visto las demás, y aplica las reglas "
            "anteriores a cada una por separado. Las referencias [F1], [F2]… son las del contexto de esa pregunta.\n"
            'Devuelve solo un JSON con esta forma: [{"n": <número de pregunta>, "respuesta": "<texto>"}, ...]\n\n'
            + '\n\n'.join(blocks))

def _models():
    """GEMINI_MODEL admite varios modelos separados por comas: si uno no está disponible se prueba el siguiente."""
    return [m.strip() for m in os.getenv('GEMINI_MODEL', 'gemini-3.5-flash').split(',') if m.strip()]

def _generate(prompt, as_json=False):
    """Llama a Gemini (temperatura baja) con caché en disco: un mismo prompt no se envía dos veces."""
    from google.genai import errors, types
    cache = _cache_load('respuestas.json')
    key = _key(prompt)
    # GEMINI_EXIGIR_MODELO: repite una respuesta guardada si la dio otro modelo (la anterior se conserva si la llamada falla)
    required = os.getenv('GEMINI_EXIGIR_MODELO', '').strip()
    if key not in cache or (required and cache[key]['modelo'] != required):
        config = types.GenerateContentConfig(temperature=0.1, response_mime_type='application/json' if as_json else None)
        client = _client()
        models = [required] if required else _models()
        for n, model in enumerate(models):
            try:
                text = str(with_retry(lambda: client.models.generate_content(model=model, contents=prompt, config=config)).text)
                break
            except errors.APIError as ex:   # modelo retirado (404), saturado (503) o sin cuota (429): se prueba el siguiente
                if key in cache:
                    print(f'Gemini {ex.code} con {model}: se mantiene la respuesta guardada de {cache[key]["modelo"]}', flush=True)
                    return cache[key]['texto']
                if ex.code not in (404, 429, 503) or n == len(models) - 1: raise
                print(f'Gemini {ex.code} con {model}: se prueba con {models[n + 1]}', flush=True)
        cache[key] = {'modelo': model, 'texto': text}
        _cache_save('respuestas.json', cache)
        print(f'Gemini: 1 llamada de generación ({model})', flush=True)
    return cache[key]['texto']

def list_models():
    """Modelos disponibles para la clave configurada (útil si el modelo de .env ya no existe)."""
    for m in _client().models.list():
        actions = getattr(m, 'supported_actions', None) or []
        if 'generateContent' in actions or 'embedContent' in actions: print(f"{m.name:50} {', '.join(actions)}")

def gemini_answer(question, sources, version='estricto'):
    if not sources: return RECHAZO    # el umbral de rechazo ya ha descartado la pregunta: no se gasta una llamada
    return _generate(build_prompt(question, sources, version)).strip()

def gemini_answer_batch(items, version='estricto'):
    """items: [(n, pregunta, fuentes)]. Devuelve {n: respuesta} con una sola llamada para todo el lote."""
    answers = {n: RECHAZO for n, _, sources in items if not sources}
    pending = [it for it in items if it[2]]
    if pending:
        raw = _generate(build_batch_prompt(pending, version), as_json=True)
        data = json.loads(raw[raw.find('['):raw.rfind(']') + 1])
        answers.update({int(d['n']): str(d['respuesta']).strip() for d in data})
    return answers

def batch_model(items, version='estricto'):
    """Modelo que respondió un lote ya generado (queda guardado en la caché)."""
    pending = [it for it in items if it[2]]
    entry = _cache_load('respuestas.json').get(_key(build_batch_prompt(pending, version))) if pending else None
    return entry['modelo'] if entry else None

def answer(question, gemini=False, k=3, prompt_version='estricto', **opts):
    sources = retrieve(question, k, **opts)
    if not sources:
        print(f'\nNingún fragmento supera el umbral de relevancia.\n\nRESPUESTA:\n{RECHAZO}')
        return sources
    print(f"\nFUENTES RECUPERADAS (modo {sources[0]['modo']}):")
    for s in sources:
        mark = ' ⚠ OBSOLETO' if s['meta'].get('estado') == 'obsoleto' else ''
        print(f"[{s['id']}] {s['source']}{mark} (distancia {s['distance']})\n  {s['text'][:230]}...")
    if gemini:
        print('\nRESPUESTA GEMINI:\n' + gemini_answer(question, sources, prompt_version))
    else:
        print('\nMODO LOCAL (extractivo, no LLM): lee los fragmentos citados y formula tu respuesta; comprueba la vigencia documental.')
    return sources


# ----------------------------------------------------------------------------- evaluación de la recuperación

def load_questions(path):
    """Lee el CSV de pruebas. `fuentes_relevantes` (separadas por ;) es opcional: si falta se usa `fuente_esperada`."""
    with open(path, encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    for n, row in enumerate(rows, 1):
        row['n'] = n
        relevant = row.get('fuentes_relevantes') or row.get('fuente_esperada') or ''
        row['relevantes'] = [s.strip() for s in relevant.split(';') if s.strip()]
        row['tipo'] = row.get('tipo') or ('trampa' if not row['relevantes'] else 'factual')
    return rows

def retrieval_metrics(rows, k=3, **opts):
    """Métricas por documento sobre las preguntas con respuesta, y tasa de rechazo sobre las preguntas trampa.

    Hit@k y MRR@k usan la fuente principal (`fuente_esperada`). Precision, Recall y F1 usan todas las fuentes
    relevantes: P = relevantes recuperadas / recuperadas, R = relevantes recuperadas / relevantes.
    """
    warm_up(r['pregunta'] for r in rows)
    answerable = [r for r in rows if r['relevantes']]
    traps = [r for r in rows if not r['relevantes']]
    totals = dict(hit=0, mrr=0.0, p=0.0, r=0.0, f1=0.0, ctx=0, sin_resultado=0)
    detail = []
    for row in answerable:
        got = retrieve(row['pregunta'], k, **opts)
        sources = list(dict.fromkeys(s['source'] for s in got))
        main = row.get('fuente_esperada') or row['relevantes'][0]
        rank = next((i + 1 for i, s in enumerate(sources) if main in s), 0)
        good = sum(any(rel in s for rel in row['relevantes']) for s in sources)
        p = good / len(sources) if sources else 0.0
        r = good / len(row['relevantes'])
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        totals['hit'] += bool(rank); totals['mrr'] += 1 / rank if rank else 0
        totals['p'] += p; totals['r'] += r; totals['f1'] += f1
        totals['ctx'] += len(got); totals['sin_resultado'] += not got
        detail.append({'n': row['n'], 'pregunta': row['pregunta'], 'tipo': row['tipo'], 'rank': rank, 'p': p, 'r': r,
                       'fuentes': [s[:2] for s in sources], 'dist': got[0]['distance'] if got else None})
    n = len(answerable) or 1
    rejected = sum(not retrieve(t['pregunta'], k, **opts) for t in traps)
    return {'n': len(answerable), 'hit': totals['hit'] / n, 'mrr': totals['mrr'] / n, 'precision': totals['p'] / n,
            'recall': totals['r'] / n, 'f1': totals['f1'] / n, 'contexto': totals['ctx'] / n,
            'falsos_rechazos': totals['sin_resultado'], 'trampas': len(traps), 'trampas_rechazadas': rejected, 'detalle': detail}

def evaluate(path, k=3, **opts):
    rows = load_questions(path)
    m = retrieval_metrics(rows, k, **opts)
    for d in m['detalle']:
        print(f"{'OK   ' if d['rank'] else 'FALLO'} | rank={d['rank']} | P={d['p']:.2f} R={d['r']:.2f} | {d['fuentes']} | {d['pregunta']}")
    print(f"\n{m['n']} preguntas con respuesta | Hit@{k}: {m['hit']:.1%} | MRR@{k}: {m['mrr']:.3f} | "
          f"Precision@{k}: {m['precision']:.3f} | Recall@{k}: {m['recall']:.3f} | F1@{k}: {m['f1']:.3f} | "
          f"fragmentos de contexto (media): {m['contexto']:.2f}")
    print(f"{m['trampas']} preguntas sin respuesta en el corpus | rechazadas en la recuperación: {m['trampas_rechazadas']} | "
          f"preguntas con respuesta rechazadas por error: {m['falsos_rechazos']}")
    return m


# ----------------------------------------------------------------------------- línea de comandos

def _retrieval_args(sp):
    d = defaults()
    sp.add_argument('-k', type=int, default=3)
    sp.add_argument('--modo', choices=['auto', 'vigente', 'historico'], default=d['modo'],
                    help='auto: las preguntas históricas consultan también los documentos obsoletos')
    sp.add_argument('--penal-secundarias', type=float, default=d['secondary_penalty'], help='penalización de distancia para la FAQ')
    sp.add_argument('--umbral', type=float, default=d['umbral'], help='distancia máxima del mejor resultado; si se supera, se rechaza la pregunta')
    sp.add_argument('--margen', type=float, default=d['margen'], help='solo se conservan los fragmentos a esta distancia del mejor')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Asistente RAG de NébulaLab')
    sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('index')
    i.add_argument('--size', type=int, default=650); i.add_argument('--overlap', type=int, default=100)
    i.add_argument('--sin-titulo', action='store_true', help='no añadir el título del documento a los fragmentos')
    a = sub.add_parser('ask')
    a.add_argument('question'); a.add_argument('--gemini', action='store_true')
    a.add_argument('--prompt', choices=['estricto', 'mejorado', 'original'], default='estricto')
    e = sub.add_parser('evaluate'); e.add_argument('--file', required=True)
    sub.add_parser('modelos', help='lista los modelos de Gemini disponibles para la clave')
    for sp in (a, e): _retrieval_args(sp)
    args = p.parse_args()
    try:
        if args.cmd == 'index':
            index(args.size, args.overlap, not args.sin_titulo)
        elif args.cmd == 'modelos':
            list_models()
        else:
            opts = {'modo': args.modo, 'secondary_penalty': args.penal_secundarias, 'umbral': args.umbral, 'margen': args.margen}
            if args.cmd == 'ask': answer(args.question, args.gemini, args.k, args.prompt, **opts)
            else: evaluate(args.file, args.k, **opts)
    except (KeyError, FileNotFoundError, RuntimeError) as ex:
        p.error(f'Configuración o fichero ausente: {ex}')
