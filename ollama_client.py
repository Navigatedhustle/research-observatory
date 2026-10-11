from __future__ import annotations
import json, urllib.request, os
from typing import List

BASE = os.environ.get('ZGX_OLLAMA_URL','http://127.0.0.1:11434').rstrip('/') + '/api'

def _get(path: str, timeout: int = 3):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def _post(path: str, payload: dict, timeout: int = 240):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))

def is_available() -> bool:
    try:
        _get('/tags', 2)
        return True
    except Exception:
        return False

def running_models():
    try:
        return _get('/ps', 3).get('models', [])
    except Exception:
        return []

def model_loaded(prefix='llama3.1:70b') -> bool:
    for m in running_models():
        name = m.get('name') or m.get('model') or ''
        if name.startswith(prefix) or prefix.startswith(name):
            return True
    return False

def generate(prompt: str, system: str = '', model: str = 'llama3.1:70b', timeout: int = 300, keep_alive: str = '30m') -> str:
    obj = _post('/generate', {
        'model': model,
        'prompt': prompt,
        'system': system,
        'stream': False,
        'keep_alive': keep_alive,
        'options': {'temperature': 0.2, 'num_predict': 450},
    }, timeout=timeout)
    return obj.get('response','').strip()

def embed(texts: List[str], model: str = 'nomic-embed-text:latest', timeout: int = 90):
    obj = _post('/embed', {'model':model, 'input':texts, 'truncate':True}, timeout=timeout)
    return obj.get('embeddings', [])
