from __future__ import annotations
from pathlib import Path
import sqlite3, json, math, time
from typing import List, Dict
from ollama_client import embed, is_available

DB = Path(__file__).resolve().parent / "data" / "memory.sqlite3"

def init_db():
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as cx:
        cx.execute("""CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            embedding TEXT
        )""")

def remember(source: str, text: str):
    init_db()
    vec = None
    if is_available():
        try:
            embs = embed([text])
            if embs:
                vec = json.dumps(embs[0])
        except Exception:
            vec = None
    with sqlite3.connect(DB) as cx:
        cx.execute("INSERT INTO memories(ts,source,text,embedding) VALUES(?,?,?,?)", (time.time(), source, text, vec))

def _cos(a, b):
    if not a or not b or len(a) != len(b):
        return -1
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-9)

def search(query: str, k: int = 5) -> List[Dict]:
    init_db()
    if not is_available():
        return []
    q = embed([query])
    if not q:
        return []
    qv = q[0]
    rows = []
    with sqlite3.connect(DB) as cx:
        for rid, ts, source, text, emb in cx.execute(
            "SELECT id,ts,source,text,embedding FROM memories WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 500"
        ):
            try:
                score = _cos(qv, json.loads(emb))
            except Exception:
                continue
            rows.append({"id": rid, "ts": ts, "source": source, "text": text, "score": score})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:k]
