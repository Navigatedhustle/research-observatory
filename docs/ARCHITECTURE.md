# Architecture

FastAPI browser dashboard → OpenCV feed workers → TensorRT YOLO detector → bounded events → Ollama Llama 3.1 reasoning → SQLite event memory with local embeddings. Mock mode is explicitly configured and does not establish real detector accuracy.

## Implemented endpoints inspected from source

| Method | Route | Source |
| --- | --- | --- |
| GET | `/` | `app.py` |
| GET | `/api/status` | `app.py` |
| GET | `/api/dashboard` | `app.py` |
| POST | `/api/start/{mode}` | `app.py` |
| POST | `/api/stop` | `app.py` |
| POST | `/api/reset-workflow` | `app.py` |
| GET | `/stream/{feed_id}` | `app.py` |
| POST | `/api/analyze` | `app.py` |
| POST | `/api/prewarm` | `app.py` |
| POST | `/api/auto-reason` | `app.py` |
| POST | `/api/memory/search` | `app.py` |

These routes establish implementation shape. They do not establish successful provider integration or model quality. Generated state and data stores are excluded from the repository.
