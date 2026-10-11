# Research Observatory

A local multi-feed vision demonstration that detects objects, summarizes events with a local language model and stores searchable event memory.

**Maturity:** V4 is the active source target. V0.1/V0.2/V0.3 and duplicate downloads are preserved locally as historical versions. Source import/HTTP smoke checks use mock mode and excluded-media empty directories. Actual camera, licensed video feeds and TensorRT detection have not been verified in this preparation.

## How it works

FastAPI browser dashboard → OpenCV feed workers → TensorRT YOLO detector → bounded events → Ollama Llama 3.1 reasoning → SQLite event memory with local embeddings. Mock mode is explicitly configured and does not establish real detector accuracy.

## Getting started

Read [installation, configuration and shutdown](docs/OPERATIONS.md) and [model/data retrieval](docs/MODELS.md). Review [dependency evidence](docs/DEPENDENCIES.md) before choosing an environment. The safe review launcher uses loopback and does not install services or load models.

## Security and data

Read [security and safe updates](SECURITY.md) and [ownership/license status](docs/LICENSE-STATUS.md). Runtime originals are preserved separately. This is a sanitized source snapshot with new history: no account state, business records, model weights, unknown media assets or original logs/history are distributed. Third-party notices included in this copy remain applicable.

## Verification

See [verification evidence and limitations](docs/VERIFICATION.md). Tests distinguish synthetic/mocked workflows from live integrations. Original demo screenshots and document recordings are excluded; use the synthetic workflow in the operations guide for your own reviewed demonstration.

## Roadmap

Pin a reproducible detector-engine build and media provenance; improve feed/model readiness errors and bounded event retention; automate real GPU and camera validation with authorized assets.

For route/source details see [architecture](docs/ARCHITECTURE.md).
