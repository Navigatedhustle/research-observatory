# Installation and operation

## Environment

Recorded development hardware: NVIDIA GB10 ARM64, nominal 128 GB unified memory (about 121 GiB usable), Ubuntu 24.04.4 LTS, NVIDIA driver 580.173.02 and CUDA driver support 13.0. These are recorded facts, not universal minimum requirements. Tests used existing project environments on that machine; fresh dependency installation on another machine was not performed. Model-heavy modes need enough free unified memory and storage for the assets. Start heavyweight applications sequentially and check available resources first.

Python target: 3.13. Dependencies: source manifests and `docs/DEPENDENCIES.md`. Exact installed versions are recorded where available; an installed-environment freeze is not proof every optional package is required. Use project-scoped environments and do not replace the host NVIDIA driver/toolkit.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Configuration, startup and shutdown

Set `ZGX_MOCK=1` for a control-plane-only review; model-backed or video/camera operation requires separate assets. `ZGX_OLLAMA_URL` defaults to loopback 11434; `ZGX_ENGINE` defaults to a local ignored model path. Use your own synthetic/authorized footage and keep recordings/event databases private. Mock mode may still invoke configured local reasoning on explicit analyze/start actions; API presence alone is not live inference proof.

For foreground source review, run `bash scripts/run-local.sh`. It binds only `127.0.0.1:18120`, refuses an occupied port and stops with Ctrl-C. Set `PROJECT_PORT` explicitly if needed. Set `PROJECT_PYTHON` only to the intended project interpreter. It does not install services or launch model servers. Original all-stack startup scripts are retained as historical references and may control named runtime services; inspect them before use.

## Troubleshooting

If imports fail, activate the correct project environment and inspect dependency manifests. If a port is occupied, stop your own review process or choose another review port; do not stop an unrelated runtime. If model readiness fails, verify local assets, model descriptors, license acceptance and loopback endpoints. Missing excluded assets are an honest setup error; no cloud inference fallback is configured by the publication launcher. Empty directories and generated state are recreated during setup/startup.

For frontend missing-directory errors, build frontend assets or create the required empty output directory; do not commit generated bundles. For optional model failures, distinguish control-plane/unit-test success from verified live inference. Logs, exports, databases, screenshots and cache indexes belong in private runtime storage and remain ignored. Restore the original project's service state only if you intentionally changed it; portfolio preparation never did.
