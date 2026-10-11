from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from yolo_trt import TRTDetector
from ollama_client import is_available as ollama_available, generate, model_loaded
from event_memory import remember, search as memory_search, init_db

APP_DIR = Path(__file__).resolve().parent
HOME = Path.home()
ENGINE = Path(os.environ.get("ZGX_ENGINE", str(APP_DIR / "models/yolov8s.trt")))
MANIFEST = APP_DIR / "media" / "manifest.json"
VERSION = "0.4"
MOCK = os.getenv("ZGX_MOCK", "0") == "1"

app = FastAPI(title="HP ZGX Research Observatory V4")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/media", StaticFiles(directory=APP_DIR / "media"), name="media")
app.mount("/printables", StaticFiles(directory=APP_DIR / "printables"), name="printables")
init_db()

EVENTS = deque(maxlen=600)
FEEDS: Dict[int, "FeedWorker"] = {}
STATE_LOCK = threading.RLock()
SESSION_LOCK = threading.RLock()

SESSION = {
    "started": time.time(),
    "stream_seconds_completed": 0.0,
    "frames_processed": 0,
    "vision_inferences": 0,
    "reasoning_calls": 0,
    "offline_proven": False,
}

EMPTY_CARDS = {
    "headline": "Ready for a local insight",
    "observation": "Start a demo and let the event engine build context.",
    "impact": "The reasoning layer will prioritize a meaningful change instead of describing every frame.",
    "local_advantage": "Vision, event processing, reasoning and semantic memory are designed to run on the Nano without a cloud AI API.",
    "next_action": "Start a feed, then generate an insight after a meaningful event.",
    "value_levers": ["Data locality", "Cloud usage avoidance", "Offline resilience"],
}

REASONING = {
    "status": "idle",
    "output": "",
    "cards": dict(EMPTY_CARDS),
    "trigger": "",
    "pending_trigger": "",
    "started": 0.0,
    "finished": 0.0,
    "error": "",
}
AUTO_REASON = {"enabled": False, "last_trigger": 0.0, "cooldown": 90.0}

SYSTEM_PROMPT = """You are the local intelligence layer of HP ZGX Research Observatory.
You receive structured outputs from a TensorRT object detector, motion/deviation engine,
QR workflow tracker, session metrics, network state and local event timeline.

Be rigorous. Never invent identities, PPE status, safety violations, scientific measurements,
equipment classes, financial savings, cloud prices, or events not present in the structured data.
Focus on decision-useful insight rather than describing every detection.

Return VALID JSON ONLY with this exact schema:
{
  "headline": "8 words or fewer",
  "observation": "What materially changed, 1-2 concise sentences",
  "impact": "Why that change matters operationally, 1-2 concise sentences",
  "local_advantage": "What this session specifically demonstrates about local AI. Only cite benefits supported by the supplied state.",
  "next_action": "One concrete next observation/action",
  "value_levers": ["up to 3 items chosen from Data locality, Cloud usage avoidance, Faster response, Staff attention, Offline resilience, Predictable infrastructure"]
}

Do not claim a dollar ROI. The UI has a separate user-input ROI model.
If network is offline AND workloads are still active, explicitly call out offline resilience.
If external_ai_calls is 0, you may state that this application made zero external AI API calls.
If media_uploads is 0, you may state that this application uploaded zero video frames to a cloud AI service.
Keep each field concise and appropriate for professors, students, research leaders and technical event audiences."""


def clock() -> str:
    return time.strftime("%H:%M:%S")


def source_manifest():
    items = []
    if MANIFEST.exists():
        try:
            for m in json.loads(MANIFEST.read_text()):
                p = APP_DIR / "media" / m["file"]
                if p.exists():
                    x = dict(m)
                    x.update({"kind": "video", "path": str(p), "media_url": f"/media/{m['file']}"})
                    items.append(x)
        except Exception:
            pass
    if Path("/dev/video0").exists():
        items.insert(
            0,
            {
                "name": "Live Research Bench",
                "discipline": "INTERACTIVE WORKFLOW",
                "kind": "camera",
                "path": "0",
                "slot": 0,
                "detection_mode": "people_objects",
                "conf_threshold": 0.28,
                "fit": "cover",
                "enhance_inference": False,
                "media_url": None,
            },
        )
    return items


def session_value():
    with SESSION_LOCK:
        stream_seconds = float(SESSION["stream_seconds_completed"])
        frames = int(SESSION["frames_processed"])
        vision = int(SESSION["vision_inferences"])
        reasoning = int(SESSION["reasoning_calls"])
        offline_proven = bool(SESSION["offline_proven"])
    for f in list(FEEDS.values()):
        stream_seconds += f.live_elapsed()
    significant = sum(1 for e in EVENTS if e.get("significant"))
    return {
        "stream_minutes_local": round(stream_seconds / 60.0, 1),
        "frames_processed_local": frames,
        "vision_inferences_local": vision,
        "reasoning_calls_local": reasoning,
        "events": len(EVENTS),
        "significant_events": significant,
        "external_ai_calls": 0,
        "media_uploads": 0,
        "offline_proven": offline_proven,
        "active_streams": sum(1 for f in FEEDS.values() if f.running),
    }


class VisionService:
    """One dedicated CUDA/TensorRT worker.

    The video/UI threads never wait on inference. If the queue is busy, old work is
    dropped rather than building latency. This is deliberate for live event use.
    """

    def __init__(self, engine_path: Path):
        self.engine_path = engine_path
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.pending = set()
        self.results = {}
        self.lock = threading.RLock()
        self.ready = False
        self.error = ""
        self.info_data = {"ready": False, "error": "initializing"}
        self.thread = threading.Thread(target=self._run, daemon=True, name="zgx-tensorrt")
        self.thread.start()

    def _run(self):
        if MOCK:
            self.error = "mock mode"
            self.info_data = {"ready": False, "error": "mock mode"}
            return
        if not self.engine_path.exists():
            self.error = f"{self.engine_path} not found"
            self.info_data = {"ready": False, "error": self.error}
            return
        try:
            detector = TRTDetector(str(self.engine_path))
            self.info_data = detector.info()
            self.ready = bool(detector.ready)
            self.error = detector.error if not detector.ready else ""
            if not self.ready:
                return
            while True:
                feed_id, frame, mode, conf = self.q.get()
                try:
                    dets = detector.infer(frame, conf_thres=float(conf))
                    if mode == "people_only":
                        dets = [d for d in dets if d.get("label") == "person"]
                    elif mode == "people_objects":
                        allowed = {
                            "person", "bottle", "cup", "book", "laptop", "keyboard",
                            "mouse", "cell phone", "chair", "backpack", "scissors",
                        }
                        dets = [d for d in dets if d.get("label") in allowed]
                    with self.lock:
                        self.results[feed_id] = {"ts": time.time(), "detections": dets}
                    with SESSION_LOCK:
                        SESSION["vision_inferences"] += 1
                except Exception as e:
                    with self.lock:
                        self.results[feed_id] = {"ts": time.time(), "detections": [], "error": str(e)}
                finally:
                    with self.lock:
                        self.pending.discard(feed_id)
                    self.q.task_done()
        except Exception as e:
            self.ready = False
            self.error = f"{type(e).__name__}: {e}"
            self.info_data = {"ready": False, "error": self.error}

    def submit(self, feed_id: int, frame, mode: str, conf: float):
        if not self.ready:
            return False
        with self.lock:
            if feed_id in self.pending:
                return False
            self.pending.add(feed_id)
        try:
            self.q.put_nowait((feed_id, frame.copy(), mode, float(conf)))
            return True
        except queue.Full:
            with self.lock:
                self.pending.discard(feed_id)
            return False

    def get(self, feed_id: int):
        with self.lock:
            return dict(self.results.get(feed_id, {"ts": 0.0, "detections": []}))

    def info(self):
        x = dict(self.info_data)
        x["queue_depth"] = self.q.qsize()
        x["ready"] = self.ready
        if self.error:
            x["error"] = self.error
        return x


VISION = VisionService(ENGINE)


def add_event(feed_id: int, feed_name: str, kind: str, text: str, severity="info", significant=False):
    ev = {
        "ts": time.time(), "clock": clock(), "feed_id": feed_id,
        "feed": feed_name, "kind": kind, "text": text,
        "severity": severity, "significant": significant,
    }
    EVENTS.appendleft(ev)
    if significant:
        with STATE_LOCK:
            REASONING["pending_trigger"] = f"{feed_name}: {text}"
    if significant or severity in ("warn", "critical") or kind in ("workflow", "analysis"):
        threading.Thread(target=remember, args=(feed_name, text), daemon=True).start()
    if significant:
        maybe_auto_reason(f"{feed_name}: {text}")
    return ev


def reasoning_snapshot():
    return {
        "feeds": [f.stats() for f in FEEDS.values()],
        "events": list(EVENTS)[:28],
        "workflow": [f.workflow_stats() for f in FEEDS.values() if f.mode == "lab"],
        "value_proof": session_value(),
        "network": network_state(),
    }


def _parse_cards(text: str):
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {
                "headline": "Local insight generated",
                "observation": text[:450],
                "impact": "Review the event timeline for the structured evidence behind this interpretation.",
                "local_advantage": "The reasoning request was executed against the local event state.",
                "next_action": "Continue observing for the next meaningful state change.",
                "value_levers": ["Staff attention"],
            }
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return dict(EMPTY_CARDS)
    cards = {}
    for k in ("headline", "observation", "impact", "local_advantage", "next_action"):
        cards[k] = str(obj.get(k, EMPTY_CARDS[k])).strip()
    levers = obj.get("value_levers", [])
    if not isinstance(levers, list):
        levers = [str(levers)]
    cards["value_levers"] = [str(x) for x in levers[:3]]
    return cards


def _reason_worker(trigger: str):
    with STATE_LOCK:
        REASONING.update({"status": "busy", "trigger": trigger, "started": time.time(), "error": ""})
    try:
        snap = reasoning_snapshot()
        prompt = "TRIGGER:\n" + trigger + "\n\nSTRUCTURED LIVE STATE:\n" + json.dumps(snap, indent=2)
        text = generate(prompt, system=SYSTEM_PROMPT, keep_alive="30m")
        cards = _parse_cards(text)
        with SESSION_LOCK:
            SESSION["reasoning_calls"] += 1
        with STATE_LOCK:
            REASONING.update({
                "status": "ready", "output": text, "cards": cards,
                "pending_trigger": "", "finished": time.time(), "error": "",
            })
        add_event(-1, "LOCAL 70B REASONING", "analysis", cards["headline"] + ": " + cards["observation"])
    except Exception as e:
        with STATE_LOCK:
            REASONING.update({"status": "error", "error": f"{type(e).__name__}: {e}", "finished": time.time()})


def maybe_auto_reason(trigger: str):
    if not AUTO_REASON["enabled"] or not ollama_available():
        return
    now = time.time()
    with STATE_LOCK:
        if REASONING["status"] in ("busy", "warming") or now - AUTO_REASON["last_trigger"] < AUTO_REASON["cooldown"]:
            return
        AUTO_REASON["last_trigger"] = now
    threading.Thread(target=_reason_worker, args=(trigger,), daemon=True).start()


def zone_rects(w, h):
    top = int(h * 0.30)
    bottom = int(h * 0.92)
    return {
        "PREP": (int(w * 0.04), top, int(w * 0.31), bottom),
        "ANALYSIS": (int(w * 0.365), top, int(w * 0.635), bottom),
        "COMPLETE": (int(w * 0.69), top, int(w * 0.96), bottom),
    }


def point_zone(cx, cy, zones):
    for name, (x1, y1, x2, y2) in zones.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return name
    return "OUTSIDE"


class FeedWorker:
    def __init__(self, feed_id: int, source: dict, mode: str):
        self.feed_id = feed_id
        self.source = source
        self.mode = mode
        self.name = source.get("name", f"Feed {feed_id + 1}")
        self.discipline = source.get("discipline", "RESEARCH")
        self.detection_mode = source.get("detection_mode", "people_only")
        self.conf_threshold = float(source.get("conf_threshold", 0.30))
        self.enhance_inference = bool(source.get("enhance_inference", False))
        self.fit = source.get("fit", "cover")
        self.media_url = source.get("media_url")
        self.running = False
        self.latest_jpeg = None
        self.thread = None
        self.frame_index = 0
        self.frame_width = 1280
        self.frame_height = 720
        self.people = 0
        self.objects = 0
        self.fps = 0.0
        self.counts = Counter()
        self.motion = 0.0
        self.deviation = 0.0
        self.activity = "STARTING"
        self.prev_gray = None
        self.motion_hist = deque(maxlen=90)
        self.people_hist = deque(maxlen=90)
        self.last_anomaly = 0.0
        self.last_people_event = 0.0
        self.last_counts = Counter()
        self.last_detect_submit = 0.0
        self.last_detect_ts = 0.0
        self.detections = []
        self.started_at = 0.0
        self.accumulated = False
        self.qr = cv2.QRCodeDetector()
        self.specimen = {
            "id": "SPECIMEN A", "zone": "NOT SEEN", "candidate": None,
            "candidate_since": 0.0, "zone_since": 0.0, "sequence": [],
            "durations": {}, "complete": False, "last_seen": 0.0,
        }
        self.person_in_analysis = False
        self.person_zone_since = 0.0
        self.person_dwell_alerted = False

    @property
    def target_fps(self):
        # Command videos render natively in Chromium at 15 fps. Backend analytics
        # sample at 4 fps, dramatically reducing decode/JPEG load.
        return 15.0 if self.mode == "lab" else 4.0

    @property
    def detect_interval(self):
        return 0.24 if self.mode == "lab" else 0.80

    def live_elapsed(self):
        return max(0.0, time.time() - self.started_at) if self.running and self.started_at else 0.0

    def _open(self):
        if self.source["kind"] == "camera":
            cap = cv2.VideoCapture(int(self.source["path"]), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap = cv2.VideoCapture(self.source["path"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self):
        if self.running:
            return
        self.running = True
        self.started_at = time.time()
        self.accumulated = False
        self.thread = threading.Thread(target=self.run, daemon=True, name=f"feed-{self.feed_id}")
        self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
        if not self.accumulated and self.started_at:
            with SESSION_LOCK:
                SESSION["stream_seconds_completed"] += max(0.0, time.time() - self.started_at)
            self.accumulated = True

    def _resize_for_pipeline(self, frame):
        h, w = frame.shape[:2]
        max_side = 1280 if self.mode == "lab" else 960
        side = max(h, w)
        if side > max_side:
            scale = max_side / side
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def _inference_frame(self, frame):
        if not self.enhance_inference:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _motion(self, frame):
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        return float((diff > 16).mean() * 100.0)

    def _update_anomaly(self):
        self.motion_hist.append(self.motion)
        self.people_hist.append(self.people)
        if len(self.motion_hist) < 18:
            self.deviation = 0.0
            self.activity = "LEARNING"
            return
        motion_base = np.asarray(list(self.motion_hist)[:-1], dtype=np.float32)
        med = float(np.median(motion_base))
        sd = float(np.std(motion_base)) + 0.75
        mz = max(0.0, (self.motion - med) / sd)
        people_base = np.asarray(list(self.people_hist)[:-1], dtype=np.float32)
        pmed = float(np.median(people_base))
        psd = float(np.std(people_base)) + 0.75
        pz = max(0.0, (self.people - pmed) / psd)
        self.deviation = max(mz, pz)

        now = time.time()
        if self.deviation >= 2.8 and now - self.last_anomaly > 15:
            self.last_anomaly = now
            self.activity = "ATTENTION"
            add_event(
                self.feed_id, self.name, "anomaly",
                f"Activity changed materially from this feed's local baseline. Motion {self.motion:.1f}% vs {med:.1f}%; people {self.people} vs ≈{pmed:.1f}.",
                "warn", True,
            )
        elif self.deviation >= 1.5:
            self.activity = "ELEVATED"
        elif self.people > 0 or self.motion > 1.8:
            self.activity = "ACTIVE"
        else:
            self.activity = "QUIET"

    def _qr_points(self, frame):
        found = []
        try:
            ok, infos, points, _ = self.qr.detectAndDecodeMulti(frame)
            if ok and points is not None:
                for txt, pts in zip(infos, points):
                    if txt:
                        found.append((txt, np.asarray(pts, dtype=np.float32)))
        except Exception:
            try:
                txt, pts, _ = self.qr.detectAndDecode(frame)
                if txt and pts is not None:
                    found.append((txt, np.asarray(pts, dtype=np.float32)))
            except Exception:
                pass
        return found

    def _update_workflow(self, frame, zones):
        if self.source["kind"] != "camera" or self.frame_index % 4:
            return []
        qrs = self._qr_points(frame)
        seen = []
        for txt, pts in qrs:
            if txt not in ("ZGX_SPECIMEN_A", "ZGX_SPECIMEN_B"):
                continue
            cx = int(np.mean(pts[:, 0])); cy = int(np.mean(pts[:, 1]))
            zone = point_zone(cx, cy, zones)
            seen.append((txt, pts, cx, cy, zone))
            if txt != "ZGX_SPECIMEN_A":
                continue
            s = self.specimen; now = time.time(); s["last_seen"] = now
            if zone != s["candidate"]:
                s["candidate"] = zone; s["candidate_since"] = now
            if zone != s["zone"] and now - s["candidate_since"] > 0.45:
                prev = s["zone"]
                if prev in ("PREP", "ANALYSIS", "COMPLETE") and s["zone_since"]:
                    s["durations"][prev] = s["durations"].get(prev, 0.0) + (now - s["zone_since"])
                s["zone"] = zone; s["zone_since"] = now
                if zone in ("PREP", "ANALYSIS", "COMPLETE"):
                    if not s["sequence"] or s["sequence"][-1] != zone:
                        s["sequence"].append(zone)
                    add_event(self.feed_id, self.name, "workflow", f"Specimen A entered {zone}. Previous={prev}.")
                    if zone == "COMPLETE" and not s["complete"]:
                        seq = s["sequence"]
                        try:
                            p = seq.index("PREP"); a = seq.index("ANALYSIS", p + 1); c = seq.index("COMPLETE", a + 1)
                            if p < a < c:
                                s["complete"] = True
                                add_event(self.feed_id, self.name, "workflow", "SPECIMEN A WORKFLOW COMPLETE: PREP → ANALYSIS → COMPLETE.", "warn", True)
                        except ValueError:
                            pass
        return seen

    def _update_person_zone(self, detections, zones):
        if self.source["kind"] != "camera":
            return
        x1, y1, x2, y2 = zones["ANALYSIS"]
        inside = False
        for d in detections:
            if d.get("label") != "person":
                continue
            bx1, by1, bx2, by2 = d["bbox"]
            cx = (bx1 + bx2) // 2; cy = (by1 + by2) // 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                inside = True; break
        now = time.time()
        if inside and not self.person_in_analysis:
            self.person_in_analysis = True; self.person_zone_since = now; self.person_dwell_alerted = False
            add_event(self.feed_id, self.name, "zone", "Participant entered ANALYSIS zone.")
        elif inside and self.person_in_analysis and not self.person_dwell_alerted and now - self.person_zone_since >= 3.0:
            self.person_dwell_alerted = True
            add_event(self.feed_id, self.name, "zone", f"Participant remained in ANALYSIS zone for {now-self.person_zone_since:.1f}s.", "warn", True)
        elif not inside and self.person_in_analysis:
            dwell = now - self.person_zone_since
            self.person_in_analysis = False; self.person_dwell_alerted = False
            add_event(self.feed_id, self.name, "zone", f"Participant exited ANALYSIS zone after {dwell:.1f}s.")

    def _refresh_detection_result(self):
        result = VISION.get(self.feed_id)
        ts = float(result.get("ts", 0.0))
        if ts > self.last_detect_ts:
            self.last_detect_ts = ts
            self.detections = result.get("detections", [])

    def _draw_lab(self, frame, qr_seen, zones):
        h, w = frame.shape[:2]
        for d in self.detections:
            bx1, by1, bx2, by2 = d["bbox"]
            lab = f"{d['label'].upper()} {d['conf'] * 100:.0f}%"
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 169, 224), 2)
            cv2.rectangle(frame, (bx1, max(0, by1 - 24)), (bx1 + max(110, len(lab) * 8), by1), (6, 13, 18), -1)
            cv2.putText(frame, lab, (bx1 + 5, max(16, by1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 250, 252), 1, cv2.LINE_AA)
        overlay = frame.copy()
        for zn, (zx1, zy1, zx2, zy2) in zones.items():
            col = (118, 185, 0) if zn == "COMPLETE" else (0, 169, 224)
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), col, -1)
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), col, 2)
            cv2.putText(frame, zn, (zx1 + 12, zy1 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, col, 2, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, 0.065, frame, 0.935, 0)
        for txt, pts, cx, cy, zn in qr_seen:
            poly = pts.astype(int).reshape(-1, 1, 2)
            cv2.polylines(frame, [poly], True, (118, 185, 0), 3)
            cv2.circle(frame, (cx, cy), 6, (118, 185, 0), -1)
            cv2.putText(frame, f"SPECIMEN A → {zn}", (max(8, cx - 75), max(28, cy - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (118, 235, 90), 2)
        cv2.rectangle(frame, (0, 0), (w, 46), (4, 9, 13), -1)
        cv2.putText(frame, f"{self.name.upper()}  ·  LOCAL TENSORRT  ·  {self.activity}", (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 247, 250), 1, cv2.LINE_AA)
        return frame

    def run(self):
        cap = self._open()
        if not cap.isOpened():
            add_event(self.feed_id, self.name, "error", f"Could not open source {self.source['path']}.", "critical")
            self.running = False
            return
        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        skip_count = max(0, int(round(source_fps / self.target_fps)) - 1) if self.source["kind"] == "video" else 0
        add_event(self.feed_id, self.name, "source", f"{self.discipline} feed online. Local vision + motion analytics active.")
        fpsbuf = deque(maxlen=20); next_tick = time.perf_counter(); last_frame_time = time.perf_counter()

        while self.running:
            now_perf = time.perf_counter()
            if now_perf < next_tick:
                time.sleep(min(0.01, next_tick - now_perf)); continue
            next_tick = max(next_tick + 1.0 / self.target_fps, now_perf)
            ok, frame = cap.read()
            if not ok:
                if self.source["kind"] == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); self.prev_gray = None; continue
                time.sleep(0.05); continue
            if self.source["kind"] == "video":
                for _ in range(skip_count): cap.grab()
            self.frame_index += 1
            with SESSION_LOCK: SESSION["frames_processed"] += 1
            frame = self._resize_for_pipeline(frame)
            self.frame_height, self.frame_width = frame.shape[:2]
            self.motion = self._motion(frame)
            now = time.time()
            if now - self.last_detect_submit >= self.detect_interval:
                if VISION.submit(self.feed_id, self._inference_frame(frame), self.detection_mode, self.conf_threshold):
                    self.last_detect_submit = now
            self._refresh_detection_result()
            self.counts = Counter(d.get("label") for d in self.detections)
            self.people = self.counts.get("person", 0); self.objects = len(self.detections)
            self._update_anomaly()
            if self.people != self.last_counts.get("person", 0) and now - self.last_people_event > 3.0:
                add_event(self.feed_id, self.name, "people", f"People count changed {self.last_counts.get('person', 0)} → {self.people}.")
                self.last_people_event = now
            self.last_counts = self.counts.copy()

            if self.mode == "lab":
                zones = zone_rects(self.frame_width, self.frame_height)
                qr_seen = self._update_workflow(frame, zones)
                self._update_person_zone(self.detections, zones)
                rendered = self._draw_lab(frame, qr_seen, zones)
                okjpg, jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if okjpg: self.latest_jpeg = jpg.tobytes()

            t = time.perf_counter(); dt = t - last_frame_time; last_frame_time = t
            if dt > 0:
                fpsbuf.append(1 / dt); self.fps = sum(fpsbuf) / len(fpsbuf)
        cap.release()

    def workflow_stats(self):
        s = dict(self.specimen)
        if s["zone"] in ("PREP", "ANALYSIS", "COMPLETE") and s["zone_since"]:
            d = dict(s["durations"]); d[s["zone"]] = d.get(s["zone"], 0.0) + (time.time() - s["zone_since"]); s["durations"] = d
        s["person_analysis_dwell"] = round(time.time() - self.person_zone_since, 1) if self.person_in_analysis else 0.0
        return s

    def stats(self):
        return {
            "id": self.feed_id, "name": self.name, "discipline": self.discipline,
            "running": self.running, "kind": self.source["kind"], "fit": self.fit,
            "media_url": self.media_url, "detection_mode": self.detection_mode,
            "conf_threshold": self.conf_threshold, "people": self.people, "objects": self.objects,
            "motion": round(self.motion, 1), "deviation": round(self.deviation, 1),
            "activity": self.activity, "fps": round(self.fps, 1),
            "frame_width": self.frame_width, "frame_height": self.frame_height,
            "detections": self.detections if self.mode == "command" else [],
            "workflow": self.workflow_stats() if self.mode == "lab" else None,
        }


def stop_all():
    workers = list(FEEDS.values())
    for f in workers: f.stop()
    for f in workers:
        if f.thread and f.thread.is_alive(): f.thread.join(timeout=2.0)
    FEEDS.clear()


def start_mode(mode: str, selected: Optional[int] = None):
    stop_all(); srcs = source_manifest()
    if not srcs: raise HTTPException(400, "No local sources found.")
    if mode == "lab":
        cameras = [s for s in srcs if s["kind"] == "camera"]
        if not cameras: raise HTTPException(400, "No /dev/video0 camera detected. Connect the Logitech webcam.")
        chosen = [cameras[0]]
    elif mode == "command":
        chosen = sorted([s for s in srcs if s["kind"] == "video"], key=lambda x: x.get("slot", 99))[:4]
        if not chosen: raise HTTPException(400, "No Command Center videos found.")
    else:
        raise HTTPException(400, "Unknown mode")
    for i, src in enumerate(chosen):
        w = FeedWorker(i, src, mode); FEEDS[i] = w; w.start()
    add_event(-1, "SYSTEM", "mode", f"{'Autonomous Research Lab' if mode == 'lab' else 'Multi-Lab Command Center'} started locally.")
    return [f.stats() for f in FEEDS.values()]


def network_state():
    active = []
    try:
        p = subprocess.run(["ip", "-4", "-o", "addr", "show", "up"], capture_output=True, text=True, timeout=2)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] != "lo": active.append(parts[1])
    except Exception: pass
    try:
        route = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        route = ""
    online = bool(active and route)
    if not online and any(f.running for f in FEEDS.values()):
        with SESSION_LOCK: SESSION["offline_proven"] = True
    return {"online": online, "interfaces": sorted(set(active)), "cloud_dependency": "NONE"}


def system_metrics():
    out = {
        "gpu": {}, "memory": {}, "network": network_state(), "reasoning": dict(REASONING),
        "llama_loaded": model_loaded() if not MOCK else False, "vision": VISION.info(),
    }
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
        vals = [x.strip() for x in p.stdout.strip().split(",")]
        if len(vals) >= 3: out["gpu"] = {"util": vals[0], "temp": vals[1], "power": vals[2]}
    except Exception: pass
    try:
        d = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1); d[k] = int(v.strip().split()[0])
        total = d.get("MemTotal", 0); avail = d.get("MemAvailable", 0)
        out["memory"] = {"total_gb": round(total / 1024 / 1024, 1), "used_gb": round((total - avail) / 1024 / 1024, 1)}
    except Exception: pass
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text()


@app.get("/api/status")
def status():
    return {
        "ok": True, "version": VERSION, "vision": VISION.info(),
        "ollama": ollama_available() if not MOCK else False,
        "llama_loaded": model_loaded() if not MOCK else False,
        "sources": source_manifest(), "feeds": [f.stats() for f in FEEDS.values()],
        "reasoning": dict(REASONING), "auto_reason": dict(AUTO_REASON),
        "network": network_state(), "value": session_value(),
    }


@app.get("/api/dashboard")
def dashboard():
    return {
        "feeds": [f.stats() for f in FEEDS.values()], "events": list(EVENTS)[:60],
        "reasoning": dict(REASONING), "system": system_metrics(),
        "auto_reason": dict(AUTO_REASON), "value": session_value(),
    }


@app.post("/api/start/{mode}")
def api_start(mode: str, body: dict = Body(default={})):
    return {"feeds": start_mode(mode, body.get("source_index"))}


@app.post("/api/stop")
def api_stop():
    stop_all(); add_event(-1, "SYSTEM", "mode", "Demo workloads stopped."); return {"ok": True}


@app.post("/api/reset-workflow")
def reset_workflow():
    for f in FEEDS.values():
        if f.mode == "lab":
            f.specimen = {"id": "SPECIMEN A", "zone": "NOT SEEN", "candidate": None, "candidate_since": 0.0, "zone_since": 0.0, "sequence": [], "durations": {}, "complete": False, "last_seen": 0.0}
            f.person_in_analysis = False; f.person_dwell_alerted = False
    add_event(-1, "SYSTEM", "workflow", "Live workflow reset."); return {"ok": True}


@app.get("/stream/{feed_id}")
def stream(feed_id: int):
    if feed_id not in FEEDS: raise HTTPException(404, "Feed not running")
    w = FEEDS[feed_id]
    if w.mode != "lab": raise HTTPException(400, "Command Center media is rendered natively by the browser.")
    def gen():
        last = None
        while w.running:
            jpg = w.latest_jpeg
            if jpg and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/analyze")
def analyze(body: dict = Body(default={})):
    if not ollama_available(): raise HTTPException(503, "Ollama is not available.")
    with STATE_LOCK:
        if REASONING["status"] in ("busy", "warming"): return {"started": False, "reason": "reasoning already active"}
        trigger = body.get("trigger") or REASONING.get("pending_trigger") or "Presenter requested the most important live system insight."
    threading.Thread(target=_reason_worker, args=(trigger,), daemon=True).start(); return {"started": True, "trigger": trigger}


@app.post("/api/prewarm")
def prewarm():
    if not ollama_available(): raise HTTPException(503, "Ollama is not available.")
    if model_loaded(): return {"started": False, "reason": "already warm"}
    def work():
        with STATE_LOCK: REASONING.update({"status": "warming", "trigger": "Prewarming Llama 3.1 70B", "started": time.time(), "error": ""})
        try:
            generate('Return valid JSON: {"ready":true}', system="You are initializing for a local live demonstration.", keep_alive="30m")
            with STATE_LOCK: REASONING.update({"status": "idle", "finished": time.time(), "error": ""})
        except Exception as e:
            with STATE_LOCK: REASONING.update({"status": "error", "error": str(e), "finished": time.time()})
    threading.Thread(target=work, daemon=True).start(); return {"started": True}


@app.post("/api/auto-reason")
def auto_reason(body: dict = Body(...)):
    AUTO_REASON["enabled"] = bool(body.get("enabled")); return dict(AUTO_REASON)


@app.post("/api/memory/search")
def memory(body: dict = Body(...)):
    q = (body.get("query") or "").strip(); return {"results": memory_search(q, 5) if q else []}


@app.on_event("shutdown")
def on_shutdown():
    stop_all()
