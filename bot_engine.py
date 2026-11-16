#!/usr/bin/env python3
"""
Telegram Auto-Clipper (single file).

Send a video file or a link in Telegram  ->  bot finds the best moments with Gemini,
crops them to 9:16 with Autocrop-vertical (YOLOv8), burns word-by-word captions + a hook
banner, and sends the finished clips back in the same chat. Everything is controlled from
Telegram with commands (send /help).

Only clip content you have the right to use (your own videos, or sources that an
active Whop / Content Rewards campaign explicitly gives you).
"""
import os
import re
import sys
import json
import glob
import time
import shutil
import asyncio
import logging
import subprocess
import traceback
import collections
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import yt_dlp
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from telethon.tl.functions.bots import SetBotCommandsRequest
from telethon.tl.types import BotCommand, BotCommandScopeDefault
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# --------------------------------------------------------------------------- settings
API_ID = int(os.getenv("TG_API_ID", "0") or 0)
API_HASH = os.getenv("TG_API_HASH", "")
BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# comma separated: your user id, your sister's user id, and/or the group id (negative number)
ALLOWED = {int(x) for x in re.split(r"[,\s]+", os.getenv("ALLOWED_USERS", "").strip()) if x.lstrip("-").isdigit()}
GH_PAT = os.getenv("GH_PAT", "")                       # optional: lets the bot save settings permanently
GH_REPO = os.getenv("GITHUB_REPOSITORY", "")           # set automatically by GitHub Actions
RUN_SECONDS = int(os.getenv("RUN_SECONDS", "0") or 0)  # 0 = run forever (VPS/PC); Actions sets ~19800
AUTOCROP_DIR = os.getenv("AUTOCROP_DIR", "/tmp/Autocrop")
WORKDIR = os.path.abspath(os.getenv("WORKDIR", "work"))
IST = ZoneInfo("Asia/Kolkata")

# Verified against Google's model list on 2026-11-16. The bot tries these in order if the
# selected model is rejected, and /models shows the live list from your own API key.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

DEFAULTS = {
    "model": "gemini-3.8-flash",
    "sources": [],            # channel URLs / search phrases used by /daily (only ones you may clip)
    "daily_on": True,
    "daily_time": "08:00",    # IST
    "daily_count": 6,
    "daily_chat": None,
    "manual_clips": 4,
    "clip_min": 25,
    "clip_max": 55,
    "captions": True,
    "hook": True,
    "whisper": "base",        # tiny | base | small
    "lang": "auto",           # auto | en | hi | ...
    "processed": [],          # video ids already clipped by /daily
    "last_daily": "",
}

# --------------------------------------------------------------------------- logging
RING = collections.deque(maxlen=300)


class RingHandler(logging.Handler):
    def emit(self, record):
        RING.append(self.format(record))


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
_rh = RingHandler()
_rh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
logging.getLogger().addHandler(_rh)
logging.getLogger("telethon").setLevel(logging.WARNING)
log = logging.getLogger("clipper")

# --------------------------------------------------------------------------- config (persisted)
CFG = dict(DEFAULTS)
try:
    CFG.update(json.loads(os.getenv("BOT_CONFIG", "") or "{}"))
except Exception as e:  # noqa: BLE001
    log.warning("BOT_CONFIG could not be parsed: %s", e)


def persist_config() -> bool:
    """Save CFG into the repo variable BOT_CONFIG (needs GH_PAT). Returns True if saved."""
    if not (GH_PAT and GH_REPO):
        return False
    CFG["processed"] = CFG["processed"][-150:]
    headers = {"Authorization": f"Bearer {GH_PAT}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    body = {"name": "BOT_CONFIG", "value": json.dumps(CFG, separators=(",", ":"))}
    base = f"https://api.github.com/repos/{GH_REPO}/actions/variables"
    try:
        r = requests.patch(f"{base}/BOT_CONFIG", headers=headers, json=body, timeout=20)
        if r.status_code == 404:
            r = requests.post(base, headers=headers, json=body, timeout=20)
        if r.status_code not in (200, 201, 204):
            log.warning("persist_config failed: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("persist_config error: %s", e)
        return False


async def save_cfg() -> str:
    ok = await asyncio.to_thread(persist_config)
    return "(saved)" if ok else "(this session only - add the GH_PAT secret to make settings permanent)"


# --------------------------------------------------------------------------- shell helpers
def run(cmd, timeout=None, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-600:]
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed ({p.returncode}): {tail}")
    return p.stdout


def probe_duration(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def ff_escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


# --------------------------------------------------------------------------- Gemini
def _client():
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


def list_text_models():
    """Models from your own API key that can generate text from audio (live list)."""
    out = []
    for m in _client().models.list():
        name = (m.name or "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        if not name.startswith("gemini") and "latest" not in name:
            continue
        if any(bad in name for bad in ("tts", "live", "image", "embedding", "robotics", "omni",
                                       "transcribe", "translate", "computer-use", "veo", "imagen")):
            continue
        out.append(name)
    return sorted(set(out))


def test_model(name: str) -> str:
    from google.genai import types
    r = _client().models.generate_content(
        model=name, contents="Reply with the single word: ok",
        config=types.GenerateContentConfig(temperature=0))
    return (r.text or "").strip()[:40]


def parse_time(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = [float(x) for x in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_moments(text: str, duration: float, n: int, cmin: int, cmax: int):
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("clips") or data.get("moments") or next(iter(data.values()), [])
    moments = []
    for it in data:
        try:
            s, e = parse_time(it["start"]), parse_time(it["end"])
        except Exception:  # noqa: BLE001
            continue
        s = max(0.0, s)
        e = min(duration, e)
        if e - s > cmax + 10:
            e = s + cmax
        if e - s < 10:
            continue
        moments.append({"start": s, "end": e,
                        "hook": str(it.get("hook") or "Watch this")[:60],
                        "title": str(it.get("title") or "Viral clip #Shorts")[:95]})
    moments.sort(key=lambda m: m["start"])
    clean, last_end = [], -1.0
    for m in moments:
        if m["start"] >= last_end:
            clean.append(m)
            last_end = m["end"]
    return clean[:n]


def find_moments(video_path: str, n: int):
    """Returns (moments, model_used)."""
    from google.genai import types
    dur = probe_duration(video_path)
    mp3 = video_path.rsplit(".", 1)[0] + "_audio.mp3"
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", mp3])
    client = _client()
    up = client.files.upload(file=mp3)
    try:
        for _ in range(60):
            state = getattr(up.state, "name", str(up.state))
            if "PROCESSING" not in state:
                break
            time.sleep(2)
            up = client.files.get(name=up.name)
        cmin, cmax = int(CFG["clip_min"]), int(CFG["clip_max"])
        prompt = (
            f"You are an expert short-form video editor. This audio is from a {dur/60:.1f}-minute video. "
            f"Pick the {n} best moments for YouTube Shorts: strong hook in the first seconds, a complete "
            f"thought or story beat, emotional / funny / surprising / insightful. Each clip must be between "
            f"{cmin} and {cmax} seconds, must not overlap another clip, and must start and end on natural "
            f"sentence boundaries. Return ONLY JSON: a list of objects with keys "
            f'"start" (seconds from video start, number), "end" (seconds, number), '
            f'"hook" (max 6 words, shown as on-screen text), '
            f'"title" (max 80 chars, catchy, ending with #Shorts). Sort by start time.'
        )
        tried, last_err = [], None
        for m in dict.fromkeys([CFG["model"]] + FALLBACK_MODELS):
            try:
                resp = client.models.generate_content(
                    model=m, contents=[up, prompt],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.3))
                moments = parse_moments(resp.text, dur, n, cmin, cmax)
                if moments:
                    return moments, m
                last_err = RuntimeError(f"{m} returned no usable moments")
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("model %s failed: %s", m, str(e)[:200])
            tried.append(m)
        raise RuntimeError(f"Gemini failed on {', '.join(tried)}. Last error: {str(last_err)[:300]}")
    finally:
        try:
            client.files.delete(name=up.name)
        except Exception:  # noqa: BLE001
            pass
        if os.path.exists(mp3):
            os.remove(mp3)


# --------------------------------------------------------------------------- captions (ASS)
_whisper = {}


def transcribe_words(path: str):
    size = CFG["whisper"]
    if size not in _whisper:
        from faster_whisper import WhisperModel
        _whisper[size] = WhisperModel(size, device="cpu", compute_type="int8")
    lang = None if CFG["lang"] == "auto" else CFG["lang"]
    segments, _info = _whisper[size].transcribe(path, word_timestamps=True, vad_filter=True,
                                                beam_size=1, language=lang)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            t = w.word.strip()
            if t:
                words.append((t, float(w.start), float(w.end)))
    return words


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}"


def _ass_text(s: str) -> str:
    return re.sub(r"[{}\\\r\n]", " ", s).strip()


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Noto Sans,80,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,560,1
Style: Hook,Noto Sans,66,&H00FFFFFF,&H000000FF,&HB4000000,&H00000000,-1,0,0,0,100,100,0,0,3,16,0,8,70,70,300,1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(words, hook: str, path: str, clip_len: float, words_per_line: int = 3) -> bool:
    lines = []
    if CFG["hook"] and hook:
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(min(3.5, clip_len))},Hook,0,0,0,,{_ass_text(hook)}")
    if CFG["captions"] and words:
        chunks, cur = [], []
        for w in words:
            if cur and (len(cur) >= words_per_line or w[1] - cur[-1][2] > 0.8):
                chunks.append(cur)
                cur = []
            cur.append(w)
        if cur:
            chunks.append(cur)
        for ch in chunks:
            for i, w in enumerate(ch):
                start = w[1]
                end = ch[i + 1][1] if i + 1 < len(ch) else w[2] + 0.12
                parts = []
                for j, x in enumerate(ch):
                    t = _ass_text(x[0]).upper()
                    parts.append("{\\c&H00FFFF&}" + t + "{\\r}" if j == i else t)
                lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,0,0,0,,{' '.join(parts)}")
    if not lines:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(ASS_HEAD + "\n".join(lines) + "\n")
    return True


# --------------------------------------------------------------------------- video processing
def process_clip(src: str, m: dict, out_path: str):
    base = out_path[:-4]
    raw, cropped, ass = base + "_raw.mp4", base + "_crop.mp4", base + ".ass"
    try:
        # 1) cut the segment
        run(["ffmpeg", "-y", "-ss", f"{m['start']:.2f}", "-i", src, "-t", f"{m['end'] - m['start']:.2f}",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-c:a", "aac", "-b:a", "128k", raw])
        # 2) smart 9:16 crop (YOLOv8 person tracking); fall back to a plain center crop
        try:
            run(["python3", os.path.join(AUTOCROP_DIR, "main.py"), "-i", raw, "-o", cropped,
                 "--ratio", "9:16", "--quality", "fast", "--frame-skip", "1"],
                timeout=1500, cwd=AUTOCROP_DIR)
        except Exception as e:  # noqa: BLE001
            log.warning("autocrop failed, using center crop: %s", str(e)[:300])
            run(["ffmpeg", "-y", "-i", raw, "-vf", "crop=ih*9/16:ih", "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "22", "-c:a", "copy", cropped])
        # 3) captions + hook
        words = []
        if CFG["captions"]:
            try:
                words = transcribe_words(cropped)
            except Exception as e:  # noqa: BLE001
                log.warning("transcription failed (clip sent without captions): %s", str(e)[:300])
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        if build_ass(words, m.get("hook", ""), ass, m["end"] - m["start"]):
            vf += f",ass='{ff_escape(ass)}'"
        run(["ffmpeg", "-y", "-i", cropped, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-movflags", "+faststart", out_path])
    finally:
        for f in (raw, cropped, ass):
            if os.path.exists(f):
                os.remove(f)


def download_video(url: str, out_base: str) -> str:
    opts = {"format": "bv*[height<=720]+ba/b[height<=720]/b", "merge_output_format": "mp4",
            "outtmpl": out_base + ".%(ext)s", "quiet": True, "noprogress": True, "noplaylist": True,
            "socket_timeout": 30, "retries": 3}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not download that link (YouTube often blocks cloud servers). "
            "Download the video to your phone/PC and send the video FILE here instead. "
            f"Details: {str(e)[:200]}")
    found = sorted(glob.glob(out_base + ".*"))
    if not found:
        raise RuntimeError("Download finished but no file was found.")
    return found[0]


def pick_daily_video():
    """Newest not-yet-clipped video from the configured sources. Returns (url, title, id) or None."""
    cands = []
    for src in CFG["sources"]:
        target = src
        if not src.startswith("http"):
            target = f"ytsearchdate5:{src}"
        elif re.search(r"youtube\.com/(@|c/|channel/|user/)", src) and not re.search(r"/(videos|streams|shorts)/?$", src):
            target = src.rstrip("/") + "/videos"
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "playlistend": 8,
                                   "socket_timeout": 30}) as ydl:
                info = ydl.extract_info(target, download=False)
        except Exception as e:  # noqa: BLE001
            log.warning("source %s failed: %s", src, str(e)[:200])
            continue
        entries = []
        for e in (info.get("entries") or []):
            entries.extend(e.get("entries") or [e])
        for e in entries:
            vid, dur = e.get("id"), e.get("duration")
            if not vid or vid in CFG["processed"]:
                continue
            if dur is not None and not (240 <= dur <= 5400):
                continue
            cands.append((e.get("view_count") or 0, f"https://www.youtube.com/watch?v={vid}", e.get("title") or vid, vid))
            break  # newest eligible video of this source
    if not cands:
        return None
    cands.sort(key=lambda c: c[0], reverse=True)
    _, url, title, vid = cands[0]
    return url, title, vid


# --------------------------------------------------------------------------- bot
bot = TelegramClient(StringSession(), API_ID, API_HASH)
queue: "asyncio.Queue[Job]" = asyncio.Queue()
state = {"busy": None, "done": 0, "failed": 0}
scheduler = AsyncIOScheduler(timezone=IST)


@dataclass
class Job:
    chat_id: int
    kind: str            # "file" or "url"
    payload: object      # telethon Message or url string
    n: int
    label: str = ""


async def say(chat_id, text):
    try:
        await bot.send_message(chat_id, text)
    except Exception as e:  # noqa: BLE001
        log.warning("send_message failed: %s", e)


async def run_job(job: Job):
    os.makedirs(WORKDIR, exist_ok=True)
    stamp = int(time.time())
    base = os.path.join(WORKDIR, f"in_{stamp}")
    video = None
    try:
        if job.kind == "file":
            await say(job.chat_id, "📥 Downloading your video...")
            video = await job.payload.download_media(file=base + ".mp4")
        else:
            await say(job.chat_id, f"🔗 Downloading: {job.label or job.payload}")
            video = await asyncio.to_thread(download_video, job.payload, base)
        await say(job.chat_id, f"🧠 Finding the best {job.n} moments with {CFG['model']}...")
        moments, used = await asyncio.to_thread(find_moments, video, job.n)
        if used != CFG["model"]:
            await say(job.chat_id, f"ℹ️ {CFG['model']} was rejected, used {used} instead. Use /models to pick a working one.")
        await say(job.chat_id, f"✂️ Found {len(moments)} clips. Rendering (about 2-4 min each)...")
        for idx, m in enumerate(moments, 1):
            out = os.path.join(WORKDIR, f"short_{stamp}_{idx}.mp4")
            try:
                await asyncio.to_thread(process_clip, video, m, out)
                title = m["title"].replace("`", "'")
                cap = (f"✅ **Clip {idx}/{len(moments)}**\n\n📋 **Title (tap to copy):**\n`{title}`\n\n"
                       f"⏱ Source {int(m['start'])}s - {int(m['end'])}s")
                await bot.send_file(job.chat_id, out, caption=cap, supports_streaming=True)
            except Exception as e:  # noqa: BLE001
                log.error("clip %s failed: %s", idx, traceback.format_exc()[-800:])
                await say(job.chat_id, f"⚠️ Clip {idx} failed: {str(e)[:300]}")
            finally:
                if os.path.exists(out):
                    os.remove(out)
        await say(job.chat_id, "🎉 All clips delivered. Save to gallery and upload to YouTube Shorts.")
        state["done"] += 1
    except Exception as e:  # noqa: BLE001
        state["failed"] += 1
        log.error("job failed: %s", traceback.format_exc()[-1000:])
        await say(job.chat_id, f"❌ Job failed: {str(e)[:500]}")
    finally:
        for f in glob.glob(base + "*"):
            try:
                os.remove(f)
            except OSError:
                pass


async def worker():
    while True:
        job = await queue.get()
        state["busy"] = job.label or job.kind
        try:
            await run_job(job)
        finally:
            state["busy"] = None
            queue.task_done()


def default_chat():
    if CFG.get("daily_chat"):
        return int(CFG["daily_chat"])
    groups = [c for c in ALLOWED if c < 0]
    if groups:
        return groups[0]
    return next(iter(ALLOWED), None)


async def daily_job():
    chat = default_chat()
    if chat is None:
        return
    CFG["last_daily"] = datetime.now(IST).strftime("%Y-%m-%d")
    if not CFG["sources"]:
        await say(chat, "⚠️ Daily run skipped: no sources set. Add one with /addsource <channel URL> "
                        "(only channels whose content you are allowed to clip).")
        return
    pick = await asyncio.to_thread(pick_daily_video)
    if not pick:
        await say(chat, "⚠️ No new eligible video (4-90 min) found in your sources.")
        return
    url, title, vid = pick
    CFG["processed"].append(vid)
    await save_cfg()
    await say(chat, f"🌅 Daily run: {title}\n{url}\nMaking {CFG['daily_count']} clips...")
    await queue.put(Job(chat, "url", url, int(CFG["daily_count"]), title))


def schedule_daily():
    if scheduler.get_job("daily"):
        scheduler.remove_job("daily")
    if CFG["daily_on"]:
        h, m = (int(x) for x in CFG["daily_time"].split(":"))
        scheduler.add_job(daily_job, CronTrigger(hour=h, minute=m, timezone=IST), id="daily",
                          misfire_grace_time=3600)


def maybe_catch_up():
    """If the bot restarted shortly after the daily time and it has not run today, run it now."""
    if not CFG["daily_on"]:
        return
    now = datetime.now(IST)
    h, m = (int(x) for x in CFG["daily_time"].split(":"))
    sched = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if CFG["last_daily"] != now.strftime("%Y-%m-%d") and 0 <= (now - sched).total_seconds() <= 3 * 3600:
        asyncio.create_task(daily_job())


HELP = """🎬 **Clipper Bot - command list**

**Make clips**
Send a **video file** or a **YouTube/Twitch link** -> clips come back here.
/clips <1-10> - clips per manual video
/cliplen <min> <max> - clip length in seconds (e.g. /cliplen 25 55)

**Daily auto clips**
/daily - run the daily job now
/dailyon, /dailyoff
/dailytime HH:MM - IST, 24h (e.g. /dailytime 08:00)
/dailycount <1-10> - clips per day
/dailyhere - send daily clips to THIS chat
/sources - list sources
/addsource <channel URL or search words>
/delsource <number>, /clearsource

**AI / look**
/models - models available to your API key
/model <name> - switch + test model
/captions on|off - word-by-word captions
/hook on|off - top hook banner
/whisper tiny|base|small - caption accuracy vs speed
/lang auto|en|hi - speech language for captions

**System**
/status, /ping, /queue, /clearqueue, /logs, /myid, /resetconfig"""

MENU = [("help", "Show all commands"), ("status", "Current settings & queue"), ("daily", "Run daily job now"),
        ("models", "List Gemini models"), ("model", "Switch Gemini model"), ("sources", "List daily sources"),
        ("addsource", "Add a daily source"), ("delsource", "Remove a source"), ("dailytime", "Set daily time (IST)"),
        ("dailycount", "Clips per day"), ("dailyon", "Enable daily job"), ("dailyoff", "Disable daily job"),
        ("dailyhere", "Send daily clips to this chat"), ("clips", "Clips per manual video"),
        ("cliplen", "Clip length range"), ("captions", "Captions on/off"), ("hook", "Hook banner on/off"),
        ("whisper", "Caption model size"), ("lang", "Caption language"), ("queue", "Show queue"),
        ("clearqueue", "Clear waiting jobs"), ("logs", "Recent logs"), ("ping", "Is the bot alive?"),
        ("myid", "Show chat & user id"), ("resetconfig", "Reset settings")]


def status_text():
    src = "\n".join(f"  {i}. {s}" for i, s in enumerate(CFG["sources"], 1)) or "  (none)"
    return (f"🟢 **Bot online**\nModel: `{CFG['model']}`\nCaptions: {CFG['captions']} | Hook: {CFG['hook']} | "
            f"Whisper: {CFG['whisper']} | Lang: {CFG['lang']}\n"
            f"Manual clips/video: {CFG['manual_clips']} | Length: {CFG['clip_min']}-{CFG['clip_max']}s\n"
            f"Daily: {'ON' if CFG['daily_on'] else 'OFF'} at {CFG['daily_time']} IST, {CFG['daily_count']} clips, "
            f"chat `{default_chat()}`\nSources:\n{src}\n"
            f"Queue: {queue.qsize()} waiting | Now: {state['busy'] or 'idle'} | "
            f"Done: {state['done']} | Failed: {state['failed']}\n"
            f"Permanent settings: {'yes' if GH_PAT else 'no (add GH_PAT secret)'}")


def parse_cmd(text: str):
    first, _, rest = text.partition(" ")
    cmd = first.split("@")[0].lower() if first.startswith("/") else ""
    return cmd, rest.strip()


def _int(arg, lo, hi):
    try:
        v = int(arg)
        return v if lo <= v <= hi else None
    except (TypeError, ValueError):
        return None


def authorized(event) -> bool:
    return event.chat_id in ALLOWED or event.sender_id in ALLOWED


async def handle(event):
    text = (event.raw_text or "").strip()
    cmd, arg = parse_cmd(text)
    chat = event.chat_id

    if cmd == "/myid":
        await event.reply(f"Chat ID: `{chat}`\nYour user ID: `{event.sender_id}`")
        return
    if not authorized(event):
        if cmd:
            await event.reply("🔒 Not authorised. Send /myid and add those numbers to the ALLOWED_USERS secret.")
        return

    # ---- commands
    if cmd in ("/start", "/help"):
        await event.reply(HELP)
    elif cmd == "/ping":
        await event.reply("🟢 Alive and polling.")
    elif cmd == "/status":
        await event.reply(status_text())
    elif cmd == "/logs":
        await event.reply("```\n" + "\n".join(list(RING)[-25:])[-3500:] + "\n```")
    elif cmd == "/queue":
        await event.reply(f"Queue: {queue.qsize()} waiting. Now: {state['busy'] or 'idle'}")
    elif cmd == "/clearqueue":
        n = 0
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
            n += 1
        await event.reply(f"🧹 Removed {n} waiting job(s). The running one (if any) will finish.")
    elif cmd == "/models":
        try:
            names = await asyncio.to_thread(list_text_models)
            await event.reply("🤖 **Models on your API key:**\n" + "\n".join(f"`{n}`" for n in names[:40]) +
                              f"\n\nCurrent: `{CFG['model']}`\nSwitch: /model <name>")
        except Exception as e:  # noqa: BLE001
            await event.reply(f"❌ Could not list models: {str(e)[:300]}\nCheck GEMINI_API_KEY (it starts with AIza).")
    elif cmd == "/model":
        if not arg:
            await event.reply(f"Current model: `{CFG['model']}`\nUse /models to see options, then /model <name>")
            return
        name = arg.replace("models/", "").strip()
        await event.reply(f"🧪 Testing `{name}`...")
        try:
            reply = await asyncio.to_thread(test_model, name)
            CFG["model"] = name
            await event.reply(f"✅ Model set to `{name}` (it answered: {reply}) {await save_cfg()}")
        except Exception as e:  # noqa: BLE001
            await event.reply(f"❌ `{name}` did not work, model unchanged.\n{str(e)[:300]}")
    elif cmd == "/sources":
        await event.reply("📚 **Sources:**\n" + ("\n".join(f"{i}. {s}" for i, s in enumerate(CFG["sources"], 1))
                                                or "(none) - add with /addsource"))
    elif cmd == "/addsource":
        if not arg:
            await event.reply("Usage: /addsource https://www.youtube.com/@channel  (or search words)\n"
                              "Only add sources you are allowed to clip.")
            return
        CFG["sources"].append(arg)
        await event.reply(f"➕ Added. Total {len(CFG['sources'])}. {await save_cfg()}")
    elif cmd == "/delsource":
        i = _int(arg, 1, len(CFG["sources"]) or 1)
        if i is None or not CFG["sources"]:
            await event.reply("Usage: /delsource <number> (see /sources)")
            return
        removed = CFG["sources"].pop(i - 1)
        await event.reply(f"➖ Removed: {removed} {await save_cfg()}")
    elif cmd == "/clearsource":
        CFG["sources"] = []
        await event.reply(f"🧹 Sources cleared. {await save_cfg()}")
    elif cmd == "/daily":
        await event.reply("🔥 Running the daily job now...")
        await daily_job()
    elif cmd in ("/dailyon", "/dailyoff"):
        CFG["daily_on"] = cmd == "/dailyon"
        schedule_daily()
        await event.reply(f"Daily job {'ENABLED' if CFG['daily_on'] else 'DISABLED'}. {await save_cfg()}")
    elif cmd == "/dailytime":
        if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", arg or ""):
            await event.reply("Usage: /dailytime 08:00 (24h, IST)")
            return
        h, m = arg.split(":")
        CFG["daily_time"] = f"{int(h):02d}:{m}"
        schedule_daily()
        await event.reply(f"⏰ Daily time set to {CFG['daily_time']} IST. {await save_cfg()}")
    elif cmd == "/dailycount":
        v = _int(arg, 1, 10)
        if v is None:
            await event.reply("Usage: /dailycount 6  (1-10)")
            return
        CFG["daily_count"] = v
        await event.reply(f"Daily clips: {v}. {await save_cfg()}")
    elif cmd == "/dailyhere":
        CFG["daily_chat"] = chat
        await event.reply(f"📍 Daily clips will be sent to this chat. {await save_cfg()}")
    elif cmd == "/clips":
        v = _int(arg, 1, 10)
        if v is None:
            await event.reply("Usage: /clips 4  (1-10)")
            return
        CFG["manual_clips"] = v
        await event.reply(f"Manual clips per video: {v}. {await save_cfg()}")
    elif cmd == "/cliplen":
        try:
            a, b = (int(x) for x in arg.split())
            assert 10 <= a < b <= 90
        except Exception:  # noqa: BLE001
            await event.reply("Usage: /cliplen 25 55  (seconds, min 10, max 90)")
            return
        CFG["clip_min"], CFG["clip_max"] = a, b
        await event.reply(f"Clip length {a}-{b}s. {await save_cfg()}")
    elif cmd in ("/captions", "/hook"):
        if arg.lower() not in ("on", "off"):
            await event.reply(f"Usage: {cmd} on|off")
            return
        CFG[cmd[1:]] = arg.lower() == "on"
        await event.reply(f"{cmd[1:].title()}: {arg.lower()}. {await save_cfg()}")
    elif cmd == "/whisper":
        if arg not in ("tiny", "base", "small"):
            await event.reply("Usage: /whisper tiny|base|small  (small = most accurate, slowest)")
            return
        CFG["whisper"] = arg
        await event.reply(f"Caption model: {arg}. {await save_cfg()}")
    elif cmd == "/lang":
        if not re.fullmatch(r"auto|[a-z]{2}", arg or ""):
            await event.reply("Usage: /lang auto | en | hi | ...")
            return
        CFG["lang"] = arg
        await event.reply(f"Caption language: {arg}. {await save_cfg()}")
    elif cmd == "/resetconfig":
        CFG.clear()
        CFG.update(DEFAULTS)
        schedule_daily()
        await event.reply(f"♻️ Settings reset to defaults. {await save_cfg()}")
    elif cmd:
        await event.reply("Unknown command. Send /help")

    # ---- media / links
    else:
        msg = event.message
        is_video = bool(msg.video) or bool(msg.document and (msg.document.mime_type or "").startswith("video/"))
        m = re.search(r"https?://\S+", text)
        n = int(CFG["manual_clips"])
        if is_video:
            await queue.put(Job(chat, "file", msg, n, "uploaded video"))
            await event.reply(f"✅ Video queued ({queue.qsize()} waiting). Making {n} clips...")
        elif m:
            url = m.group(0)
            await queue.put(Job(chat, "url", url, n, url))
            await event.reply(f"✅ Link queued ({queue.qsize()} waiting). Making {n} clips...")
        elif event.is_private:
            await event.reply("Send a video file or a link, or /help for commands.")


async def main():
    if not (API_ID and API_HASH and BOT_TOKEN and GEMINI_API_KEY):
        sys.exit("Missing secrets: TG_API_ID, TG_API_HASH, TG_BOT_TOKEN, GEMINI_API_KEY")
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    await bot.start(bot_token=BOT_TOKEN)
    bot.add_event_handler(handle, events.NewMessage(incoming=True))
    try:
        await bot(SetBotCommandsRequest(scope=BotCommandScopeDefault(), lang_code="",
                                        commands=[BotCommand(c, d) for c, d in MENU]))
    except Exception as e:  # noqa: BLE001
        log.warning("could not register command menu: %s", e)
    scheduler.start()
    schedule_daily()
    asyncio.create_task(worker())
    maybe_catch_up()
    me = await bot.get_me()
    log.info("Bot @%s online. model=%s allowed=%d persistent=%s", me.username, CFG["model"], len(ALLOWED), bool(GH_PAT))
    try:
        if RUN_SECONDS:
            await asyncio.wait_for(bot.run_until_disconnected(), timeout=RUN_SECONDS)
        else:
            await bot.run_until_disconnected()
    except asyncio.TimeoutError:
        log.info("Run window over; letting the running job finish, then restarting via the scheduler.")
        try:
            await asyncio.wait_for(queue.join(), timeout=1200)
        except asyncio.TimeoutError:
            pass
    finally:
        await bot.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
