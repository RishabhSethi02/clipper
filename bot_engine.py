import os, time, json, asyncio, subprocess
from telethon import TelegramClient, events
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import yt_dlp
from google import genai

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
ALLOWED_USERS = [int(x.strip()) for x in os.environ.get("ALLOWED_USERS", "").split(",") if x.strip()]
NICHE_QUERY = os.environ.get("TRENDING_NICHE", "podcast clips OR streamer highlights")

WORKDIR = "/tmp/clipper_jobs"
os.makedirs(WORKDIR, exist_ok=True)
ai_client = genai.Client(api_key=GEMINI_KEY)
bot = TelegramClient('/tmp/bot_session', API_ID, API_HASH)

def download_youtube_video(url: str, out_path: str) -> str:
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        'outtmpl': out_path,
        'merge_output_format': 'mp4',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info.get('title', 'Video')

def find_viral_timestamps_with_gemini(video_path: str, num_clips: int = 6):
    audio_path = video_path.replace(".mp4", ".mp3")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    uploaded_audio = ai_client.files.upload(file=audio_path)
    while uploaded_audio.state.name == "PROCESSING":
        time.sleep(3)
        uploaded_audio = ai_client.files.get(name=uploaded_audio.name)

    prompt = f"""
    Listen to this audio and pick the {num_clips} most viral, high-retention segments suitable for YouTube Shorts (each strictly between 25 and 55 seconds long).
    Return ONLY a raw JSON list in this exact format:
    [
      {{"start": "00:01:15", "end": "00:01:55", "title": "Shocking Truth About... #Shorts", "hook": "Wait till the end"}}
    ]
    """
    resp = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_audio, prompt],
        config={"response_mime_type": "application/json"}
    )
    ai_client.files.delete(name=uploaded_audio.name)
    os.remove(audio_path)
    return json.loads(resp.text)

def process_clip_with_autocrop(source_mp4: str, start: str, end: str, hook: str, out_mp4: str):
    raw_cut = out_mp4.replace(".mp4", "_raw.mp4")
    cropped_cut = out_mp4.replace(".mp4", "_cropped.mp4")

    subprocess.run([
        "ffmpeg", "-y", "-ss", start, "-to", end, "-i", source_mp4,
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", raw_cut
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "python3", "/tmp/Autocrop/main.py",
        "-i", raw_cut, "-o", cropped_cut,
        "--ratio", "9:16", "--quality", "fast", "--frame-skip", "1"
    ], check=True)

    safe_hook = hook.replace("'", "").replace(":", "")[:35]
    subprocess.run([
        "ffmpeg", "-y", "-i", cropped_cut,
        "-vf", f"drawtext=text='{safe_hook}':fontcolor=white:fontsize=42:box=1:boxcolor=black@0.75:boxborderw=12:x=(w-text_w)/2:y=120",
        "-c:a", "copy", out_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for f in [raw_cut, cropped_cut]:
        if os.path.exists(f):
            os.remove(f)

async def run_pipeline(chat_id: int, video_path: str, num_clips: int = 6):
    await bot.send_message(chat_id, f"🧠 Finding top {num_clips} viral moments & running YOLOv8 9:16 tracking...")
    clips_meta = await asyncio.to_thread(find_viral_timestamps_with_gemini, video_path, num_clips)

    for idx, item in enumerate(clips_meta, 1):
        out_clip = os.path.join(WORKDIR, f"short_{int(time.time())}_{idx}.mp4")
        try:
            await asyncio.to_thread(
                process_clip_with_autocrop,
                video_path, item["start"], item["end"], item.get("hook", "Watch This!"), out_clip
            )
            caption = (
                f"✅ **Clip {idx}/{len(clips_meta)} Ready!**\n\n"
                f"📋 **Copy-Paste Title for YouTube:**\n`{item.get('title', 'Viral Clip #Shorts')}`\n\n"
                f"⏱ **Segment:** {item['start']} - {item['end']}"
            )
            await bot.send_file(chat_id, out_clip, caption=caption)
        except Exception as e:
            await bot.send_message(chat_id, f"⚠️ Clip {idx} error: {e}")
        finally:
            if os.path.exists(out_clip):
                os.remove(out_clip)

    if os.path.exists(video_path):
        os.remove(video_path)
    await bot.send_message(chat_id, "🎉 All clips delivered! Save to gallery & upload to YouTube Shorts.")

async def daily_6_trending_clips():
    ydl_opts = {'quiet': True, 'extract_flat': True, 'playlistend': 10}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearchdate10:{NICHE_QUERY}", download=False)
        candidates = [e for e in info.get('entries', []) if 240 <= (e.get('duration') or 0) <= 2400]
        candidates.sort(key=lambda x: x.get('view_count') or 0, reverse=True)

    if not candidates:
        return
    top_url = f"https://www.youtube.com/watch?v={candidates[0]['id']}"
    job_mp4 = os.path.join(WORKDIR, f"daily_{int(time.time())}.mp4")

    for uid in ALLOWED_USERS:
        await bot.send_message(uid, f"🌅 **Daily Automatic 6 Clips Started!**\nTrending Video: {top_url}")
    await asyncio.to_thread(download_youtube_video, top_url, job_mp4)
    for uid in ALLOWED_USERS:
        await run_pipeline(uid, job_mp4, num_clips=6)

@bot.on(events.NewMessage(chats=ALLOWED_USERS))
async def handler(event):
    text = (event.raw_text or "").strip()
    job_mp4 = os.path.join(WORKDIR, f"input_{int(time.time())}.mp4")

    if text.lower() == "/daily":
        await event.reply("🔥 Triggering the 6 daily trending clips right now...")
        await daily_6_trending_clips()
    elif event.video or event.document:
        await event.reply("📥 Downloading your video file (up to 2GB supported)...")
        await event.download_media(file=job_mp4)
        await run_pipeline(event.chat_id, job_mp4, num_clips=4)
    elif text.startswith("http"):
        await event.reply("🔗 Downloading video from link...")
        await asyncio.to_thread(download_youtube_video, text, job_mp4)
        await run_pipeline(event.chat_id, job_mp4, num_clips=4)
    else:
        await event.reply("👋 Send any YouTube/Instagram link, upload a full video file, or send `/daily` to get 6 trending Shorts immediately!")

async def main():
    await bot.start(bot_token=BOT_TOKEN)
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(daily_6_trending_clips, 'cron', hour=8, minute=0)
    scheduler.start()
    # Run for 5 hours; GitHub Actions automatically queues the next 5-hour session for 24/7 uptime
    await asyncio.sleep(18000)
    await bot.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
