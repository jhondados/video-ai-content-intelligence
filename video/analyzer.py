"""Video AI content analyzer."""
from google.genai import Client, types
import whisper
from pathlib import Path
from typing import List, Dict
import subprocess, tempfile

class VideoContentAnalyzer:
    def __init__(self):
        self.gemini = Client()
        self.whisper_model = whisper.load_model("large-v3")

    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[str]:
        """Extract frames from video at specified FPS."""
        out_dir = tempfile.mkdtemp()
        cmd = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", f"{out_dir}/frame_%04d.jpg", "-y", "-loglevel", "quiet"]
        subprocess.run(cmd, check=True)
        return sorted(Path(out_dir).glob("*.jpg"))

    def transcribe(self, video_path: str) -> Dict:
        """Multi-speaker transcription with timestamps."""
        audio_path = video_path.replace(".mp4", "_audio.wav")
        subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path, "-y", "-loglevel", "quiet"])
        result = self.whisper_model.transcribe(audio_path, language="pt", word_timestamps=True)
        return {"text": result["text"], "segments": result["segments"], "language": result["language"]}

    def generate_chapters(self, transcript: str, video_duration_sec: int) -> List[Dict]:
        """Auto-generate video chapters from transcript."""
        prompt = f"""Given this video transcript ({video_duration_sec}s total), generate 5-10 chapter markers.
Transcript: {transcript[:4000]}
Return JSON list: [{{"time_sec": 0, "title": "Introduction"}}, ...]
Spread evenly, make titles concise and descriptive."""
        import json
        resp = self.gemini.models.generate_content(model="gemini-2.0-flash-exp", contents=prompt)
        text = resp.text
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        return json.loads(text)

    def moderate_content(self, video_path: str) -> Dict:
        """Detect harmful content in video."""
        frames = self.extract_frames(video_path, fps=0.5)[:30]  # Sample 30 frames
        parts = [types.Part.from_bytes(data=f.read_bytes(), mime_type="image/jpeg") for f in frames]
        prompt = "Analyze these video frames for: violence, explicit content, hate speech, dangerous activities. Rate each 0-1. Return JSON: {violence: 0.0, explicit: 0.0, hate_speech: 0.0, dangerous: 0.0, overall_safe: true}"
        import json
        resp = self.gemini.models.generate_content(model="gemini-2.0-flash-exp", contents=parts + [prompt])
        text = resp.text
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        return json.loads(text)
