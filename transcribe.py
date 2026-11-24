"""Transcribe the videos in a Kearney Instagram media bundle.

Started by the platform crawler kearney_instagram_video_transcribe through AWS SSM:

    python transcribe.py --job <history id> --input <bundle url> --output s3://<bucket>/<key>.jsonl

--input   The run's media bundle: a zip holding videos/ and images/ (only videos are
          transcribed). A local path or a local folder also works, for testing.
--output  Where to write the results, one JSON line per video. An s3:// URL is
          uploaded with the instance role's credentials; anything else is a local path.

Each line: {"file", "language", "language_probability", "duration", "text", "status", "error"}
status is Success, No speech or Failed. One bad video never stops the job.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from faster_whisper import WhisperModel

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
DOWNLOAD_CHUNK = 1024 * 1024
SAMPLE_RATE = 16000  # what Whisper expects
FFMPEG_TIMEOUT_S = 600

log = logging.getLogger("transcribe")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="history ID of the platform run (used in log lines only)")
    parser.add_argument("--input", required=True, help="media bundle: URL, zip path or folder")
    parser.add_argument("--output", required=True, help="s3://bucket/key.jsonl or a local path")
    parser.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "large-v3"))
    parser.add_argument("--limit", type=int, default=0, help="transcribe at most N videos (testing)")
    return parser.parse_args()


def fetch_bundle(source, workdir):
    """Returns a folder holding the bundle's files."""
    if os.path.isdir(source):
        return Path(source)
    if source.startswith(("http://", "https://")):
        zip_path = workdir / "bundle.zip"
        log.info("Downloading media bundle")
        with requests.get(source, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as out:
                for chunk in response.iter_content(DOWNLOAD_CHUNK):
                    out.write(chunk)
        log.info("Downloaded %.1f MB", zip_path.stat().st_size / 1e6)
    else:
        zip_path = Path(source)
    folder = workdir / "bundle"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(folder)
    return folder


def find_videos(folder):
    """Video files, sorted, as (path, name relative to the bundle root, e.g. videos/ABC.mp4)."""
    videos = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    result = []
    for path in videos:
        parts = path.relative_to(folder).parts
        # The bundle may wrap everything in a top folder (e.g. <history id>_supporting_files/):
        # name files from their videos/ folder so they match the report's media_s3_path.
        name = "/".join(parts[parts.index("videos"):]) if "videos" in parts else "/".join(parts)
        result.append((path, name))
    return result


def load_model(name):
    try:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    device, compute_type = ("cuda", "float16") if cuda else ("cpu", "int8")
    log.info("Loading Whisper %s on %s (%s)", name, device.upper(), compute_type)
    return WhisperModel(name, device=device, compute_type=compute_type)


class NoAudio(Exception):
    pass


def load_audio(path):
    """The video's audio as 16 kHz mono float32 samples, decoded by ffmpeg.

    faster-whisper can decode files itself through PyAV, but its call breaks on
    current PyAV releases (removed `metadata_errors` option); ffmpeg avoids that.
    """
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-f", "s16le", "-"]
    run = subprocess.run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    if run.returncode != 0:
        message = run.stderr.decode(errors="replace").strip()
        if "does not contain any stream" in message or "matches no streams" in message:
            raise NoAudio("video has no audio track")
        raise RuntimeError(f"ffmpeg: {message[-300:]}")
    if not run.stdout:
        raise NoAudio("video has no audio track")
    return np.frombuffer(run.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe_one(model, path, name):
    started = time.perf_counter()
    try:
        audio = load_audio(path)
        # vad_filter skips music-only and silent stretches, which Whisper otherwise fills with made-up text.
        segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        result = {
            "file": name,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "text": text,
            "status": "Success" if text else "No speech",
            "error": "",
        }
    except NoAudio as error:
        result = {"file": name, "language": "", "language_probability": None, "duration": 0,
                  "text": "", "status": "No speech", "error": str(error)}
    except Exception as error:  # one bad file must not stop the job
        result = {"file": name, "language": "", "language_probability": None, "duration": None,
                  "text": "", "status": "Failed", "error": f"{type(error).__name__}: {error}"[:500]}
    log.info("%s: %s %s in %.1fs", name, result["status"], result["language"], time.perf_counter() - started)
    return result


def write_output(lines_path, output):
    if output.startswith("s3://"):
        import boto3

        bucket, _, key = output[len("s3://"):].partition("/")
        boto3.client("s3").upload_file(str(lines_path), bucket, key)
        log.info("Uploaded results to %s", output)
    else:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lines_path, output)
        log.info("Wrote results to %s", output)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log.info("Job %s started", args.job)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"transcribe_{args.job}_") as tmp:
        workdir = Path(tmp)
        folder = fetch_bundle(args.input, workdir)
        videos = find_videos(folder)
        if args.limit:
            videos = videos[: args.limit]
        log.info("Found %d videos", len(videos))

        lines_path = workdir / "transcripts.jsonl"
        counts = {}
        with open(lines_path, "w", encoding="utf-8") as out:
            if videos:
                model = load_model(args.model)
                for index, (path, name) in enumerate(videos, 1):
                    log.info("[%d/%d] %s", index, len(videos), name)
                    result = transcribe_one(model, path, name)
                    counts[result["status"]] = counts.get(result["status"], 0) + 1
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
        # An empty file still goes out: the crawler then reports 0 videos instead of a missing result.
        write_output(lines_path, args.output)

    # Exits 0 even when videos failed: their errors are in the results, which the crawler
    # saves as Failed rows. Only a job-level error (bundle download, S3 upload) exits non-zero.
    log.info("Job %s done in %.0fs: %s", args.job, time.perf_counter() - started, counts)


if __name__ == "__main__":
    main()
