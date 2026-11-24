# kearney-transcribe

Speech-to-text for the Kearney Instagram report. The platform crawler
`kearney_instagram_video_transcribe` starts the EC2 instance, runs this script through
AWS SSM, reads the results from S3 and stops the instance.

```
python transcribe.py --job <history id> --input <media bundle url> --output s3://<bucket>/kearney/transcripts/<history id>/transcripts.jsonl
```

- `--input`: the run's media bundle (zip with `videos/` and `images/`). Only videos are
  transcribed. A local zip or folder works for testing.
- `--output`: one JSON line per video:
  `file`, `language`, `language_probability`, `duration`, `text`, `status` (Success, No speech, Failed), `error`.
- Model: faster-whisper `large-v3` (override with `--model` or `WHISPER_MODEL`), GPU when
  available, otherwise CPU int8. Original language is kept; the detected language is recorded.
- Audio is decoded with `ffmpeg` (16 kHz mono) and passed to Whisper as samples. faster-whisper's
  own decoding (PyAV) breaks on current PyAV releases, so `ffmpeg` must be installed.
- A video with no audio track gives `status: No speech`; one bad video never stops the job.

## EC2 instance setup

The crawler runs: `cd /home/ubuntu/kearney-transcribe && . .venv/bin/activate && python transcribe.py ...`

1. Ubuntu instance, ideally with an NVIDIA GPU (large-v3 on CPU is very slow). Install the
   NVIDIA driver plus CUDA 12 and cuDNN 9 libraries for CTranslate2.
2. IAM instance role with:
   - `AmazonSSMManagedInstanceCore` (so SSM can run the command),
   - `s3:PutObject` on the output bucket/prefix.
3. Code and environment:
   ```
   cd /home/ubuntu
   git clone <repo url> kearney-transcribe
   cd kearney-transcribe
   sudo apt-get install -y ffmpeg
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"   # caches the model once
   ```
4. Leave the instance **stopped**; the crawler starts and stops it per job.

The crawler side needs AWS credentials allowed to `ec2:DescribeInstances`, `ec2:StartInstances`,
`ec2:StopInstances`, `ssm:SendCommand`, `ssm:GetCommandInvocation`, `ssm:ListCommands`,
`ssm:CancelCommand` on this instance, and `s3:GetObject` on the output prefix.

## Local test

```
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python transcribe.py --job test --input ./sample_bundle --output ./out/transcripts.jsonl --model small
```
