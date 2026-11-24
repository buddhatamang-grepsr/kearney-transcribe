# kearney-transcribe

Speech-to-text for the Kearney Instagram report. The platform crawler
`kearney_instagram_video_transcribe` starts the EC2 instance, runs this script through
AWS SSM, reads the results from S3 and stops the instance.

```
python transcribe.py --job <history id> --input <media bundle url> --output /home/ubuntu/kearney-transcribe/output/<history id>.jsonl
```

The crawler then reads that file back through SSM (gzip + base64, in chunks that fit SSM's
24,000-character output limit), so the instance needs no IAM role or S3 bucket.

- `--input`: the run's media bundle (zip with `videos/` and `images/`). Only videos are
  transcribed. A local zip or folder works for testing.
- `--output`: a local path (what the crawler uses) or `s3://bucket/key` (needs AWS credentials). One JSON line per video:
  `file`, `language`, `language_probability`, `duration`, `text`, `status` (Success, No speech, Failed), `error`.
- Model: faster-whisper `large-v3` (override with `--model` or `WHISPER_MODEL`), GPU when
  available, otherwise CPU int8. Original language is kept; the detected language is recorded.
- GPU: the CUDA 12 libraries come from pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). The script
  restarts itself once with their folders on `LD_LIBRARY_PATH`, which CTranslate2 needs to load them.
- Audio is decoded with `ffmpeg` (16 kHz mono) and passed to Whisper as samples. faster-whisper's
  own decoding (PyAV) breaks on current PyAV releases, so `ffmpeg` must be installed.
- A video with no audio track gives `status: No speech`; one bad video never stops the job.

## EC2 instance setup

The crawler runs: `cd /home/ubuntu/kearney-transcribe && . .venv/bin/activate && python transcribe.py ...`

Current server: `i-0e124e2bd64915cf7` (us-east-1, g4dn.xlarge, Tesla T4, Ubuntu 24.04). On a T4,
large-v3 transcribes a 72 s reel in about 5 s.

1. Ubuntu instance with an NVIDIA GPU and its driver (large-v3 on CPU is very slow). The CUDA
   libraries themselves come from pip (requirements.txt).
2. SSM agent running and registered (an instance role with `AmazonSSMManagedInstanceCore`, or
   Default Host Management). No S3 permission is needed.
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
`ec2:StopInstances`, `ssm:SendCommand`, `ssm:GetCommandInvocation`, `ssm:ListCommands` and
`ssm:CancelCommand` on this instance. SSM runs the command as root; the crawler runs the script with
`sudo -u ubuntu -H` so it uses ubuntu's model cache and files stay owned by ubuntu.

## Local test

```
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python transcribe.py --job test --input ./sample_bundle --output ./out/transcripts.jsonl --model small
```
