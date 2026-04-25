# Enhanced Face Swap Live Video Application (Decart AI + Local Fallback)

This project provides a high-performance live video pipeline with **two transformation modes**:

1. **Reference image face swapping**
2. **Prompt-driven video transformation**

It supports Decart AI Lucy-style real-time transformation APIs and automatically falls back to local processing when cloud credentials/endpoints are unavailable.

## What is included

- Real-time webcam processing loop with queue buffering
- Face analysis with MediaPipe Face Mesh (landmarks) + Haar fallback
- Landmark-aware face swap pipeline (affine align + seamless clone)
- Prompt-based local style transforms (anime, cyberpunk, oil paint, sketch, watercolor, night scene, stylize)
- Decart adapter methods for:
  - Real-time prompt editing (`lucy_edit`)
  - Real-time reference face swap
  - Text-to-video request helper
- FPS + frame latency overlay
- Quality presets, intensity controls, recording, optional virtual camera output

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py \
  --mode face_swap \
  --quality balanced \
  --intensity 0.7 \
  --prompt "cyberpunk portrait"
```

### CLI flags

- `--camera` webcam index (default `0`)
- `--mode` `face_swap|prompt`
- `--prompt` default prompt string
- `--quality` `high|balanced|low`
- `--intensity` transformation strength `0.0..1.0`
- `--no-decart` disable Decart API calls
- `--virtual-camera` send output to virtual camera (requires `pyvirtualcam`)
- `--model` Decart model name (default `lucy-2.0`)
- `--task-prompt` Decart task for prompt mode (default `video_to_video`)
- `--task-faceswap` Decart task for face swap mode (default `real_time_video_editing`)
- `--output` recording file path

## Runtime controls

- `m` toggle mode (face swap / prompt)
- `r` load reference image path from terminal
- `t` update prompt text from terminal
- `[` / `]` decrease / increase intensity
- `1` / `2` / `3` set quality (high / balanced / low)
- `v` start/stop recording
- `g` send text-to-video request via Decart
- `q` quit

## Decart API configuration

Set environment variables:

- `DECART_API_KEY`
- `DECART_BASE_URL` (default: `https://api.decart.ai`)
- `DECART_MODEL` (optional override)
- `DECART_TIMEOUT_MS` (default: `1200`)

## Performance notes

- For <40ms/frame target latency, run with GPU-accelerated OpenCV and balanced/low quality on weaker hardware.
- MediaPipe landmarks improve face swap quality and motion consistency versus basic bbox-only blending.
- Queue buffering protects stream continuity during occasional heavy frames.
