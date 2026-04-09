# Eulerian Video Magnification

A Python implementation of **Eulerian Video Magnification (EVM)** — a computer vision technique that amplifies subtle, invisible color changes in video to reveal physiological signals like heartbeats and breathing.

## How It Works

1. **Load video** and convert frames from BGR to RGB
2. **Detect face** using Haar Cascade to define the region of interest
3. **Convert to YIQ color space** — separates luminance (Y) from chrominance (I, Q)
4. **Build Gaussian pyramid** to spatially blur and downsample each frame
5. **Apply temporal bandpass filter** across frames to isolate the target frequency range (e.g., 0.4–2.0 Hz for heartbeats)
6. **Amplify chrominance channels only** (I and Q) — avoids brightness artifacts
7. **Reconstruct and export** the magnified video

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FREQ_LO` | 0.4 Hz | Low cutoff (≈ 24 BPM) |
| `FREQ_HI` | 2.0 Hz | High cutoff (≈ 180 BPM) |
| `ALPHA` | 20 | Amplification factor |
| `LEVELS` | 4 | Gaussian pyramid levels |

## Usage

```bash
pip install -r requirements.txt
jupyter notebook color_evm.ipynb
```

Place your input video as `my_video.mp4` in the same directory. The magnified output will be saved as `my_video_magnified.mp4`.

## Tech Stack

- Python
- OpenCV
- NumPy
- SciPy (bandpass filtering)
- Matplotlib
