"""
Eulerian Video Magnification — Color Amplification for Heartbeat Detection
Author: Luna Sbahtu | Arizona State University EEE515 Machine Vision

Amplifies subtle color changes in video (chrominance only) to reveal
physiological signals like heartbeats using temporal bandpass filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from scipy import signal


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
VIDEO_IN  = "my_video.mp4"
VIDEO_OUT = "my_video_magnified.mp4"
FREQ_LO   = 0.4    # Hz (24 BPM)
FREQ_HI   = 2.0    # Hz (120 BPM)
ALPHA     = 20     # Amplification factor
LEVELS    = 4      # Gaussian pyramid levels


# ─────────────────────────────────────────────
# Color Space Conversion (RGB ↔ YIQ)
# ─────────────────────────────────────────────
RGB2YIQ = np.array([
    [ 0.299,  0.587,  0.114],
    [ 0.596, -0.274, -0.322],
    [ 0.211, -0.523,  0.312]], dtype=np.float32)
YIQ2RGB = np.linalg.inv(RGB2YIQ)

def rgb2yiq(img): return img @ RGB2YIQ.T
def yiq2rgb(img): return np.clip(img @ YIQ2RGB.T, 0.0, 1.0)


# ─────────────────────────────────────────────
# Gaussian Pyramid
# ─────────────────────────────────────────────
def gauss_pyr(frame, levels):
    pyr = [frame.copy()]
    for _ in range(levels - 1):
        frame = cv2.pyrDown(frame)
        pyr.append(frame)
    return pyr


# ─────────────────────────────────────────────
# Temporal Bandpass Filter
# ─────────────────────────────────────────────
def bandpass_filter(frames_stack, fps, freq_lo, freq_hi):
    nyq = fps / 2.0
    b, a = signal.butter(2, [freq_lo / nyq, freq_hi / nyq], btype='band')
    # frames_stack: (N, H, W, C)
    padlen = 3 * max(len(a), len(b))
    return signal.filtfilt(b, a, frames_stack, axis=0, padlen=padlen)


# ─────────────────────────────────────────────
# Core: Color Magnification
# ─────────────────────────────────────────────
def color_magnify(frames, fps, freq_lo, freq_hi, alpha, levels=4):
    N = len(frames)

    # Build pyramid stack for each frame — use lowest level for efficiency
    pyr_stack = np.stack([gauss_pyr(rgb2yiq(f), levels)[-1] for f in frames])  # (N, h, w, 3)

    # Separate chrominance (I, Q channels = indices 1, 2)
    chroma = pyr_stack[..., 1:].copy()  # (N, h, w, 2)

    # Temporal bandpass filter on chrominance
    filtered = bandpass_filter(chroma, fps, freq_lo, freq_hi)

    # Amplify
    amplified = filtered * alpha

    # Reconstruct: upsample back and add to original
    H, W = frames[0].shape[:2]
    result = []
    for i, frame in enumerate(frames):
        yiq = rgb2yiq(frame).copy()

        # Upsample amplified chroma to original resolution
        amp_up = cv2.resize(amplified[i], (W, H), interpolation=cv2.INTER_LINEAR)

        yiq[..., 1] += amp_up[..., 0]
        yiq[..., 2] += amp_up[..., 1]

        result.append(yiq2rgb(yiq))

    return result


# ─────────────────────────────────────────────
# Load Video
# ─────────────────────────────────────────────
def load_video(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Cannot open {path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    print(f"Loaded {len(frames)} frames at {fps:.1f} FPS ({W}x{H})")
    return frames, fps, W, H


# ─────────────────────────────────────────────
# Save Video
# ─────────────────────────────────────────────
def save_video(frames, path, fps, W, H):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for f in frames:
        writer.write(cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved magnified video to: {path}")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def visualize(frames_orig, frames_mag):
    mid = len(frames_orig) // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(frames_orig[mid])
    axes[0].set_title("Original Frame")
    axes[0].axis("off")
    axes[1].imshow(frames_mag[mid])
    axes[1].set_title(f"Magnified Frame (α={ALPHA})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("evm_comparison.png", dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    frames, fps, W, H = load_video(VIDEO_IN)

    print(f"Running Color EVM (chrominance only)...")
    t0 = time.time()
    frames_mag = color_magnify(frames, fps, FREQ_LO, FREQ_HI, ALPHA, levels=LEVELS)
    print(f"Done in {time.time() - t0:.1f}s")

    visualize(frames, frames_mag)
    save_video(frames_mag, VIDEO_OUT, fps, W, H)
