"""Uniform frame sampling from a video file or a folder of frames.

load_video_frames decodes with decord when it is installed and with PyAV
otherwise; both read the same frame indices. A folder of image files (sorted by
name) is sampled with the same index rule.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms as T

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def sample_indices(total: int, num_frames: int, endpoint: bool = True) -> np.ndarray:
    """Indices of num_frames uniformly spaced frames out of total.

    endpoint=True spans [0, total - 1]; endpoint=False spans [0, total) as the
    public VideoPrism evaluation does.
    """
    if endpoint:
        return np.linspace(0, total - 1, num=num_frames, dtype=int)
    indices = np.linspace(0, total, num=num_frames, endpoint=False, dtype=int)
    return np.clip(indices, 0, total - 1)


def list_frame_files(folder: str) -> List[str]:
    """Image files of a folder of frames, sorted by name."""
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS))
    return [os.path.join(folder, n) for n in names]


def _read_decord(video_path: str, num_frames: int, endpoint: bool) -> Optional[torch.Tensor]:
    import decord
    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    if total == 0:
        return None
    indices = sample_indices(total, num_frames, endpoint)
    return vr.get_batch(indices)  # (T, H, W, C) uint8


def _read_pyav(video_path: str, num_frames: int, endpoint: bool) -> Optional[torch.Tensor]:
    import av

    # Frame count as decord indexes it: one frame per demuxed video packet
    # (PyAV ends the demux with an empty flush packet, which is skipped).
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total = sum(1 for packet in container.demux(stream) if packet.size > 0)
    if total == 0:
        return None
    indices = sample_indices(total, num_frames, endpoint)
    wanted = set(int(i) for i in indices)
    decoded = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted:
                decoded[i] = frame.to_ndarray(format="rgb24")
            if i >= max(wanted):
                break
    if not decoded:
        return None
    # A frame that does not decode is replaced by the nearest earlier decoded frame.
    frames, last = [], decoded[min(decoded)]
    for i in indices:
        last = decoded.get(int(i), last)
        frames.append(last)
    return torch.from_numpy(np.stack(frames))


def _read_folder(folder: str, num_frames: int, endpoint: bool) -> Optional[torch.Tensor]:
    from PIL import Image

    files = list_frame_files(folder)
    if not files:
        return None
    indices = sample_indices(len(files), num_frames, endpoint)
    frames = [np.asarray(Image.open(files[int(i)]).convert("RGB")) for i in indices]
    return torch.from_numpy(np.stack(frames))


def load_video_frames(
    video_path: str,
    num_frames: int = 8,
    size: int = 224,
    normalize: bool = True,
    aspect_preserve: bool = False,
    endpoint: bool = True,
    norm_mean: list = None,
    norm_std: list = None,
) -> torch.Tensor:
    """Decode a video and uniformly sample T frames.

    Args:
        video_path: path to an MP4/AVI/etc. file, or to a folder of frame images
        num_frames: number of frames to sample
        size: spatial resize target (square)
        normalize: if True, apply ImageNet mean/std normalization.
                   If False, keep raw [0,1] range (VideoPrism).
        norm_mean/norm_std: override the default ImageNet statistics.
        aspect_preserve: if True, center-crop to square before resizing
            (as the public DeepMind VideoPrism evaluation does). If False, stretch.
        endpoint: passed to np.linspace. The public VideoPrism evaluation uses
            endpoint=False with range [0, total); the default uses
            endpoint=True with [0, total-1].

    Returns:
        (T, C, H, W) float tensor, resized to (size, size)
    """
    if os.path.isdir(video_path):
        frames = _read_folder(video_path, num_frames, endpoint)
    else:
        try:
            import decord  # noqa: F401
            reader = _read_decord
        except ImportError:
            reader = _read_pyav
        frames = reader(video_path, num_frames, endpoint)
    if frames is None:
        return torch.zeros(num_frames, 3, size, size)

    # (T, H, W, C) -> (T, C, H, W) float [0, 1]
    frames = frames.permute(0, 3, 1, 2).float() / 255.0

    if aspect_preserve:
        # Center-crop to square (the public VideoPrism eval does this before resize).
        _, _, h, w = frames.shape
        s = min(h, w)
        y = (h - s) // 2
        x = (w - s) // 2
        frames = frames[:, :, y:y + s, x:x + s]

    resize = T.Resize((size, size), antialias=True)
    if normalize:
        transform = T.Compose([
            resize,
            T.Normalize(
                mean=norm_mean or [0.485, 0.456, 0.406],
                std=norm_std or [0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = resize

    frames = torch.stack([transform(f) for f in frames])
    return frames
