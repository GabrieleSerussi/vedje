import sys

import numpy as np
import pytest
import torch

from vedje.video import load_video_frames, sample_indices

NUM_FRAMES_IN_VIDEO = 20


def _frame(i, h=24, w=32):
    """Frame i is a solid gray image of level 10 * i."""
    return np.full((h, w, 3), 10 * i, dtype=np.uint8)


@pytest.fixture(scope="module")
def frame_folder(tmp_path_factory):
    from PIL import Image
    folder = tmp_path_factory.mktemp("frames")
    for i in range(NUM_FRAMES_IN_VIDEO):
        Image.fromarray(_frame(i)).save(folder / f"frame_{i:03d}.png")
    (folder / "notes.txt").write_text("not a frame")
    return str(folder)


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("video") / "clip.avi"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("png", rate=10)  # lossless, so pixel values survive
        stream.width, stream.height, stream.pix_fmt = 32, 24, "rgb24"
        for i in range(NUM_FRAMES_IN_VIDEO):
            for packet in stream.encode(av.VideoFrame.from_ndarray(_frame(i), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def _levels(frames):
    return [round(float(f.mean()) * 255 / 10) for f in frames]


def test_sample_indices():
    assert sample_indices(20, 5).tolist() == [0, 4, 9, 14, 19]
    assert sample_indices(20, 5, endpoint=False).tolist() == [0, 4, 8, 12, 16]
    assert sample_indices(3, 5).tolist() == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("endpoint", [True, False])
def test_folder_of_frames_uses_the_same_index_rule(frame_folder, endpoint):
    frames = load_video_frames(frame_folder, num_frames=5, size=16, normalize=False, endpoint=endpoint)
    assert frames.shape == (5, 3, 16, 16)
    assert _levels(frames) == sample_indices(NUM_FRAMES_IN_VIDEO, 5, endpoint).tolist()


@pytest.mark.parametrize("endpoint", [True, False])
def test_pyav_reads_the_same_frame_indices(video_file, frame_folder, monkeypatch, endpoint):
    monkeypatch.setitem(sys.modules, "decord", None)  # force the PyAV path
    frames = load_video_frames(video_file, num_frames=5, size=16, normalize=False, endpoint=endpoint)
    assert _levels(frames) == sample_indices(NUM_FRAMES_IN_VIDEO, 5, endpoint).tolist()
    from_folder = load_video_frames(frame_folder, num_frames=5, size=16, normalize=False, endpoint=endpoint)
    torch.testing.assert_close(frames, from_folder)


def test_resize_crop_and_normalize(frame_folder):
    raw = load_video_frames(frame_folder, num_frames=4, size=288, normalize=False)
    assert raw.shape == (4, 3, 288, 288) and 0.0 <= raw.min() and raw.max() <= 1.0
    cropped = load_video_frames(frame_folder, num_frames=4, size=8, normalize=False, aspect_preserve=True)
    assert cropped.shape == (4, 3, 8, 8)
    normed = load_video_frames(frame_folder, num_frames=4, size=8, normalize=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    torch.testing.assert_close(normed, (load_video_frames(frame_folder, 4, size=8, normalize=False) - mean) / std)


def test_empty_folder_gives_zero_frames(tmp_path):
    frames = load_video_frames(str(tmp_path), num_frames=3, size=8)
    assert frames.shape == (3, 3, 8, 8) and frames.abs().sum() == 0
