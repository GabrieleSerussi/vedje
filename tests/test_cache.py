import torch
import torch.nn.functional as F

from vedje.cache import FramewiseCrossAttentionCompressor, cache_bytes


def _compressor(m, frames=3, patches=5, vision_dim=12, dim=16):
    torch.manual_seed(0)
    return FramewiseCrossAttentionCompressor(
        num_queries_per_frame=m, vision_dim=vision_dim, bert_dim=dim,
        patches_per_frame=patches, max_frames=frames, num_heads=4,
    ).eval()


def test_output_shape():
    comp = _compressor(m=4)
    x = torch.randn(2, 3 * 5, 12)
    with torch.no_grad():
        z = comp(x)
    assert z.shape == (2, 3 * 4, 16)
    assert comp.num_queries == 12


def test_frame_t_occupies_rows_t_m_to_t_m_plus_m_minus_1():
    m, frames, patches = 2, 3, 5
    comp = _compressor(m=m, frames=frames, patches=patches)
    x = torch.randn(2, frames * patches, 12)
    with torch.no_grad():
        z = comp(x)
        for t in range(frames):
            frame_patches = x[:, t * patches:(t + 1) * patches]
            expected = comp._inner(frame_patches)  # this frame alone
            torch.testing.assert_close(z[:, t * m:t * m + m], expected)


def test_each_frame_is_compressed_independently():
    comp = _compressor(m=2)
    x = torch.randn(1, 15, 12)
    y = x.clone()
    y[:, 5:10] = torch.randn(1, 5, 12)  # change frame 1 only
    with torch.no_grad():
        zx, zy = comp(x), comp(y)
    torch.testing.assert_close(zx[:, 0:2], zy[:, 0:2])
    torch.testing.assert_close(zx[:, 4:6], zy[:, 4:6])
    assert not torch.allclose(zx[:, 2:4], zy[:, 2:4])


def test_cache_bytes():
    assert cache_bytes(16, 4, 384) == 49152 == 48 * 1024
    assert cache_bytes(16, 1, 384) == 12288 == 12 * 1024
    assert cache_bytes(16, 257, 768) == 6316032
    assert cache_bytes(16, 4, 384, bytes_per_element=1) == 24576


def test_cache_bytes_match_bf16_compressor_output():
    comp = FramewiseCrossAttentionCompressor(
        num_queries_per_frame=4, vision_dim=768, bert_dim=384, patches_per_frame=256, max_frames=16,
    ).to(torch.bfloat16).eval()
    with torch.inference_mode():
        z = comp(torch.randn(1, 16 * 256, 768, dtype=torch.bfloat16))[0]
    assert z.shape == (64, 384)
    assert z.numel() * z.element_size() == cache_bytes(16, 4, 384)


def test_decoder_layers_use_gelu():
    # Appendix A.2: two decoder layers, FFN width 4 d, GELU
    comp = _compressor(m=2)
    assert len(comp._inner.layers) == 2
    for layer in comp._inner.layers:
        assert layer.activation is F.gelu and layer.linear1.out_features == 4 * 16
