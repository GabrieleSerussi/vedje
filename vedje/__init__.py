"""VEDJE: Video-Efficient Discriminative Joint Encoder for Scalable Video-Text Retrieval.

A frozen backbone encodes each video once; a frame-indexed compressor writes an
ordered cache of a few tokens per sampled frame; a compact joint encoder reranks
first-stage candidates from these cached tokens, with the first-stage score
added as a residual prior before the score head.

Modules:
    vedje.cache      frame-indexed compressor and cache sizes (Section 3.2)
    vedje.delta      training-only future-delta predictor and L_delta (Section 3.3)
    vedje.model      the reranker, its losses and build_model (Sections 3.4 and 3.5)
    vedje.features   frozen VideoPrism-B patch features (indexing)
    vedje.video      frame sampling from video files or folders of frames
    vedje.lvt        the VideoPrism-LvT first-stage model
    vedje.retrieval  two-stage evaluation, R@1, R@5 and R@10 in both directions
    vedje.data       MSR-VTT, MSVD, DiDeMo and ActivityNet loaders
    vedje.config     configuration loading
    vedje.inference  index your videos and search them with a trained checkpoint (VEDJE, Index)
    vedje.backbone   another frozen backbone: a Backbone class and prepare, which writes its features and config
    vedje.cli        the vedje command: train, prepare, index, search
"""

__version__ = "0.1.0"
