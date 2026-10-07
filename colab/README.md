# VEDJE in a few minutes (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrieleSerussi/vedje/blob/main/colab/vedje_tour.ipynb)

`vedje_tour.ipynb` runs VEDJE's indexing and times a query in about two minutes on a free CPU runtime, in four steps:

1. Set up and choose a video: the sample clip, a video you upload, or a URL or path.
2. Index it once: the frozen VideoPrism-B encoder reads its 16 sampled frames and VEDJE's compressor writes the cache, a few tokens per frame in temporal order.
3. Pick a point in the first frame and see where the frozen encoder finds the same thing in every frame.
4. Time a query on the runtime: VEDJE scoring its candidates' caches in one batch, against a typical reranker that encodes every candidate again, next to the storage per video (Table 9) and the online parameters (Appendix B).

The release has no trained VEDJE weights, so the compressor and the reranker are freshly initialised on top of the public MiniLM checkpoint. Their shapes, sizes and speed are those of the trained model, and the notebook computes no retrieval scores. Training and search follow the commands in its last section and in the repository's README.

The sample clip is "Peacock walking and eating" by Mx. Granger on Wikimedia Commons, released under CC0 1.0 (https://commons.wikimedia.org/wiki/File:Peacock_walking_and_eating.webm); `sample/` holds 16 frames sampled from it.

Project page: https://gabrieleserussi.github.io/vedje/
