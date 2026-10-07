# VEDJE in a few minutes (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrieleSerussi/vedje/blob/main/colab/vedje_tour.ipynb)

`vedje_tour.ipynb` is a short tour of VEDJE on a free CPU runtime. It indexes one sample clip with the frozen VideoPrism encoder, then builds the frame-indexed cache with the released code and measures what it stores. It computes the feature-change targets of the training-only head and shows how the joint encoder reads a query next to the cache. It ends with the paper's results.

The release has no trained VEDJE weights, so the compressor and the VEDJE heads are freshly initialised on top of the public MiniLM checkpoint. The shapes, sizes and parameter counts it reports are those of the trained model, and it computes no retrieval scores. Training and evaluation follow the commands in the last section of the notebook and in the repository's README.

To try the first steps on your own clip, set `my_video` in the setup cell to the path or URL of a video file.

The sample clip is "Peacock walking and eating" by Mx. Granger on Wikimedia Commons, released under CC0 1.0 (https://commons.wikimedia.org/wiki/File:Peacock_walking_and_eating.webm); `sample/` holds 16 frames sampled from it.

Project page: https://gabrieleserussi.github.io/vedje/
