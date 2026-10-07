# Third-party notices

## Code

`vedje/utils.py`: `SmoothedValue`, `MetricLogger`, the warmup and step learning-rate schedules and the distributed helpers follow the training utilities of BLIP (https://github.com/salesforce/BLIP). BLIP is released by Salesforce under the BSD-3-Clause License:

```
Copyright (c) 2022, Salesforce.com, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Salesforce.com nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Model checkpoints

The code downloads these checkpoints from the Hugging Face Hub on first use. They are not part of this repository and keep their own licences.

| Checkpoint | Used as | Licence |
| --- | --- | --- |
| `MHRDYN7/videoprism-base-f16r288` | frozen visual backbone (VideoPrism-B) | Apache License 2.0, as declared by the official VideoPrism-Base model card; the mirror page carries no additional licence declaration |
| `MHRDYN7/videoprism-lvt-base-f16r288` | first-stage retriever (VideoPrism-LvT-B) | Apache License 2.0, as declared by the official VideoPrism-LvT-B model card; the mirror page carries no additional licence declaration |
| `microsoft/MiniLM-L12-H384-uncased` | joint reranker (MiniLM-L12-H384) | MIT License |

## Datasets

MSR-VTT, MSVD, DiDeMo and ActivityNet Captions are not distributed with this repository; they are used under their original release terms.

## Notebook and project page assets

- `colab/sample/`: 16 frames sampled from "Peacock walking and eating" by Mx. Granger, Wikimedia Commons (https://commons.wikimedia.org/wiki/File:Peacock_walking_and_eating.webm), released under the CC0 1.0 Universal Public Domain Dedication.
- `docs/assets/img/logos/`: the logos of the INSIGHT Lab, Ben-Gurion University of the Negev, Texas A&M University and Decart, shown to identify the authors' affiliations. The Texas A&M mark and wordmark come from Wikimedia Commons (public domain, trademarked). The logos remain the trademarks of their owners and are not covered by the MIT License.
- `docs/assets/figures/`: Figures 1 and 2 of the paper.
