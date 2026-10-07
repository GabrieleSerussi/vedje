import json

import torch
import torch.nn.functional as F

from vedje.data.activitynet_dataset import activitynet_retrieval_eval
from vedje.data.msrvtt_dataset import msrvtt_retrieval_eval, msrvtt_train


def test_contrastive_target_is_the_stage1_video_embedding(tmp_path):
    # Table 7: the VTC targets come from the first stage, read from the step 2a file
    ids = [f"video{i}" for i in range(3)]
    ann = [{"video_id": v, "video": f"{v}.mp4", "caption": [f"caption {i}"]} for i, v in enumerate(ids)]
    (tmp_path / "train.json").write_text(json.dumps(ann))
    for v in ids:
        torch.save({"local_patches": torch.randn(4, 8).bfloat16(), "v_global": torch.randn(8).bfloat16()},
                   tmp_path / f"{v}.pt")
    stage1 = F.normalize(torch.randn(3, 8), dim=-1)
    torch.save({"video_embeds": stage1, "text_embeds": F.normalize(torch.randn(3, 8), dim=-1),
                "video_ids": ids, "video_id_to_idx": {v: i for i, v in enumerate(ids)},
                "caption_to_video_idx": [0, 1, 2], "vid_to_caption_indices": {v: [i] for i, v in enumerate(ids)}},
               tmp_path / "stage1.pt")

    ds = msrvtt_train("", str(tmp_path / "train.json"), precomputed_dir=str(tmp_path),
                      lvt_embeds_path=str(tmp_path / "stage1.pt"))
    for i in range(3):
        torch.testing.assert_close(ds[i][1], stage1[i].bfloat16())
    # without the stage-1 file, the global feature of the cached features is kept
    ds = msrvtt_train("", str(tmp_path / "train.json"), precomputed_dir=str(tmp_path))
    torch.testing.assert_close(ds[0][1], torch.load(tmp_path / "video0.pt")["v_global"])


def test_first_stage_reads_the_captions_as_written(tmp_path):
    # the first stage gets the captions as written, as in step 2a; the joint encoder reads the cleaned text
    caps = ["A man, cooking pasta.", "Two Dogs run!"]
    (tmp_path / "msrvtt.json").write_text(json.dumps([{"video": f"video{i}.mp4", "caption": c} for i, c in enumerate(caps)]))
    (tmp_path / "anet.json").write_text(json.dumps([{"video_id": f"v_{i}", "video": f"v_{i}.mp4", "caption": c} for i, c in enumerate(caps)]))
    for ds in (msrvtt_retrieval_eval("", str(tmp_path / "msrvtt.json")), activitynet_retrieval_eval("", str(tmp_path / "anet.json"))):
        assert ds.raw_text == caps
        assert ds.text == ["a man cooking pasta", "two dogs run"]

