import os

from vlmeval.dataset.image_vqa import ImageVQADataset


def _list_frames(frame_dir):
    if not os.path.isdir(frame_dir):
        return []
    out = []
    for f in os.listdir(frame_dir):
        p = os.path.join(frame_dir, f)
        if os.path.isfile(p) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(p)
    return sorted(out)


def _sample_indices(start_idx, end_idx, max_frames):
    if end_idx < start_idx:
        end_idx = start_idx
    count = end_idx - start_idx + 1
    if count <= max_frames:
        return list(range(start_idx, end_idx + 1))
    step = max(1, count // max_frames)
    idxs = list(range(start_idx, end_idx + 1, step))
    return idxs[:max_frames]


class EgoNight(ImageVQADataset):
    DATASET_URL = {
        "EgoNight": "",
    }
    DATASET_MD5 = {}

    def __init__(self, dataset="EgoNight", max_frames=16, **kwargs):
        self.max_frames = max_frames
        super().__init__(dataset=dataset, skip_noimg=False, **kwargs)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        frame_dir = str(line["frame_dir"])
        question = str(line["question"])
        sample_fps = int(line.get("sample_fps", 1))
        start_frame = int(line.get("start_frame", 0))
        end_frame = int(line.get("end_frame", start_frame))
        frames = _list_frames(frame_dir)
        if not frames:
            return [dict(type="text", value=question)]

        start_frame = max(0, min(start_frame, len(frames) - 1))
        end_frame = max(0, min(end_frame, len(frames) - 1))
        idxs = _sample_indices(start_frame, end_frame, self.max_frames)

        prompt = (
            "Please carefully read the question and answer using the visual cues from the sampled "
            "video frames. The original FPS of the video is 30, and this frame set is sampled at "
            f"{sample_fps} fps. All actions refer to the camera wearer.\n\n"
            f"Question: {question}\n"
            "Answer briefly and directly."
        )

        msg = [dict(type="image", value=frames[i]) for i in idxs]
        msg.append(dict(type="text", value=prompt))
        return msg
