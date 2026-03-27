import re


def egonight_doc_to_visual(doc):
    images = doc.get("images", [])
    if not isinstance(images, list):
        return []
    return images


def egonight_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    cfg = lmms_eval_specific_kwargs or {}
    pre = cfg.get("pre_prompt", "")
    post = cfg.get("post_prompt", "")
    prompt = doc.get("prompt")
    if not prompt:
        question = doc.get("question", "")
        sample_fps = doc.get("sample_fps", 1)
        prompt = (
            "Please carefully read the question and answer using the visual cues from the sampled "
            "video frames. The original FPS of the video is 30, and this frame set is sampled at "
            f"{sample_fps} fps. All actions refer to the camera wearer.\n\n"
            f"Question: {question}\n"
            "Answer briefly and directly."
        )
    return f"{pre}{prompt}{post}"


def egonight_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    visuals = egonight_doc_to_visual(doc)
    text = egonight_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    content = [{"type": "image", "url": p} for p in visuals]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def egonight_process_results(doc, results):
    pred = ""
    if isinstance(results, (list, tuple)) and results:
        pred = results[0]
    else:
        pred = results
    gt = doc.get("answer", "")
    score = 1.0 if _normalize_text(pred) == _normalize_text(gt) else 0.0
    return {"exact_match": score}


def egonight_aggregate_exact_match(items):
    if not items:
        return 0.0
    return float(sum(items)) / float(len(items))
